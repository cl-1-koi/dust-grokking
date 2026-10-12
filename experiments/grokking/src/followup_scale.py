"""Resume existing follow-ups with exact complete-sub-batch CPU parallelism."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch
import torch.distributed as dist

from .batch_parallel import parallel_update
from .collectives import any_stop
from .followup_data import build_training_data, canonical_hash
from .followup_train import evaluate_student
from .seed_continue import restore, adam_sha, tensor_sha, equal_nested
from .seed_train import ARCHIVE_STEPS, save_predictions
from .task import dataset
from .train import checkpoint, disk_guard, source_hashes, update, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_sha(model):
    return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()


def load_parent(root):
    root = Path(root)
    receipt = json.loads((root/'manifest.json').read_text())
    required = {'config.json', 'status.json', 'completion.json', 'latest.pt',
                'exit.txt', 'plan.json', 'metrics.jsonl'}
    required.update(str(p.relative_to(root)) for p in (root/'predictions').glob('*.npz'))
    if not required <= set(receipt['files']) or not (root/'predictions').is_dir():
        raise ValueError('Parent manifest omits required checkpoint/history files')
    for name, digest in receipt['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Parent manifest paths must stay inside its package')
        if sha(root/name) != digest:
            raise ValueError('Immutable parent package changed: '+name)
    config = json.loads((root/'config.json').read_text())
    status = json.loads((root/'status.json').read_text())
    completion = json.loads((root/'completion.json').read_text())
    if (root/'exit.txt').read_text().strip() != '0' or completion['state'] != 'BUDGET_STOP' \
       or not completion['signal_received']:
        raise ValueError('Parent must be gracefully checkpointed at a migration boundary')
    saved = torch.load(root/'latest.pt', map_location='cpu', weights_only=False)
    step = int(saved['step'])
    if step != status['step'] or step != completion['step'] or not step < config['steps'] == 3200:
        raise ValueError('Parent step/horizon mismatch')
    if config['threads'] != 8 or config['chunk'] != 256 or config['method'] != 'dust':
        raise ValueError('Unexpected original execution recipe')
    if config['inherited_teacher_weights'] or not config['zeroth_order_only']:
        raise ValueError('Parent is not a fresh pure-Dust follow-up')
    for name in ('src/model.py', 'src/task.py', 'src/dust.py', 'src/train.py',
                 'src/seed_train.py', 'src/followup_train.py'):
        if sha(name) != config['source_hashes'][name]:
            raise ValueError('Original estimator/training source changed: '+name)
    plan = json.loads((root/'plan.json').read_text())
    if sha(root/'plan.json') != config['plan_file_sha256'] or \
       plan['plan_sha256'] != config['canonical_plan_sha256'] or \
       canonical_hash({k:v for k,v in plan.items() if k != 'plan_sha256'}) != plan['plan_sha256'] or \
       canonical_hash(plan['provenance']) != plan['provenance_sha256']:
        raise ValueError('Frozen data plan changed')
    truth = dataset()
    if plan['vocabulary'] != truth['vocab'] or \
       hashlib.sha256(truth['train'].tobytes()).hexdigest() != plan['provenance']['original_train_sha256']:
        raise ValueError('Original vocabulary or split changed')
    training = build_training_data(truth, plan['variants'][config['variant']])
    if hashlib.sha256(training['x'].numpy().tobytes()+training['y'].numpy().tobytes()).hexdigest() != config['training_tensor_sha256']:
        raise ValueError('Training equations changed')
    if len(training['train']) != config['training_examples']:
        raise ValueError('Training sample count changed')
    return config, status, saved, plan, truth, training, receipt


def require_shared_state(model, opt, dust, distributed):
    state = dict(model_sha256=model_sha(model), adam_sha256=adam_sha(opt),
                 rng_sha256=tensor_sha(dust.rng.get_state()), logical_counts=dict(dust.counts))
    if distributed:
        gathered = [None]*dist.get_world_size()
        dist.all_gather_object(gathered, state)
        if any(other != state for other in gathered):
            raise AssertionError('Ranks diverged in parameters/Adam/RNG/logical counts')
    return state


def run(args):
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if distributed:
        dist.init_process_group('gloo', timeout=timedelta(seconds=120))
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    if world not in (1, 2): raise ValueError('Only the audited one/two-rank execution is allowed')
    config, parent_status, saved, plan, truth, training, receipt = load_parent(args.inputs)
    if distributed:
        packages = [None]*world
        dist.all_gather_object(packages, sha(args.inputs/'manifest.json'))
        if len(set(packages)) != 1:
            raise ValueError('Ranks received different immutable parent packages')
    deadline = float(receipt.get('original_internal_deadline', config['start_unix']+config['seconds']))
    external_deadline = float(receipt.get('original_external_deadline', config['start_unix']+72000))
    if not all(math.isfinite(float(v)) for v in (deadline, external_deadline, config['start_unix'], config['seconds'])):
        raise ValueError('Runtime bounds must be finite')
    if deadline > config['start_unix']+config['seconds'] or external_deadline > config['start_unix']+72000:
        raise ValueError('Migration must not extend original deadlines')
    if time.time() >= min(deadline-180, external_deadline-180):
        raise ValueError('Original absolute runtime budget exhausted')
    output = args.output/f'rank{rank}'
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True); disk_guard(output.parent)
    output.mkdir(exist_ok=False)
    model, opt, dust = restore(saved, config)
    if type(dust).__name__ != 'Dust' or dust.target_average is not None:
        raise ValueError('Preserve original serial Dust and local sub-batch targets')
    if any(p.requires_grad for p in model.parameters()): raise AssertionError('Backprop must be disabled')
    assert equal_nested(model.state_dict(), saved['model'])
    step = int(saved['step']); resume_step = step
    counters = dict(saved['counters']); update_seconds = parent_status['update_seconds']
    initial = require_shared_state(model, opt, dust, distributed)
    migration = dict(parent_package_sha256=sha(args.inputs/'manifest.json'),
        parent_checkpoint_sha256=receipt['files']['latest.pt'],
        parent_config_sha256=receipt['files']['config.json'], resume_step=step,
        resumed_unix=time.time(), original_internal_deadline=deadline,
        original_external_deadline=external_deadline,
        rank=rank, world_size=world, threads_per_rank=8,
        restored_state=initial, model_restored_exactly=True, adam_restored_exactly=True,
        rng_restored_exactly=True, unchanged_estimator='original Dust, P16, original chunk order and noise stream',
        gradient_communication='broadcast each original chunk estimate, then identical ordered weighted accumulation',
        execution_source_hashes=source_hashes())
    config = dict(config, execution_migration=migration,
                  continuation_protocol_sha256=sha('PROTOCOL.md'))
    write_json(output/'config.json', config)
    write_json(output/'resume_verification.json', migration)
    if rank == 0:
        disk_guard(output)
        shutil.copy2(args.inputs/'metrics.jsonl', output/'metrics.jsonl')
        # Inputs are sealed read-only; only these new continuation destinations
        # must accept append/new files. Keep the original package unchanged.
        (output/'metrics.jsonl').chmod(0o644)
        shutil.copytree(args.inputs/'predictions', output/'predictions')
        (output/'predictions').chmod(0o755)
        shutil.copy2(args.inputs/'latest.pt', output/'resume.pt')
        shutil.copy2(args.inputs/'latest.pt', output/'latest.pt')
        for name in ('config.json', 'status.json', 'completion.json'):
            shutil.copy2(args.inputs/name, output/('parent-'+name))
        write_json(output/'status.json', dict(parent_status, state='TRAINING', execution='parallel_subbatches'))
    del saved
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    last_saved = step; recent = []; resumed_at = time.monotonic()

    def record():
        nonlocal last_saved, counters
        shared = require_shared_state(model, opt, dust, distributed)
        write_json(output/'execution_counts.json', dict(rank=rank, step=step,
                   logical_counts=dict(dust.counts),
                   local_parallel_execution=getattr(dust, 'batch_parallel_counts', {})))
        if rank == 0:
            row, payload = evaluate_student(model, truth, training, plan)
            counters['eval_forward_examples'] += row.pop('evaluation_forward_examples')
            row.update(step=step, method='dust', variant=config['variant'],
                       wall_seconds=time.time()-config['start_unix'], update_seconds=update_seconds,
                       counters=dict(counters), dust_counts=dict(dust.counts),
                       execution='parallel_subbatches', world_size=world, resume_step=resume_step,
                       recent_seconds_per_update=sum(recent)/len(recent) if recent else None,
                       shared_state=shared)
            row['rank0_parallel_execution_counts'] = getattr(dust, 'batch_parallel_counts', {})
            save_predictions(output, step, payload)
            row.update(checkpoint(output, model, opt, dust, step, counters))
            if step in ARCHIVE_STEPS:
                disk_guard(output)
                shutil.copy2(output/'latest.pt', output/f'step-{step:05d}.pt')
            with (output/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            write_json(output/'status.json', dict(state='TRAINING', **row))
            print(json.dumps(row), flush=True)
        if distributed:
            shared_counters = [counters if rank == 0 else None]
            dist.broadcast_object_list(shared_counters, src=0); counters = shared_counters[0]
        last_saved = step

    try:
        while step < config['steps']:
            stop_now = stop[0] or time.time() >= deadline-180
            if distributed: stop_now = any_stop(stop_now)
            if stop_now: break
            start = time.monotonic()
            loss = (parallel_update(model, opt, dust, training, step, config['chunk']) if distributed
                    else update(model, opt, dust, training, step, config['chunk']))
            seconds = time.monotonic()-start
            recent.append(seconds); recent[:] = recent[-20:]
            update_seconds += seconds; step += 1
            counters['supervised_token_presentations'] += 2*len(training['train'])
            if step % 100 == 0 or step == 3177 or step in (resume_step+1, resume_step+5): record()
            if step % 5 == 0 or step == resume_step+1:
                write_json(output/'progress.json', dict(step=step, loss_before_update=loss,
                    seconds_per_update=sum(recent)/len(recent), seconds_since_resume=time.monotonic()-resumed_at,
                    rank=rank, world_size=world, original_internal_deadline=deadline))
        if step != last_saved: record()
        for name, digest in receipt['files'].items():
            if sha(args.inputs/name) != digest: raise ValueError('Input package changed during run')
        write_json(output/'completion.json', dict(state='HORIZON_COMPLETE' if step == config['steps'] else 'BUDGET_STOP',
            step=step, wall_seconds=time.time()-config['start_unix'], seconds_since_resume=time.monotonic()-resumed_at,
            budget_stop_is_failure=False, zeroth_order_only=True, signal_received=stop[0],
            original_deadlines_preserved=True, rank=rank, world_size=world, inputs_unchanged=True))
    except Exception as error:
        write_json(output/'failure.json', dict(state='SCALE_OUT_FAILURE', step=step, error=repr(error)))
        raise
    finally:
        if distributed: dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-inputs', action='store_true')
    args = parser.parse_args()
    if args.verify_inputs:
        torch.set_num_threads(1)
        config, status, *_ = load_parent(args.inputs)
        print(json.dumps(dict(state='PARENT_VERIFIED', variant=config['variant'], step=status['step'])))
    else:
        if args.output is None: parser.error('--output is required')
        run(args)
