"""Checkpoint-preserving continuation with a common CPU execution cutover."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from .cpu_dust import CpuDust
from .seed_train import ARCHIVE_STEPS, initialize, score, save_predictions
from .task import dataset, hashes
from .train import disk_guard, update, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def adam_sha(opt):
    h = hashlib.sha256()
    state = opt.state_dict()
    h.update(json.dumps(state['param_groups'], sort_keys=True).encode())
    for key, values in sorted(state['state'].items()):
        h.update(str(key).encode())
        for name, value in sorted(values.items()):
            h.update(name.encode())
            h.update(value.detach().cpu().numpy().tobytes() if torch.is_tensor(value) else repr(value).encode())
    return h.hexdigest()


def equal_nested(left, right):
    if torch.is_tensor(left): return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict): return isinstance(right, dict) and left.keys() == right.keys() and all(equal_nested(v, right[k]) for k,v in left.items())
    if isinstance(left, (list, tuple)): return isinstance(right, (list, tuple)) and len(left) == len(right) and all(equal_nested(x,y) for x,y in zip(left,right))
    return left == right


def restore(saved, config):
    if config['method'] != 'dust' or config['inherited_backprop_steps'] != 0 or not config['zeroth_order_only']:
        raise ValueError('Continuation requires a pure Dust parent')
    model, opt, dust = initialize('dust', config['model_seed'], config['noise_seed'])
    if asdict(dust.s) != config['dust_settings']:
        raise ValueError('Estimator settings differ')
    model.load_state_dict(saved['model'], strict=True)
    # CPU load_state_dict can reuse the supplied tensors. Independent restores
    # must not share Adam moments with each other or the checkpoint object.
    opt.load_state_dict(copy.deepcopy(saved['optimizer']))
    dust.rng.set_state(saved['dust_rng'])
    dust.counts = dict(saved['dust_counts'])
    if saved['step'] and any(int(v['step']) != saved['step'] for v in opt.state.values()):
        raise ValueError('Adam update counters differ from checkpoint step')
    if saved['step'] and len(opt.state) != len(list(model.parameters())):
        raise ValueError('Incomplete Adam state')
    if saved['counters']['bp_forward_examples'] != 0:
        raise ValueError('Backprop counter is nonzero')
    if not equal_nested(opt.state_dict(), saved['optimizer']):
        raise ValueError('Adam state changed during restoration')
    return model, opt, dust


def activate_cpu(model, dust):
    optimized = CpuDust(model, dust.s, seed=0, target_average=dust.target_average)
    optimized.rng.set_state(dust.rng.get_state())
    optimized.counts = dict(dust.counts)
    return optimized


def run(a):
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    parent = a.parent.resolve(); output = a.output.resolve()
    config = json.loads((parent/'config.json').read_text())
    parent_status = json.loads((parent/'status.json').read_text())
    completion = json.loads((parent/'completion.json').read_text())
    if parent.with_suffix('.exit').read_text().strip() != '0' or completion['state'] != 'BUDGET_STOP':
        raise ValueError('Parent must be gracefully checkpointed and stopped')
    saved = torch.load(parent/'latest.pt', map_location='cpu', weights_only=False)
    step = int(saved['step'])
    if step != parent_status['step'] or step != completion['step'] or step > a.cutover or a.cutover % 100:
        raise ValueError('Invalid common cutover or parent step')
    if not step < config['steps'] == 3200:
        raise ValueError('Unexpected horizon')
    internal_deadline = config['start_unix']+config['seconds']
    external_deadline = config['start_unix']+72000
    if time.time() >= internal_deadline-45:
        raise ValueError('Original training budget exhausted')
    data = dataset()
    if hashlib.sha256(data['order'].tobytes()).hexdigest() != config['split_sha256']:
        raise ValueError('Split changed')
    if data['vocab'] != config['vocabulary'] or config['reference_hashes'] != hashes():
        raise ValueError('Vocabulary/reference changed')
    with np.load(parent/'split.npz') as split:
        for key, value in [('pairs', data['pairs']), ('train', data['train']), ('held', data['held']),
                           ('tokens', data['x'].numpy()), ('labels', data['y'].numpy())]:
            np.testing.assert_array_equal(split[key], value)
    for path in ('src/model.py', 'src/task.py', 'src/dust.py', 'src/train.py', 'src/seed_train.py'):
        if sha(path) != config['source_hashes'][path]:
            raise ValueError('Original training source changed: '+path)
    disk_guard(parent)
    output.mkdir(parents=True, exist_ok=False)
    for name in ['config.json', 'status.json', 'completion.json']:
        disk_guard(output)
        shutil.copy2(parent/name, output/('parent-'+name))
    for name in ['split.npz', 'metrics.jsonl', 'latest.pt']:
        disk_guard(output)
        shutil.copy2(parent/name, output/name)
    disk_guard(output)
    shutil.copy2(parent/'latest.pt', output/'resume.pt')
    disk_guard(output)
    shutil.copytree(parent/'predictions', output/'predictions')
    archives = json.loads((parent/'archives.json').read_text()) if (parent/'archives.json').exists() else []
    for entry in archives:
        disk_guard(output)
        shutil.copy2(parent/entry['file'], output/entry['file'])
        if sha(output/entry['file']) != entry['sha256']:
            raise ValueError('Parent archive checksum mismatch')
    if archives: write_json(output/'archives.json', archives)
    model, opt, dust = restore(saved, config)
    counters = dict(saved['counters']); update_seconds = parent_status['update_seconds']
    source_files = ['src/cpu_dust.py', 'src/cpu_suffix_candidate.py', 'src/seed_continue.py']
    lineage = dict(parent=str(parent), parent_checkpoint_sha256=sha(parent/'latest.pt'),
        parent_config_sha256=sha(parent/'config.json'), resume_step=step, cutover_step=a.cutover,
        original_internal_deadline=internal_deadline, original_external_deadline=external_deadline,
        original_start_unix=config['start_unix'], resumed_unix=time.time(),
        restored_adam_sha256=adam_sha(opt), restored_rng_sha256=tensor_sha(dust.rng.get_state()),
        restored_adam_exact=equal_nested(opt.state_dict(), saved['optimizer']),
        restored_rng_exact=torch.equal(dust.rng.get_state(), saved['dust_rng']),
        runtime_affinity=sorted(__import__('os').sched_getaffinity(0)), threads=torch.get_num_threads(),
        execution_source_sha256={p:sha(p) for p in source_files},
        checkpoint_model_exact=all(torch.equal(v, saved['model'][k]) for k,v in model.state_dict().items()),
        interpretation='Original estimator through common cutover; CPU execution thereafter; no optimizer/noise reset')
    config.update(execution_migration=lineage, continuation_protocol_sha256=sha('PROTOCOL.md'))
    write_json(output/'config.json', config)
    write_json(output/'resume_verification.json', lineage)
    write_json(output/'status.json', dict(parent_status, state='TRAINING', execution='original', resumed=True))
    del saved
    last_saved = step
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    recent = []
    resume_step = step; resume_compute = update_seconds; resume_time = time.monotonic()

    def archive():
        if step not in ARCHIVE_STEPS | {a.cutover} or any(e['step'] == step for e in archives): return
        disk_guard(output)
        target = output/f'step-{step:05d}.pt'
        if target.exists(): raise FileExistsError(target)
        shutil.copy2(output/'latest.pt', target)
        archives.append(dict(step=step, file=target.name, sha256=sha(target)))
        write_json(output/'archives.json', archives)

    def record():
        nonlocal last_saved
        t = time.monotonic(); row, payload = score(model, data)
        counters['eval_forward_examples'] += row.pop('evaluation_forward_examples')
        execution = 'cpu' if step > a.cutover else 'original'
        row.update(step=step, method='dust', model_seed=config['model_seed'], noise_seed=config['noise_seed'],
            population=16, wall_seconds=time.time()-config['start_unix'], update_seconds=update_seconds,
            counters=dict(counters), dust_counts=dict(dust.counts), evaluation_seconds=time.monotonic()-t,
            execution=execution, cutover_step=a.cutover,
            recent_seconds_per_update=sum(recent)/len(recent) if recent else None)
        save_predictions(output, step, payload)
        t = time.monotonic(); free = disk_guard(output)
        temporary = output/'latest.tmp.pt'
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step,
            dust_rng=dust.rng.get_state(), counters=counters, dust_counts=dict(dust.counts),
            dust_rng_by_rank=None, dust_counts_by_rank=None,
            execution=dict(path=execution, cutover_step=a.cutover, lineage=lineage)), temporary)
        temporary.replace(output/'latest.pt')
        row.update(checkpoint_seconds=time.monotonic()-t, free_disk_bytes=free)
        archive()
        with (output/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        write_json(output/'status.json', dict(state='TRAINING', **row))
        last_saved = step
        print(json.dumps(row), flush=True)

    archive()
    try:
        while step < config['steps'] and not stop[0] and time.time() < internal_deadline-45:
            if step >= a.cutover and not isinstance(dust, CpuDust):
                dust = activate_cpu(model, dust)
                recent = []
                write_json(output/'execution_switch.json', dict(step=step, unix=time.time(),
                    rng_sha256=tensor_sha(dust.rng.get_state()), adam_sha256=adam_sha(opt),
                    boundary_checkpoint_sha256=sha(output/f'step-{a.cutover:05d}.pt')))
            t = time.monotonic()
            loss = update(model, opt, dust, data, step, config['chunk'])
            dt = time.monotonic()-t; update_seconds += dt; recent = (recent+[dt])[-5:]
            step += 1
            counters['supervised_token_presentations'] += 2*len(data['train'])
            if step % config['eval_every'] == 0 or step == 3177 or step == a.cutover:
                record()
            elif step % 5 == 0:
                write_json(output/'progress.json', dict(step=step, loss_before_update=loss,
                    seconds_per_update=update_seconds/step, recent_seconds_per_update=sum(recent)/len(recent),
                    wall_seconds=time.time()-config['start_unix'], execution='cpu' if isinstance(dust, CpuDust) else 'original',
                    cutover_step=a.cutover))
    except Exception as error:
        write_json(output/'failure.json', dict(step=step, state='CONTINUATION_FAILURE', reason=repr(error)))
        raise
    if step != last_saved: record()
    write_json(output/'completion.json', dict(state='HORIZON_COMPLETE' if step == config['steps'] else 'BUDGET_STOP',
        step=step, wall_seconds=time.time()-config['start_unix'], budget_stop_is_failure=False,
        signal_received=stop[0], zeroth_order_only=True, execution_migration=lineage,
        resumed_steps=step-resume_step, resumed_compute_seconds=update_seconds-resume_compute,
        resumed_wall_seconds=time.monotonic()-resume_time))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cutover', type=int, required=True)
    run(p.parse_args())
