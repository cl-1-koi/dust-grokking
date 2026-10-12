"""Fresh-student subset/reconstruction controls, isolated from seed replicas."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from .followup_data import build_training_data, canonical_hash
from .model import token_losses
from .seed_train import ARCHIVE_STEPS, initialize, save_predictions, score
from .task import dataset
from .train import checkpoint, disk_guard, source_hashes, update, write_json


@torch.no_grad()
def evaluate_student(model, truth_data, training_data, plan):
    original, payload = score(model, truth_data)
    metrics = {}
    for key, value in original.items():
        if key.startswith(('train_', 'heldout_', 'unseen_pair_', 'reversed_train_')):
            key = 'original_'+key
        metrics[key] = value
    predicted = payload['predicted_token']
    teacher = np.asarray(plan['teacher_predictions'])
    for label in ('S', 'R', 'H', 'U', 'clean_eval', 'P'):
        ids = np.asarray(plan['partitions'][label], dtype=np.int64)
        metrics[label+'_count'] = len(ids)
        metrics[label+'_correct'] = int(payload['correct'][ids].sum())
        metrics[label+'_accuracy'] = float(payload['correct'][ids].mean())
        metrics[label+'_answer_ce'] = float(payload['answer_ce'][ids].mean())
        metrics[label+'_margin'] = float(payload['margin'][ids].mean())
        metrics[label+'_teacher_agreement'] = float((predicted[ids] == teacher[ids]).mean())
    losses, correct = [], []
    train_ids = training_data['train']
    for begin in range(0, len(train_ids), 512):
        ids = train_ids[begin:begin+512]
        y = training_data['y'][ids]
        z = model(training_data['x'][ids])
        losses.append(token_losses(z, y)[:, 4:6])
        correct.append(z[:, 4].argmax(-1) == y[:, 4])
    loss = torch.cat(losses)
    metrics.update(fit_count=len(train_ids), fit_accuracy=float(torch.cat(correct).float().mean()),
                   fit_rhs_loss=float(loss.mean()), fit_answer_loss=float(loss[:, 0].mean()))
    metrics['evaluation_forward_examples'] += len(train_ids)
    return metrics, payload


def run(a):
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    disk_guard(a.output)
    start = time.monotonic()
    plan = json.loads(a.plan.read_text())
    canonical = {key: value for key, value in plan.items() if key != 'plan_sha256'}
    if canonical_hash(canonical) != plan['plan_sha256']:
        raise ValueError('Data plan canonical hash mismatch')
    if canonical_hash(plan['provenance']) != plan['provenance_sha256']:
        raise ValueError('Data plan provenance hash mismatch')
    if (a.model_seed, a.noise_seed) != (plan['model_seed'], plan['perturbation_seed']):
        raise ValueError('Follow-up initialization/noise differ from the frozen paired plan')
    variant = plan['variants'][a.variant]
    truth_data = dataset()
    if plan['vocabulary'] != truth_data['vocab'] or hashlib.sha256(truth_data['train'].tobytes()).hexdigest() != plan['provenance']['original_train_sha256']:
        raise ValueError('Data plan vocabulary or original split mismatch')
    training_data = build_training_data(truth_data, variant)
    model, opt, dust = initialize(a.method, a.model_seed, a.noise_seed)
    config = dict(method=a.method, variant=a.variant, model_seed=a.model_seed, noise_seed=a.noise_seed,
                  plan_path=str(a.plan), plan_file_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
                  canonical_plan_sha256=plan['plan_sha256'],
                  label_source=variant['label_source'], training_examples=len(training_data['train']),
                  primary_scope='Original fixed U for subset; H reconstruction and clean_eval for pseudo-label arms',
                  ordinary_original_held_contaminated=a.variant not in ('subset75', 'subset50'),
                  teacher_trained_on_H=True, student_trained_on_H=False,
                  inherited_backprop_steps=0, inherited_teacher_weights=False,
                  initial_parameter_sha256=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest(),
                  training_tensor_sha256=hashlib.sha256(training_data['x'].numpy().tobytes()+training_data['y'].numpy().tobytes()).hexdigest(),
                  steps=a.steps, seconds=a.seconds, threads=a.threads, chunk=a.chunk,
                  zeroth_order_only=dust is not None, dust_settings=dust.metadata()['settings'] if dust else None,
                  optimizer={k: opt.param_groups[0][k] for k in ('lr', 'betas', 'eps', 'weight_decay')},
                  source_hashes=source_hashes(), start_unix=time.time(),
                  protocol_sha256=hashlib.sha256(Path('PROTOCOL.md').read_bytes()).hexdigest())
    write_json(a.output/'config.json', config)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    counters = dict(bp_forward_examples=0, eval_forward_examples=0, supervised_token_presentations=0)
    step = 0
    last_saved = -1
    update_seconds = 0.

    def record():
        nonlocal last_saved
        row, payload = evaluate_student(model, truth_data, training_data, plan)
        counters['eval_forward_examples'] += row.pop('evaluation_forward_examples')
        row.update(step=step, method=a.method, variant=a.variant, wall_seconds=time.monotonic()-start,
                   update_seconds=update_seconds, counters=dict(counters),
                   dust_counts=dict(dust.counts) if dust else None)
        save_predictions(a.output, step, payload)
        row.update(checkpoint(a.output, model, opt, dust, step, counters))
        if step in ARCHIVE_STEPS:
            disk_guard(a.output)
            destination = a.output/f'step-{step:05d}.pt'
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copy2(a.output/'latest.pt', destination)
        with (a.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        write_json(a.output/'status.json', dict(state='TRAINING', **row))
        last_saved = step
        print(row, flush=True)

    record()
    try:
        while step < a.steps and not stop[0] and time.monotonic()-start < a.seconds-45:
            t = time.monotonic()
            loss = update(model, opt, dust, training_data, step, a.chunk)
            update_seconds += time.monotonic()-t
            step += 1
            counters['supervised_token_presentations'] += 2*len(training_data['train'])
            if dust is None:
                counters['bp_forward_examples'] += len(training_data['train'])
            if step % a.eval_every == 0 or step in (1, 5, 10, 3177):
                record()
            elif step % 5 == 0:
                write_json(a.output/'progress.json', dict(step=step, loss_before_update=loss,
                           seconds_per_update=update_seconds/step, wall_seconds=time.monotonic()-start))
    except (FloatingPointError, AssertionError) as error:
        write_json(a.output/'failure.json', dict(step=step, state='NUMERICAL_STOP', reason=str(error)))
        raise
    if last_saved != step:
        record()
    write_json(a.output/'completion.json', dict(state='HORIZON_COMPLETE' if step == a.steps else 'BUDGET_STOP',
               step=step, wall_seconds=time.monotonic()-start, budget_stop_is_failure=False,
               zeroth_order_only=dust is not None, signal_received=stop[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--method', choices=['dust', 'bp'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-seed', type=int, default=9278700)
    parser.add_argument('--noise-seed', type=int, default=9278902)
    parser.add_argument('--steps', type=int, default=3200)
    parser.add_argument('--seconds', type=int, default=71400)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--chunk', type=int, default=256)
    parser.add_argument('--eval-every', type=int, default=100)
    run(parser.parse_args())
