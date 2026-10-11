"""Fixed-split initialization x perturbation replicas, using the tested updater."""
import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import shutil
import signal
import socket
import time

import numpy as np
import torch

from .dust import Dust, Settings
from .model import GrokTransformer, token_losses
from .task import dataset, hashes
from .train import checkpoint, disk_guard, optimizer, source_hashes, update, write_json

ARCHIVE_STEPS = {1000, 2000, 3000, 3177, 3200, 6800, 10000}


def initialize(method, model_seed, noise_seed):
    model = GrokTransformer(seed=model_seed)
    model.train()
    settings = Settings(population=16, nrep=4, direct_sigma=.002,
                        hub_last_sigma=.004, local_sigma=.05, head_sigma=.05)
    dust = Dust(model, settings, seed=noise_seed) if method == 'dust' else None
    if dust is not None:
        model.requires_grad_(False)
    return model, optimizer(model), dust


def masks(data):
    pairs = np.asarray(data['pairs'])
    lookup = {tuple(p): i for i, p in enumerate(pairs)}
    reverse = np.array([lookup[(int(b), int(a))] for a, b in pairs])
    train = np.zeros(len(pairs), dtype=bool)
    train[data['train']] = True
    return train, (~train) & (~train[reverse]), (~train) & train[reverse]


@torch.no_grad()
def score(model, data, chunk=512):
    losses, predictions, margins = [], [], []
    for begin in range(0, len(data['x']), chunk):
        y = data['y'][begin:begin+chunk]
        logits = model(data['x'][begin:begin+chunk])
        losses.append(token_losses(logits, y)[:, -2:])
        predictions.append(logits[:, -2:].argmax(-1))
        z = logits[:, 4].clone()
        truth = z.gather(1, y[:, 4, None]).squeeze(1)
        z.scatter_(1, y[:, 4, None], -torch.inf)
        margins.append(truth-z.max(-1).values)
    loss = torch.cat(losses)
    pred = torch.cat(predictions)
    correct = pred == data['y'][:, -2:]
    train, unseen, reversed_train = masks(data)
    metrics = {}
    for label, ids in [('train', data['train']), ('heldout', data['held'])]:
        metrics[label+'_accuracy'] = float(correct[ids, 0].float().mean())
        metrics[label+'_rhs_accuracy'] = float(correct[ids].all(-1).float().mean())
        metrics[label+'_loss'] = float(loss[ids].mean())
        metrics[label+'_product_loss'] = float(loss[ids, 0].mean())
        metrics[label+'_eos_loss'] = float(loss[ids, 1].mean())
        metrics[label+'_eos_accuracy'] = float(correct[ids, 1].float().mean())
    for label, selected in [('unseen_pair', unseen), ('reversed_train', reversed_train)]:
        count = int(selected.sum())
        total = int(correct[selected, 0].sum())
        metrics[label+'_count'] = count
        metrics[label+'_correct'] = total
        metrics[label+'_accuracy'] = total/count if count else None
        metrics[label+'_answer_loss'] = float(loss[selected, 0].mean()) if count else None
    metrics['weight_norm'] = sum(float(p.double().square().sum()) for p in model.parameters())**.5
    metrics['evaluation_forward_examples'] = len(data['x'])
    payload = dict(predicted_token=pred[:, 0].numpy(), correct=correct[:, 0].numpy(),
                   answer_ce=loss[:, 0].numpy(), margin=torch.cat(margins).numpy(),
                   train=train, unseen_pair=unseen, reversed_train=reversed_train,
                   pairs=data['pairs'], true_residue=data['pairs'].sum(1) % 97)
    return metrics, payload


def save_predictions(root, step, payload):
    disk_guard(root)
    destination = root/'predictions'/f'step-{step:05d}.npz'
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, step=np.array(step), **payload)
    temporary.replace(destination)


def run(a):
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    disk_guard(a.output)
    start = time.monotonic()
    data = dataset()
    model, opt, dust = initialize(a.method, a.model_seed, a.noise_seed)
    initial_hash = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    config = dict(method=a.method, model_seed=a.model_seed, noise_seed=a.noise_seed,
                  initial_parameter_sha256=initial_hash, split_seed=0,
                  split_sha256=hashlib.sha256(data['order'].tobytes()).hexdigest(),
                  train_pairs=len(data['train']), held_pairs=len(data['held']),
                  unseen_pairs=int(masks(data)[1].sum()), vocabulary=data['vocab'],
                  steps=a.steps, seconds=a.seconds, threads=a.threads, chunk=a.chunk,
                  eval_every=a.eval_every, special_eval_steps=[1, 5, 10, 3177],
                  archive_steps=sorted(ARCHIVE_STEPS), inherited_backprop_steps=0,
                  zeroth_order_only=a.method == 'dust', initialization='fresh model and fresh Adam state',
                  dust_settings=asdict(dust.s) if dust else None,
                  optimizer={k: opt.param_groups[0][k] for k in ('lr', 'betas', 'eps', 'weight_decay')},
                  warmup_steps=10, source_hashes=source_hashes(), reference_hashes=hashes(),
                  protocol_sha256=hashlib.sha256(Path('PROTOCOL.md').read_bytes()).hexdigest(),
                  environment=dict(host=socket.gethostname(), torch=torch.__version__,
                                   affinity=sorted(os.sched_getaffinity(0))), start_unix=time.time())
    write_json(a.output/'config.json', config)
    np.savez_compressed(a.output/'split.npz', train=data['train'], held=data['held'],
                        pairs=data['pairs'], tokens=data['x'].numpy(), labels=data['y'].numpy())
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    counters = dict(bp_forward_examples=0, eval_forward_examples=0, supervised_token_presentations=0)
    step = 0
    last_saved = -1
    update_seconds = 0.
    archives = []

    def record():
        nonlocal last_saved
        t = time.monotonic()
        row, payload = score(model, data)
        counters['eval_forward_examples'] += row.pop('evaluation_forward_examples')
        row.update(step=step, method=a.method, model_seed=a.model_seed, noise_seed=a.noise_seed,
                   population=16 if dust else 0, wall_seconds=time.monotonic()-start,
                   update_seconds=update_seconds, counters=dict(counters),
                   dust_counts=dict(dust.counts) if dust else None, evaluation_seconds=time.monotonic()-t)
        save_predictions(a.output, step, payload)
        row.update(checkpoint(a.output, model, opt, dust, step, counters))
        if step in ARCHIVE_STEPS:
            disk_guard(a.output)
            archived = a.output/f'step-{step:05d}.pt'
            if archived.exists():
                raise FileExistsError(archived)
            shutil.copy2(a.output/'latest.pt', archived)
            archives.append(dict(step=step, file=archived.name,
                                 sha256=hashlib.sha256(archived.read_bytes()).hexdigest()))
            write_json(a.output/'archives.json', archives)
        with (a.output/'metrics.jsonl').open('a') as stream:
            import json
            stream.write(json.dumps(row)+'\n')
        write_json(a.output/'status.json', dict(state='TRAINING', **row))
        last_saved = step
        print(row, flush=True)

    record()
    try:
        while step < a.steps and not stop[0] and time.monotonic()-start < a.seconds-45:
            t = time.monotonic()
            loss = update(model, opt, dust, data, step, a.chunk)
            update_seconds += time.monotonic()-t
            step += 1
            counters['supervised_token_presentations'] += 2*len(data['train'])
            if dust is None:
                counters['bp_forward_examples'] += len(data['train'])
            if step % a.eval_every == 0 or step in (1, 5, 10, 3177):
                record()
            elif step % 5 == 0:
                write_json(a.output/'progress.json', dict(step=step, loss_before_update=loss,
                           seconds_per_update=update_seconds/step, wall_seconds=time.monotonic()-start))
    except (FloatingPointError, AssertionError) as error:
        write_json(a.output/'failure.json', dict(step=step, state='NUMERICAL_STOP', reason=str(error)))
        raise
    if step != last_saved:
        record()
    write_json(a.output/'completion.json', dict(state='HORIZON_COMPLETE' if step == a.steps else 'BUDGET_STOP',
               step=step, wall_seconds=time.monotonic()-start, budget_stop_is_failure=False,
               signal_received=stop[0], zeroth_order_only=dust is not None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['dust', 'bp'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-seed', type=int, required=True)
    parser.add_argument('--noise-seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=3200)
    parser.add_argument('--seconds', type=int, default=71400)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--chunk', type=int, default=256)
    parser.add_argument('--eval-every', type=int, default=100)
    run(parser.parse_args())
