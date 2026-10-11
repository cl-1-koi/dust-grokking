"""Portable fresh-start runner for the published CPU seed-study recipe.

Uses the historical initializer/updater unchanged, with the same update600
execution cutover. No checkpoint or backprop initialization is imported.
"""
import argparse
from dataclasses import asdict
import signal
import time
from pathlib import Path

import torch

from .seed_train import initialize, score, save_predictions
from .seed_continue import activate_cpu
from .task import dataset
from .train import update, checkpoint, disk_guard, write_json, source_hashes


def run(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    disk_guard(args.output)
    data = dataset()
    model, opt, dust = initialize(args.method, args.model_seed, args.noise_seed)
    write_json(args.output/'config.json', dict(
        method=args.method, model_seed=args.model_seed, noise_seed=args.noise_seed,
        steps=args.steps, threads=args.threads, chunk=256, cpu_cutover=600,
        dust_settings=asdict(dust.s) if dust else None,
        source_hashes=source_hashes(), inherited_backprop_steps=0,
        initialization='Fresh parameters, fresh Adam, fresh perturbation RNG',
        interpretation='Portable replay of estimator/execution schedule; hardware-dependent floating-point trajectories may differ.'))
    start = time.monotonic()
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    counters = dict(bp_forward_examples=0, eval_forward_examples=0,
                    supervised_token_presentations=0)
    elapsed_updates = 0.
    step = 0
    last_record = -1

    def record():
        nonlocal last_record
        values, predictions = score(model, data)
        counters['eval_forward_examples'] += values.pop('evaluation_forward_examples')
        values.update(step=step, method=args.method,
                      wall_seconds=time.monotonic()-start,
                      update_seconds=elapsed_updates, counters=dict(counters))
        save_predictions(args.output, step, predictions)
        checkpoint(args.output, model, opt, dust, step, counters)
        import json
        with (args.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(values)+'\n')
        print(json.dumps(values), flush=True)
        last_record = step

    record()
    while step < args.steps and not stop[0] and time.monotonic()-start < args.seconds-45:
        if dust is not None and step == 600:
            dust = activate_cpu(model, dust)
        t = time.monotonic()
        update(model, opt, dust, data, step, 256)
        elapsed_updates += time.monotonic()-t
        step += 1
        counters['supervised_token_presentations'] += 2*len(data['train'])
        if dust is None:
            counters['bp_forward_examples'] += len(data['train'])
        if step % 100 == 0 or step in (1, 5, 10, 3177):
            record()
    if last_record != step:
        record()
    write_json(args.output/'completion.json', dict(
        state='HORIZON_COMPLETE' if step == args.steps else 'BUDGET_STOP',
        step=step, wall_seconds=time.monotonic()-start,
        training_backprop_examples=counters['bp_forward_examples']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['dust', 'bp'], default='dust')
    parser.add_argument('--model-seed', type=int, default=9278700)
    parser.add_argument('--noise-seed', type=int, default=9278901)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--steps', type=int, default=3200)
    parser.add_argument('--seconds', type=int, default=71400)
    run(parser.parse_args())
