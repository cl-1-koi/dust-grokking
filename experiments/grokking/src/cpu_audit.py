"""Discarded CPU execution audit; no optimizer, backward pass or retained update."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from . import dust as dust_module
from .dust import Dust, Settings
from .model import GrokTransformer, token_losses
from .task import dataset
from .train import disk_guard, write_json
from .cpu_suffix_candidate import terminal_suffix


def fused_attention(q, k, v):
    b, t, d = q.shape
    q, k, v = [z.reshape(b, t, 4, d//4).transpose(1, 2) for z in (q, k, v)]
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=True).transpose(1, 2).reshape(b, t, d)


def supervised_losses(logits, labels):
    # The dataset check in main verifies the exact fixed sparse-target layout.
    loss = F.cross_entropy(logits[:, -2:].reshape(-1, logits.shape[-1]),
                           labels[:, -2:].reshape(-1), reduction='none')
    return F.pad(loss.reshape(labels.shape[0], 2), (labels.shape[1]-2, 0))


@contextmanager
def loss_kernel(sparse, reduce=False):
    previous = dust_module.token_losses
    previous_einsum = torch.einsum
    def weighted_reduction(equation, *values, **kwargs):
        if equation in ('nbt,nbtd->btd', 'nbth,nbthd->bthd'):
            return (values[0].unsqueeze(-1)*values[1]).sum(dim=0)
        return previous_einsum(equation, *values, **kwargs)
    dust_module.token_losses = supervised_losses if sparse else token_losses
    if reduce:
        torch.einsum = weighted_reduction
    try:
        yield
    finally:
        dust_module.token_losses = previous
        torch.einsum = previous_einsum


def fingerprint(m):
    return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest()


def fresh(kind, state=None):
    model = GrokTransformer().requires_grad_(False).eval()
    if state is not None:
        model.load_state_dict(state)
    if 'attention' in kind:
        model.attention = fused_attention
    if 'terminal' in kind:
        model.suffix = lambda site, noise: terminal_suffix(model, site, noise)
    d = Dust(model, Settings(population=16, nrep=4, direct_sigma=.002,
                            hub_last_sigma=.004, local_sigma=.05, head_sigma=.05), seed=9278901)
    return model, d


def compare(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError('Parameter set differs')
    per_parameter = {}
    for key in reference:
        a, b = reference[key].double().flatten(), candidate[key].double().flatten()
        per_parameter[key] = dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-30)),
                                  max_abs=float((a-b).abs().max()))
    a = torch.cat([v.double().flatten() for v in reference.values()])
    b = torch.cat([candidate[k].double().flatten() for k in reference])
    return dict(relative_l2=float((a-b).norm()/a.norm()),
                cosine=float(F.cosine_similarity(a, b, dim=0)), per_parameter=per_parameter)


def timed_methods(model, d):
    """Inclusive method timings; not additive across nested levels."""
    totals = {}
    def attach(obj, name, keyed=False):
        original = getattr(obj, name)
        def wrapped(*args, **kwargs):
            label = name+(':'+str(args[0]) if keyed else '')
            start = time.perf_counter()
            result = original(*args, **kwargs)
            item = totals.setdefault(label, dict(seconds=0., calls=0, elements=0))
            item['seconds'] += time.perf_counter()-start
            item['calls'] += 1
            if name == 'noise':
                item['elements'] += result.numel()
            return result
        setattr(obj, name, wrapped)
    for name in ['capture', 'direct', 'head', 'local', 'noise']:
        attach(d, name, name == 'direct')
    for name in ['suffix', 'attention', 'map_error']:
        attach(model, name, name in ('suffix', 'map_error'))
    return totals


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    disk_guard(a.output)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    data = dataset()
    assert (data['y'][:, :-2] == -1).all() and (data['y'][:, -2:] >= 0).all()
    ids = data['train'][:256]; x, y = data['x'][ids], data['y'][ids]
    report = dict(threads=a.threads, torch=torch.__version__, cpu=platform.processor(),
                  start_unix=time.time(), affinity=sorted(os.sched_getaffinity(0)),
                  source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                      ['src/cpu_audit.py', 'src/cpu_suffix_candidate.py', 'src/dust.py', 'src/model.py']},
                  parameters=455424, training_equations=len(data['train']), chunk=256,
                  population=16, draw_chunk=4, backward_passes=0, retained_updates=0,
                  timing_note='Isolated coordinator; microbatch latency is not an eight-thread worker speedup',
                  variants=[], validations=[], full_batch=[])
    kinds = a.kinds
    states = [('initialization', None)]
    if a.checkpoint:
        saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
        states.append(('dust3177', saved['model']))
        report['checkpoint_sha256'] = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    for label, state in states:
        reference = None
        for kind in kinds:
            model, d = fresh(kind, state)
            before = fingerprint(model)
            with loss_kernel('sparse_loss' in kind, 'reduce' in kind):
                logits = model(x)
                losses = dust_module.token_losses(logits, y)
                start = time.perf_counter(); g, loss = d.gradients(x, y)
                cold_seconds = time.perf_counter()-start
                if reference is None:
                    reference = (g, logits, losses, d.rng.get_state(), dict(d.counts))
                validation = dict(state=label, kind=kind, loss=loss,
                    logits_max_abs=float((logits-reference[1]).abs().max()),
                    losses_max_abs=float((losses-reference[2]).abs().max()),
                    gradients=compare(reference[0], g),
                    rng_identical=torch.equal(d.rng.get_state(), reference[3]),
                    counts_identical=d.counts == reference[4],
                    parameters_unchanged=fingerprint(model) == before,
                    no_parameter_autograd=all(p.grad is None and not p.requires_grad for p in model.parameters()))
                report['validations'].append(validation)
                times = []
                if label == 'initialization':
                    for _ in range(a.repeats):
                        start = time.perf_counter(); d.gradients(x, y)
                        times.append(time.perf_counter()-start)
                    report['variants'].append(dict(kind=kind, cold_seconds=cold_seconds, seconds=times))
                print(json.dumps(dict(state=label, kind=kind, seconds=times,
                     relative_l2=validation['gradients']['relative_l2'],
                     cosine=validation['gradients']['cosine'])), flush=True)
                write_json(a.output/'results.json', report)

    # Operator profile and inclusive wrapper times are separate from latency runs.
    if not a.no_profile:
        model, d = fresh('baseline')
        d.gradients(x, y)
        inclusive = timed_methods(model, d)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            start = time.perf_counter(); d.gradients(x, y); elapsed = time.perf_counter()-start
        events = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
        report['profile'] = dict(wall_seconds=elapsed, inclusive_methods=inclusive,
            operators=[dict(name=e.key, self_cpu_seconds=e.self_cpu_time_total/1e6,
                            total_cpu_seconds=e.cpu_time_total/1e6, calls=e.count) for e in events[:35]])
        write_json(a.output/'results.json', report)
        (a.output/'profile.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=35))

    if a.full_batch:
        for kind in [kinds[0], kinds[-1]]:
            model, d = fresh(kind)
            before = fingerprint(model)
            with loss_kernel('sparse_loss' in kind, 'reduce' in kind):
                start = time.perf_counter(); aggregated = {}
                for begin in range(0, len(data['train']), 256):
                    ix = data['train'][begin:begin+256]
                    g, _ = d.gradients(data['x'][ix], data['y'][ix])
                    for key, value in g.items():
                        value = value*(len(ix)/len(data['train']))
                        if key in aggregated: aggregated[key].add_(value)
                        else: aggregated[key] = value
                row = dict(kind=kind, seconds=time.perf_counter()-start,
                           counts=dict(d.counts), parameters_unchanged=fingerprint(model) == before)
                if kind == 'baseline': full_reference = (aggregated, d.rng.get_state())
                else:
                    row['gradients'] = compare(full_reference[0], aggregated)
                    row['rng_identical'] = torch.equal(full_reference[1], d.rng.get_state())
                report['full_batch'].append(row)
                write_json(a.output/'results.json', report)
                print(json.dumps({k:v for k,v in row.items() if k != 'gradients'}), flush=True)
    report['complete'] = True
    write_json(a.output/'results.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--full-batch', action='store_true')
    p.add_argument('--no-profile', action='store_true')
    p.add_argument('--kinds', nargs='+', default=['baseline', 'sparse_loss', 'attention', 'attention+sparse_loss'])
    main(p.parse_args())
