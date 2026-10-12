"""Exact original-Dust subbatch execution, with one sequential RNG per rank.

Each complete subbatch belongs to one rank. Unowned subbatches replay only
original noise draws. Raw subbatch gradients are broadcast and accumulated in
the serial order; populations, centering and attention targets stay local.
"""
from functools import lru_cache
import hashlib
import math
from pathlib import Path

import torch
import torch.distributed as dist

from .dust import Dust
from .model import GrokTransformer


DUST_SHA256 = '2d50560354aa2ecd9cb2f47857e0aa74f25dad949137d1e353933209e394e732'


@lru_cache(maxsize=1)
def _check_source():
    path = Path(__file__).with_name('dust.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != DUST_SHA256:
        raise ValueError('Noise replay schedule requires the validated original Dust source')


def _check(model, dust, chunk):
    _check_source()
    if type(dust) is not Dust or type(model) is not GrokTransformer or dust.model is not model:
        raise TypeError('Exact subbatch execution requires original Dust and GrokTransformer')
    if getattr(model.suffix, '__func__', None) is not GrokTransformer.suffix:
        raise TypeError('Model suffix must retain original numerical execution')
    if dust.target_average is not None:
        raise ValueError('Population collectives cannot be combined with subbatch parallelism')
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError('Chunk must be a positive integer')
    if min(dust.s.population, dust.s.nrep, dust.s.head_multiplier) < 1:
        raise ValueError('Population, nrep and head multiplier must be positive')
    if torch.get_default_dtype() != torch.float32 or any(
            p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Original Dust noise replay requires CPU float32')


def noise_shapes(dust, batch, tokens):
    """Ordered Dust.noise calls, validated against traced gradients in tests.

    This uses affine dimensions, never a clean or suffix forward. It deliberately
    mirrors direct sites, head, then each hub and its Q/K/V local calls.
    """
    if batch < 1 or tokens < 1:
        raise ValueError('Noise replay requires a nonempty subbatch')
    ref = dust.model.ref
    widths = [ref.embedding.weight.shape[1]]
    for block in ref.decoder.blocks:
        widths.extend([block.self_attn.Wo.weight.shape[0], block.ffn.ffn[0].weight.shape[0],
                       block.ffn.ffn[2].weight.shape[0], block.self_attn_norm.weight.numel(),
                       block.ffn_norm.weight.numel()])
    if len(widths) != 11 or ref.embedding.weight.shape[1] != 128:
        raise ValueError('Original two-block, width128 architecture required')
    for width in widths:
        for n in dust.chunks(dust.s.population): yield (n, batch, tokens, width)
    for n in dust.chunks(dust.s.population*dust.s.head_multiplier):
        yield (n, batch, tokens, ref.linear.weight.shape[0])
    for _ in range(2):
        for n in dust.chunks(dust.s.population): yield (n, batch, tokens, 128)
        for _ in 'qkv':
            for n in dust.chunks(dust.s.population): yield (n, batch, tokens, 128)


def _cost(dust):
    if not hasattr(dust, 'batch_parallel_counts'):
        dust.batch_parallel_counts = dict(owned_subbatches=0, replayed_subbatches=0,
            owned_noise_calls=0, owned_noise_values=0, replayed_noise_calls=0,
            replayed_noise_values=0)
    return dust.batch_parallel_counts


@torch.no_grad()
def parallel_update(model, opt, dust, data, step, chunk=256):
    """Return serial loss; dust.counts holds global logical counts on every rank.

    The caller initializes Gloo and supplies identical model/Adam/RNG/data on
    all ranks. Do not call train.update or average_gradients around this method.
    Local physical RNG replay costs are in dust.batch_parallel_counts and must
    not be added to scientific forward counters. Restore them separately.
    """
    _check(model, dust, chunk)
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    ids = data['train']
    total = int((data['y'][ids] != -1).sum())
    if len(ids) == 0 or total < 1:
        raise ValueError('Empty supervised training data')
    opt.zero_grad(set_to_none=True)
    opt.param_groups[0]['lr'] = .001*min(step/10., 1.)
    named = model.parameters_named()
    baseline = dict(dust.counts)
    costs = _cost(dust)
    chunks = []; owned = {}
    for index, start in enumerate(range(0, len(ids), chunk)):
        ix = ids[start:start+chunk]
        y = data['y'][ix]
        supervised = int((y != -1).sum())
        if supervised < 1:
            raise ValueError('Each original subbatch needs a supervised target')
        shapes = list(noise_shapes(dust, len(ix), data['x'].shape[1]))
        noise_values = sum(math.prod(shape) for shape in shapes)
        chunks.append((index % world, supervised/total))
        if index % world == rank:
            gradients, loss = dust.gradients(data['x'][ix], y)
            owned[index] = (torch.cat([gradients[n].reshape(-1) for n in named]), loss)
            costs['owned_subbatches'] += 1
            costs['owned_noise_calls'] += len(shapes)
            costs['owned_noise_values'] += noise_values
        else:
            for shape in shapes: dust.noise(*shape)
            costs['replayed_subbatches'] += 1
            costs['replayed_noise_calls'] += len(shapes)
            costs['replayed_noise_values'] += noise_values
    # Complete all original estimates before communications/optimizer mutation.
    length = sum(p.numel() for p in named.values())
    loss_sum = 0.
    for index, (owner, factor) in enumerate(chunks):
        if owner == rank:
            flat, loss = owned.pop(index)
            loss_tensor = torch.tensor(loss, dtype=torch.float64)
        else:
            flat = torch.empty(length, dtype=torch.float32)
            loss_tensor = torch.empty((), dtype=torch.float64)
        if distributed:
            dist.broadcast(flat, src=owner)
            dist.broadcast(loss_tensor, src=owner)
        loss_sum += float(loss_tensor)*factor
        offset = 0
        for name, parameter in named.items():
            size = parameter.numel()
            value = flat[offset:offset+size].reshape_as(parameter)*factor
            if parameter.grad is None: parameter.grad = value
            else: parameter.grad.add_(value)
            offset += size
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Missing or nonfinite parameter gradient')
    keys = sorted(baseline)
    counts = torch.tensor([dust.counts[k]-baseline[k] for k in keys], dtype=torch.int64)
    if distributed: dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    dust.counts = {key:baseline[key]+int(counts[i]) for i,key in enumerate(keys)}
    opt.step()
    return loss_sum
