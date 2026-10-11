"""CPU population parallelism follows pinned Dust: same batch, independent draws.

Hub errors are averaged BEFORE local Q/K/V credit; final parameter estimates
are averaged before AdamW. This avoids independently noisy per-rank hubs.
"""
import torch
import torch.distributed as dist

def average_tensor(value):
    dist.all_reduce(value,op=dist.ReduceOp.SUM)
    return value.div_(dist.get_world_size())

@torch.no_grad()
def average_gradients(model):
    params=list(model.parameters())
    flat=torch.cat([p.grad.reshape(-1) for p in params])
    average_tensor(flat)
    start=0
    for p in params:
        size=p.numel();p.grad.copy_(flat[start:start+size].reshape_as(p));start+=size

def sum_counts(counts):
    keys=sorted(counts);v=torch.tensor([counts[k] for k in keys],dtype=torch.int64)
    dist.all_reduce(v,op=dist.ReduceOp.SUM)
    return dict(zip(keys,v.tolist()))

def any_stop(value):
    flag=torch.tensor(int(value));dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    return bool(flag)
