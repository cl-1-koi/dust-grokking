"""Discarded immutable-checkpoint benchmark for original-Dust subbatches.

External runtime bounds belong to the launcher. This module never writes to an
input directory or retains a candidate for training. No backprop is invoked.
"""
import argparse
import copy
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist

from .batch_parallel import DUST_SHA256, parallel_update
from .dust import Dust, Settings
from .followup_data import build_training_data, canonical_hash
from .model import GrokTransformer
from .task import dataset
from .train import disk_guard, optimizer, update, write_json


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def state_hash(value):
    """Deterministic content hash, independent of torch.save ZIP metadata."""
    digest=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            digest.update(b'tensor');digest.update(str(item.dtype).encode())
            digest.update(json.dumps(list(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item,dict):
            digest.update(b'dict')
            for key in sorted(item,key=lambda k:(type(k).__name__,repr(k))):visit(key);visit(item[key])
            digest.update(b'enddict')
        elif isinstance(item,(list,tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:visit(child)
            digest.update(b'endsequence')
        else:
            digest.update(type(item).__name__.encode());digest.update(json.dumps(item,allow_nan=False).encode())
            digest.update(b';')
    visit(value)
    return digest.hexdigest()


def hashes(model,opt,dust,loss=None):
    out=dict(model=state_hash(model.state_dict()),optimizer=state_hash(opt.state_dict()),
             rng=state_hash(dust.rng.get_state()),counts=state_hash(dust.counts))
    if loss is not None:
        out.update(loss=state_hash(loss),gradients=state_hash({n:p.grad for n,p in model.parameters_named().items()}))
    return out


def restore_input(checkpoint,config,plan_path):
    config=json.loads(Path(config).read_text());plan=json.loads(Path(plan_path).read_text())
    if config['method']!='dust' or config.get('inherited_backprop_steps',0)!=0:
        raise ValueError('Original-Dust-only input required')
    if canonical_hash({k:v for k,v in plan.items() if k!='plan_sha256'})!=plan['plan_sha256']:
        raise ValueError('Frozen plan canonical checksum mismatch')
    if canonical_hash(plan['provenance'])!=plan['provenance_sha256']:
        raise ValueError('Frozen plan provenance checksum mismatch')
    if config['plan_file_sha256']!=file_hash(plan_path) or config['canonical_plan_sha256']!=plan['plan_sha256']:
        raise ValueError('Config points to a different frozen plan')
    if (config['model_seed'],config['noise_seed'])!=(plan['model_seed'],plan['perturbation_seed']):
        raise ValueError('Initialization/noise differ from the original paired follow-up')
    core=config['source_hashes']
    for name in ('dust.py','model.py','train.py','task.py'):
        source=Path(__file__).with_name(name)
        if file_hash(source)!=core['src/'+name]:raise ValueError('Scientific source differs: '+name)
    if core['src/dust.py']!=DUST_SHA256:raise ValueError('Validated original Dust required')
    settings=Settings(**config['dust_settings'])
    if settings.population!=16 or settings.nrep!=4 or config['chunk']!=256:
        raise ValueError('Original P16/nrep4/chunk256 treatment required')
    data=dataset()
    if plan['vocabulary']!=data['vocab'] or hashlib.sha256(data['train'].tobytes()).hexdigest()!=plan['provenance']['original_train_sha256']:
        raise ValueError('Original task split/vocabulary differs')
    data=build_training_data(data,plan['variants'][config['variant']])
    if len(data['train'])!=config['training_examples'] or hashlib.sha256(
            data['x'].numpy().tobytes()+data['y'].numpy().tobytes()).hexdigest()!=config['training_tensor_sha256']:
        raise ValueError('Original compact training tensors differ')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['counters'].get('bp_forward_examples',0)!=0 or saved.get('dust_rng_by_rank') is not None:
        raise ValueError('Input must have one original saved RNG and no BP training')
    model=GrokTransformer(seed=config['model_seed']);model.train();model.requires_grad_(False)
    model.load_state_dict(saved['model'])
    opt=optimizer(model);opt.load_state_dict(copy.deepcopy(saved['optimizer']))
    dust=Dust(model,settings,seed=config['noise_seed'])
    dust.rng.set_state(saved['dust_rng']);dust.counts=dict(saved['dust_counts'])
    return model,opt,dust,data,saved,config


def run(args):
    if not 1<=args.steps<=3 or not 1<=args.threads<=8:
        raise ValueError('Discarded audit requires 1–3 updates and 1–8 CPU threads')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    if args.mode=='parallel':dist.init_process_group('gloo',timeout=timedelta(seconds=120))
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    if args.mode=='parallel' and world!=2:raise ValueError('Two-rank benchmark required')
    inputs={str(path):file_hash(path) for path in (args.checkpoint,args.config,args.plan)}
    root=args.output/f'rank{rank}' if args.mode=='parallel' else args.output
    root.mkdir(parents=True,exist_ok=False);disk_guard(root)
    start=time.monotonic()
    model,opt,dust,data,saved,config=restore_input(args.checkpoint,args.config,args.plan)
    initial=hashes(model,opt,dust)
    if dist.is_initialized():
        preflight=dict(hashes=initial,input_contents=[inputs[str(path)] for path in
                       (args.checkpoint,args.config,args.plan)],step=int(saved['step']),settings=asdict(dust.s))
        gathered=[None]*world;dist.all_gather_object(gathered,preflight)
        if any(other!=preflight for other in gathered):
            raise AssertionError('Benchmark ranks do not start from identical frozen inputs/state')
    rows=[];counters=dict(saved['counters']);step=int(saved['step'])
    for _ in range(args.steps):
        if dist.is_initialized():dist.barrier()
        before=time.monotonic()
        loss=(parallel_update if args.mode=='parallel' else update)(model,opt,dust,data,step,256)
        seconds=time.monotonic()-before;step+=1
        counters['supervised_token_presentations']+=int((data['y'][data['train']]!=-1).sum())
        row=dict(step=step,seconds=seconds,loss=loss,hashes=hashes(model,opt,dust,loss),
                 dust_counts=dict(dust.counts))
        if dist.is_initialized():
            gathered=[None]*world;dist.all_gather_object(gathered,dict(rank=rank,seconds=seconds,hashes=row['hashes']))
            if any(member['hashes']!=gathered[0]['hashes'] for member in gathered):
                raise AssertionError('Ranks diverged in model/Adam/RNG/gradient/loss/logical counts')
            row.update(all_ranks=gathered,critical_update_seconds=max(x['seconds'] for x in gathered))
        rows.append(row)
    for path,expected in inputs.items():
        if file_hash(path)!=expected:raise AssertionError('Immutable input changed: '+path)
    disk_guard(root);temporary=root/'final-state.tmp.pt'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),dust_rng=dust.rng.get_state(),
                    dust_counts=dict(dust.counts),dust_rng_by_rank=None,dust_counts_by_rank=None,
                    step=step,counters=counters,batch_parallel_counts=getattr(dust,'batch_parallel_counts',{})),temporary)
    temporary.replace(root/'final-state.pt')
    result=dict(state='DISCARDED_AUDIT_COMPLETE',candidate_retained_for_training=False,
        mode=args.mode,rank=rank,world_size=world,threads=args.threads,chunk=256,
        population=dust.s.population,settings=asdict(dust.s),start_step=int(saved['step']),final_step=step,
        input_sha256=inputs,inputs_unchanged=True,initial_hashes=initial,final_hashes=hashes(model,opt,dust),
        rows=rows,physical_rng_replay=getattr(dust,'batch_parallel_counts',{}),
        total_seconds=time.monotonic()-start,backprop_updates=0,
        source_sha256={name:file_hash(Path(__file__).with_name(name)) for name in
                       ('batch_parallel.py','batch_parallel_audit.py','dust.py','model.py','train.py')},
        environment=dict(host=socket.gethostname(),torch=torch.__version__,affinity=sorted(os.sched_getaffinity(0))),
        timing_note='Update timings include estimator, communication, RNG replay and Adam; post-update hashes, audit collectives and file writes are excluded.')
    disk_guard(root);write_json(root/'results.json',result)
    if dist.is_initialized():
        gathered=[None]*world;dist.all_gather_object(gathered,result)
        if rank==0:
            disk_guard(args.output);write_json(args.output/'results.json',dict(result,all_rank_results=gathered))
        dist.destroy_process_group()
    print(json.dumps(dict(output=str(root),mode=args.mode,step=step,
                          update_seconds=[x['seconds'] for x in rows],hashes=result['final_hashes'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['serial','parallel'],required=True)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--threads',type=int,default=8)
    run(parser.parse_args())
