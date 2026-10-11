"""Bounded CPU full-batch Dust and identically configured backprop control."""
import argparse
from dataclasses import asdict
import hashlib
from datetime import timedelta
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import time
import numpy as np
import torch
import torch.distributed as dist
from .task import dataset,hashes,SEED,ROOT
from .model import GrokTransformer,token_losses
from .dust import Dust,Settings
from .collectives import average_tensor,average_gradients,sum_counts,any_stop

MIN_FREE=5*1024**3
stopping=False

def disk_guard(path):
    free=shutil.disk_usage(path).free
    if free<MIN_FREE: raise OSError(f'Checkpoint refused: free bytes {free} below {MIN_FREE}')
    return free

def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def optimizer(m):
    return torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.98),eps=1e-8,
                             weight_decay=1.,foreach=False)

def update(m,opt,d,data,step,chunk):
    ids=data['train']; total=int((data['y'][ids]!=-1).sum()); opt.zero_grad(set_to_none=True)
    opt.param_groups[0]['lr']=.001*min(step/10.,1.)
    loss_sum=0.
    if d is None:
        for start in range(0,len(ids),chunk):
            ix=ids[start:start+chunk];x=data['x'][ix];y=data['y'][ix]
            loss=token_losses(m(x),y).sum()/total
            loss.backward();loss_sum+=float(loss.detach())
    else:
        named=m.parameters_named()
        for start in range(0,len(ids),chunk):
            ix=ids[start:start+chunk];x=data['x'][ix];y=data['y'][ix]
            g,loss=d.gradients(x,y);factor=int((y!=-1).sum())/total;loss_sum+=loss*factor
            for n,p in named.items():
                value=g[n]*factor
                if p.grad is None:p.grad=value
                else:p.grad.add_(value)
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):
        raise FloatingPointError('Missing or nonfinite parameter gradient')
    if dist.is_initialized():average_gradients(m)
    opt.step();return loss_sum

@torch.no_grad()
def evaluate(m,data,chunk=512):
    loss=[];pred=[]
    for start in range(0,len(data['x']),chunk):
        x=data['x'][start:start+chunk];y=data['y'][start:start+chunk]
        z=m(x);pred.append(z[:,-2:].argmax(-1));loss.append(token_losses(z,y)[:,-2:])
    pred=torch.cat(pred);loss=torch.cat(loss);ok=pred==data['y'][:,-2:]
    out={}
    for name,ids in [('train',data['train']),('heldout',data['held'])]:
        out[name+'_accuracy']=float(ok[ids,0].float().mean())
        out[name+'_rhs_accuracy']=float(ok[ids].all(-1).float().mean())
        out[name+'_loss']=float(loss[ids].mean())
        out[name+'_product_loss']=float(loss[ids,0].mean())
        out[name+'_eos_loss']=float(loss[ids,1].mean())
        out[name+'_eos_accuracy']=float(ok[ids,1].float().mean())
    out['weight_norm']=sum(float(p.double().square().sum()) for p in m.parameters())**.5
    out['evaluation_forward_examples']=len(data['x'])
    return out

def checkpoint(root,m,opt,d,step,counters,all_rank_rng=None,all_rank_counts=None):
    free=disk_guard(root);start=time.monotonic();tmp=root/'latest.tmp.pt'
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=step,
                    dust_rng=d.rng.get_state() if d else None,counters=counters,
                    dust_rng_by_rank=all_rank_rng,dust_counts=dict(d.counts) if d else None,
                    dust_counts_by_rank=all_rank_counts),tmp)
    tmp.replace(root/'latest.pt')
    return dict(checkpoint_seconds=time.monotonic()-start,free_disk_bytes=free)

def source_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['src','scripts'] for p in sorted((ROOT/folder).glob('*')) if p.is_file()}

def run(args):
    global stopping
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    if args.distributed:
        dist.init_process_group('gloo',timeout=timedelta(minutes=3))
    rank=dist.get_rank() if args.distributed else 0
    world=dist.get_world_size() if args.distributed else 1
    if args.method=='dust' and args.population%world:
        raise ValueError('Global per-site population must be divisible by world size')
    root=args.output.resolve()
    if args.distributed:root=root/f'rank{rank}'
    root.mkdir(parents=True,exist_ok=False);disk_guard(root)
    start=time.monotonic();data=dataset();m=GrokTransformer();m.train()
    local_population=args.population//world if args.method=='dust' else args.population
    dust_settings=Settings(population=local_population,nrep=min(args.nrep,local_population),
                           direct_sigma=args.sigma,hub_last_sigma=2*args.sigma,
                           local_sigma=args.local_sigma)
    d=Dust(m,dust_settings,9278703+7777*rank,
           average_tensor if args.distributed else None) if args.method=='dust' else None
    if d is not None:m.requires_grad_(False)
    opt=optimizer(m)
    np.savez_compressed(root/'split.npz',train=data['train'],held=data['held'],
                        tokens=data['x'].numpy(),labels=data['y'].numpy(),pairs=data['pairs'])
    arg_config=vars(args).copy();arg_config['output']=str(root)
    config=dict(**arg_config,vocabulary=data['vocab'],model_seed=SEED,
                perturbation_seed=9278703+7777*rank,split_seed=0,train_pairs=len(data['train']),
                heldout_pairs=len(data['held']),split_sha256=hashlib.sha256(data['order'].tobytes()).hexdigest(),
                dust_source_commit='b20f7c03eac630b6441ba6f254128c1761dc45ad',
                reference_hashes=hashes(),implementation_hashes=source_hashes(),
                protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),
                parameters=sum(p.numel() for p in m.parameters()),initial_parameter_sha256=
                hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest(),
                optimizer='AdamW lr.001 betas(.9,.98) eps1e-8 wd1 all warmup10 no clip',
                dust_settings=asdict(dust_settings) if d else None,
                environment=dict(host=socket.gethostname(),torch=torch.__version__,python=platform.python_version(),
                                 affinity=sorted(os.sched_getaffinity(0)),threads=args.threads),
                start_unix=time.time(),internal_runtime_seconds=args.seconds,
                population_world_size=world,population_rank=rank,global_population=args.population,
                external_guard='GNU timeout --signal=TERM --kill-after='+('90s' if args.distributed else '60s')+'; seconds in launcher')
    write_json(root/'config.json',config)
    counters=dict(bp_forward_examples=0,eval_forward_examples=0,supervised_token_presentations=0)
    step=0;last_saved=-1;update_seconds=0.
    def record(step):
        nonlocal last_saved
        all_rank_rng=None
        all_rank_counts=None
        counts=dict(d.counts) if d else None
        if args.distributed:
            if d:
                counts=sum_counts(counts)
                all_rank_rng=[None]*world
                dist.all_gather_object(all_rank_rng,d.rng.get_state())
                all_rank_counts=[None]*world
                dist.all_gather_object(all_rank_counts,d.counts)
            if rank!=0:
                dist.barrier();last_saved=step;return
        t=time.monotonic();row=evaluate(m,data)
        counters['eval_forward_examples']+=row.pop('evaluation_forward_examples')
        row.update(step=step,method=args.method,population=args.population if d else 0,
                   wall_seconds=time.monotonic()-start,update_seconds=update_seconds,
                   counters=dict(counters),dust_counts=counts,
                   evaluation_seconds=time.monotonic()-t)
        row.update(checkpoint(root,m,opt,d,step,counters,all_rank_rng,all_rank_counts))
        with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write_json(root/'status.json',dict(state='TRAINING',**row));last_saved=step
        print(json.dumps(row),flush=True)
        if args.distributed:dist.barrier()
    def stop(sig,frame):
        global stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    record(0)
    while step<args.steps:
        stop_now=time.monotonic()-start>=args.seconds-45 or stopping
        if args.distributed:stop_now=any_stop(stop_now)
        if stop_now:break
        t=time.monotonic();loss=update(m,opt,d,data,step,args.chunk)
        update_seconds+=time.monotonic()-t;step+=1
        counters['supervised_token_presentations']+=2*len(data['train'])
        if d is None:counters['bp_forward_examples']+=len(data['train'])
        if step%args.eval_every==0 or step in (1,5,10):record(step)
        elif step%5==0 and rank==0:
            write_json(root/'progress.json',dict(step=step,loss_before_update=loss,
                       wall_seconds=time.monotonic()-start,seconds_per_update=update_seconds/step,
                       dust_counts=dict(d.counts) if d else None))
    if step!=last_saved:record(step)
    write_json(root/'completion.json',dict(state='HORIZON_COMPLETE' if step==args.steps else 'BUDGET_STOP',
               step=step,wall_seconds=time.monotonic()-start,budget_stop_is_failure=False))
    if args.distributed:dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['dust','bp'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--population',type=int,default=4)
    p.add_argument('--nrep',type=int,default=4);p.add_argument('--sigma',type=float,default=.2)
    p.add_argument('--local-sigma',type=float,default=.05)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--chunk',type=int,default=256)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--seconds',type=int,default=7140)
    p.add_argument('--eval-every',type=int,default=25)
    p.add_argument('--distributed',action='store_true');run(p.parse_args())
