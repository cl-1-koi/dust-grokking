"""Autograd is confined to reference audits and diagnostic oracle targets."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from .task import dataset,reference_model
from .model import GrokTransformer,token_losses
from .dust import Dust,Settings

def oracle(x,y):
    m=reference_model(); acts={}; handles=[]
    def output(site):
        def hook(module,args,out):
            out.retain_grad(); acts[site]=out
        return hook
    def hub(site):
        def hook(module,args):
            args[0].retain_grad(); acts[site]=args[0]
        return hook
    handles.append(m.embedding.register_forward_hook(output('embedding')))
    handles.append(m.linear.register_forward_hook(output('head')))
    for l,block in enumerate(m.decoder.blocks):
        s=f'b{l}'
        for kind,module in [('writer',block.self_attn.Wo),('hidden',block.ffn.ffn[0]),
                             ('ffout',block.ffn.ffn[2]),('norm1',block.self_attn_norm),('norm2',block.ffn_norm)]:
            handles.append(module.register_forward_hook(output(s+'.'+kind)))
        handles.append(block.self_attn.Wo.register_forward_pre_hook(hub(s+'.hub')))
        for h,head in enumerate(block.self_attn.attn_heads):
            for kind in 'qkv':
                handles.append(getattr(head,'W'+kind).register_forward_hook(output(s+f'.{kind}{h}')))
    z=m(x)[0]; loss=token_losses(z,y).sum(); loss.backward()
    errors={site:a.grad.detach().clone() for site,a in acts.items()}
    for l in range(2):
        for kind in 'qkv': errors[f'b{l}.{kind}']=torch.cat([errors.pop(f'b{l}.{kind}{h}') for h in range(4)],-1)
    valid=(y!=-1).sum()
    gs={n:p.grad.detach().clone()/valid for n,p in m.named_parameters()}
    for h in handles:h.remove()
    return gs,errors,z.detach(),float(loss.detach()/valid)

def comparison(a,b):
    aa=torch.cat([v.reshape(-1).double() for v in a]); bb=torch.cat([v.reshape(-1).double() for v in b])
    na=float(aa.norm());nb=float(bb.norm())
    return dict(cosine=float(aa@bb/(aa.norm()*bb.norm()).clamp_min(1e-30)),
                norm_ratio=na/max(nb,1e-30),truth_norm=nb,
                relative_error=float((aa-bb).norm())/max(nb,1e-30))

def families(g,truth):
    groups={'all':list(g),'embedding':['embedding.weight'],'head':['linear.weight']}
    for kind in 'qkv': groups[kind]=[n for n in g if f'.W{kind}.weight' in n]
    groups['writers']=[n for n in g if 'self_attn.Wo.' in n or 'ffn.ffn.2.' in n]
    groups['hidden']=[n for n in g if 'ffn.ffn.0.' in n]
    groups['norm_gain']=[n for n in g if '_norm.weight' in n]
    groups['norm_bias']=[n for n in g if '_norm.bias' in n]
    return {k:comparison([g[n] for n in ns],[truth[n] for n in ns]) for k,ns in groups.items()}

def run(root,pops,seeds,sigmas):
    root.mkdir(parents=True,exist_ok=False);data=dataset()
    # Diagnostic uses TRAINING pairs only, selected before looking at labels/results.
    ix=data['train'][:8]; x=data['x'][ix]; y=data['y'][ix]
    truth,errors,z,loss=oracle(x,y)
    m=GrokTransformer(); m.requires_grad_(False)
    audit=dict(batch_indices=ix.tolist(),true_loss=loss,
               prefix_activation_gradient_norms={k:float(v[:,:4].norm()) for k,v in errors.items()},
               supervised_positions=[4,5])
    with (root/'audit.json').open('w') as f:json.dump(audit,f,indent=2)
    for sigma in sigmas:
        for p in pops:
            for seed in seeds:
                start=time.monotonic()
                d=Dust(m,Settings(population=p,nrep=min(8,p),direct_sigma=sigma,
                     hub_last_sigma=2*sigma),seed)
                g,_=d.gradients(x,y)
                full=families(g,truth)
                # Oracle hub targets isolate local estimation from first-stage noise.
                local={}
                for l in range(2):
                    for kind in 'qkv':
                        estimated=d.local(l,kind,errors[f'b{l}.hub'])
                        local[f'b{l}.{kind}']=comparison([estimated],[errors[f'b{l}.{kind}']])
                row=dict(population=p,seed=seed,sigma=sigma,seconds=time.monotonic()-start,
                         fully_estimated=full,oracle_local=local,**d.metadata())
                with (root/'alignment.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(dict(P=p,sigma=sigma,seed=seed,seconds=round(row['seconds'],2),
                   cosines={k:round(v['cosine'],3) for k,v in full.items()},
                   oracle_local_cosines={k:round(v['cosine'],3) for k,v in local.items()})),flush=True)
    # Compare same-token source credit with task-adapted future loss credit.
    for causal in (False,True):
        d=Dust(m,Settings(population=max(pops),nrep=8,direct_sigma=min(sigmas),causal_direct=causal))
        d.capture(x,y); e=d.direct('embedding')
        row=dict(causal_direct=causal,population=max(pops),
                 prefix_estimate_norm=float(e[:,:4].norm()),
                 prefix_true_norm=float(errors['embedding'][:,:4].norm()),
                 prefix_alignment=comparison([e[:,:4]],[errors['embedding'][:,:4]]))
        with (root/'sparse_credit.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('SPARSE_CREDIT',json.dumps(row),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True)
    a.add_argument('--populations',type=int,nargs='+',default=[4,16,64,256,1024])
    a.add_argument('--seeds',type=int,nargs='+',default=[9278703,9278704,9278705])
    a.add_argument('--sigmas',type=float,nargs='+',default=[.2,.02])
    args=a.parse_args();torch.set_num_threads(2)
    run(args.output,args.populations,args.seeds,args.sigmas)
