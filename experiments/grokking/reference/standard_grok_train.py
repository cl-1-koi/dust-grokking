"""Exact full-batch standard-transformer calibration, compiled on CUDA."""
import argparse
import json
import os
from pathlib import Path
import time
os.sched_setaffinity(0,{22,23,24,25})
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
import jax
import jax.numpy as j
jax.config.update('jax_default_matmul_precision','highest')
import standard_grok_model as M
import ttt_grok_jax as J
from ttt_basin_control import write,utc


def factory(tokens,labels,mask,chunk=10):
    tokens=j.asarray(tokens);labels=j.asarray(labels);mask=j.asarray(mask,dtype=j.float32)
    @jax.jit
    def run(state):
        def once(st,_):
            p,m,v,step=st
            losses,g=jax.vmap(jax.value_and_grad(M.loss),in_axes=(0,0,0,0))(p,tokens,labels,mask)
            return M.adam(p,m,v,g,step),losses
        st,losses=jax.lax.scan(once,state,None,length=chunk)
        return st,losses[-1]
    return run


@jax.jit
def predict(p,tokens,labels):
    logits=jax.vmap(M.forward,in_axes=(0,None))(p,tokens)[:,:,-2:]
    pred=j.argmax(logits,axis=-1)
    loss=-j.take_along_axis(jax.nn.log_softmax(logits),labels[None,...,None],axis=-1)[...,0]
    return pred,loss


def evaluate(root,state,cfg,elapsed):
    rows=[];saved={}
    for name in ['add97','S5']:
        data=np.load(root/f'{name}_data.npz');xx=data['tokens'];yy=data['labels'];n=len(xx)
        indices=[i for i,c in enumerate(cfg) if c['group']==name];p={k:x[j.asarray(indices)] for k,x in state[0].items()}
        pp=[];ll=[]
        for lo in range(0,n,512):
            ids=np.minimum(np.arange(lo,lo+512),n-1);pred,loss=predict(p,j.asarray(xx[ids]),j.asarray(yy[ids]));keep=min(512,n-lo)
            pp.append(np.asarray(pred)[:,:keep]);ll.append(np.asarray(loss)[:,:keep])
        pp=np.concatenate(pp,1);ll=np.concatenate(ll,1);assert np.isfinite(ll).all();saved[name]=pp
        for local,i in enumerate(indices):
            c=cfg[i];tr=data['permutation'][:c['train_pairs']];te=data['permutation'][c['train_pairs']:]
            ok=pp[local]==yy;norm=float(np.sqrt(sum(np.square(np.asarray(w[i]),dtype=np.float64).sum() for w in state[0].values())))
            rows.append(dict(**c,update=int(state[3]),train_accuracy=float(ok[tr,0].mean()),heldout_accuracy=float(ok[te,0].mean()),
                train_rhs_accuracy=float(ok[tr].all(-1).mean()),heldout_rhs_accuracy=float(ok[te].all(-1).mean()),
                train_loss=float(ll[local,tr].mean()),heldout_loss=float(ll[local,te].mean()),weight_norm=norm,
                optimizer_presentations=int(state[3])*c['train_pairs'],elapsed_training_seconds=elapsed))
    return rows,saved


def save(root,state,rows,pred):
    with (root/'curves.jsonl').open('a') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    arrays={prefix+'::'+n:np.asarray(w) for prefix,d in zip(['p','m','v'],state[:3]) for n,w in d.items()};arrays['update']=np.asarray(state[3])
    path=root/'latest.tmp.npz';np.savez(path,**arrays);path.replace(root/'latest.npz')
    np.savez_compressed(root/'latest_predictions.npz',update=arrays['update'],**pred)
    if int(state[3])%10000==0:np.savez_compressed(root/f'update{int(state[3]):06d}.npz',update=arrays['update'],**{k:v for k,v in arrays.items() if k.startswith('p::')})


def audit(root,vocab):
    import torch
    torch.set_num_threads(1);model=M.torch_model(vocab);p={n:j.asarray(w.detach().numpy().copy()) for n,w in model.named_parameters()}
    data=np.load(root/'add97_data.npz');x=data['tokens'][[0,1,97,100,300,500,600,900]];y=data['labels'][[0,1,97,100,300,500,600,900]]
    z=model(torch.tensor(x))[0];loss=torch.nn.functional.cross_entropy(z[:,-2:].reshape(-1,vocab),torch.tensor(y).long().flatten());loss.backward()
    z2=jax.jit(M.forward)(p,j.asarray(x));lj,g=jax.jit(jax.value_and_grad(M.loss))(p,j.asarray(x),j.asarray(y),j.ones(len(x)))
    np.testing.assert_allclose(np.asarray(z2),z.detach().numpy(),atol=5e-5,rtol=2e-4)
    np.testing.assert_allclose(float(lj),loss.item(),atol=2e-5,rtol=1e-5)
    ge=0.
    for n,w in model.named_parameters():
        a=np.asarray(g[n]);b=w.grad.numpy();np.testing.assert_allclose(a,b,atol=5e-5,rtol=3e-3)
        ge=max(ge,float(np.max(np.abs(a-b))));w.grad=torch.tensor(a.copy())
    opt=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.98),eps=1e-8,weight_decay=1.,foreach=False)
    # Identical gradients and moments at a non-warmup step isolate AdamW arithmetic.
    ep={n:w[None] for n,w in p.items()};eg={n:w[None] for n,w in g.items()};m=J.zeros(ep);v=J.zeros(ep)
    for n,w in model.named_parameters():
        opt.state[w]={'step':torch.tensor(10.),'exp_avg':torch.zeros_like(w),'exp_avg_sq':torch.zeros_like(w)}
    opt.step();updated=M.adam(ep,m,v,eg,j.array(10))[0];oe=0.
    for n,w in model.named_parameters():
        a=np.asarray(updated[n])[0];b=w.detach().numpy();np.testing.assert_allclose(a,b,atol=3e-6,rtol=1e-5);oe=max(oe,float(np.max(np.abs(a-b))))
    future=x.copy();future[:,-1]=(future[:,-1]+1)%vocab
    zf=jax.jit(M.forward)(p,j.asarray(future));np.testing.assert_allclose(np.asarray(z2)[:,:-1],np.asarray(zf)[:,:-1],atol=1e-6,rtol=1e-6)
    # Weighted padding is equivalent to the true batch; no held-out or padding gradient.
    xp=np.concatenate([x,x[:3]]);yp=np.concatenate([y,y[:3]]);mask=j.asarray([1.]*8+[0.]*3)
    _,gp=jax.jit(jax.value_and_grad(M.loss))(p,j.asarray(xp),j.asarray(yp),mask)
    for n in g:np.testing.assert_allclose(np.asarray(g[n]),np.asarray(gp[n]),atol=2e-5,rtol=2e-3)
    write(root/'port_audit.json',dict(status='PASS',logit_max_abs=float(np.max(np.abs(np.asarray(z2)-z.detach().numpy()))),
        gradient_max_abs=ge,identical_gradient_adamw_max_abs=oe,causal_future_invariance='PASS',zero_weight_padding='PASS',
        reference='Official source, only unused Lightning import removed; float32 cast explicit'))
    print('STANDARD_PORT_PASS',ge,oe,flush=True)
    # Discarded overfit audit at the complete model size.
    pp=J.stack([M.initialize(vocab)]);state=(pp,J.zeros(pp),J.zeros(pp),j.array(0));step=factory(x[None],y[None],np.ones((1,len(x))),chunk=50)
    for _ in range(20):
        state,_=step(state);z=np.asarray(M.forward({n:w[0] for n,w in state[0].items()},j.asarray(x)))[:,-2:]
        acc=float((z.argmax(-1)==y).all(-1).mean())
        if acc==1.:break
    write(root/'overfit_audit.json',dict(status='PASS' if acc==1 else 'FAIL',accuracy=acc,updates=int(state[3]),weights_discarded=True))
    assert acc==1.,'TINY_OVERFIT_FAILED'


def run(root,deadline):
    assert jax.default_backend()=='gpu';design=json.loads((root/'design.json').read_text());cfg=design['configs'];vocab=len(design['vocabulary'])
    audit(root,vocab);jax.clear_caches()
    data=np.load(root/'training_data.npz');p=J.stack([M.initialize(vocab,c['seed']) for c in cfg]);state=(p,J.zeros(p),J.zeros(p),j.array(0))
    step=factory(data['tokens'],data['labels'],data['mask'],chunk=10)
    t=time.time();pilot,_=step(state);jax.block_until_ready(pilot);compile_time=time.time()-t
    t=time.time();pilot,_=step(pilot);jax.block_until_ready(pilot);dt=(time.time()-t)/10
    rows,pred=evaluate(root,state,cfg,0.)
    t=time.time();rows,pred=evaluate(root,state,cfg,0.);save(root,state,rows,pred);ev=time.time()-t
    available=deadline-time.time()-90;cost=dt+ev/500
    horizon=min(100000,int(available/(1.1*cost)/500)*500)
    assert horizon>=500,'NO_TRAINING_BUDGET'
    design.update(executed_horizon=horizon,parameters_per_fit=sum(w[0].size for w in p.values()),
        cut_reason=None if horizon==100000 else 'Measured full-batch cost exceeds two-hour deadline; common horizon cut, recipe unchanged')
    write(root/'design.json',design);write(root/'pilot.json',dict(compile_plus10_seconds=compile_time,seconds_per_full_batch_update=dt,
        evaluation_save_seconds=ev,projected_100k_seconds=cost*100000,executed_horizon=horizon,available_seconds=available,pilot_updates_discarded=20))
    print('STANDARD_START',horizon,'full_batch_seconds',dt,'projected100k',cost*100000,flush=True)
    start=time.time();durable=0
    for _ in range(horizon//10):
        if time.time()>deadline-90:break
        state,loss=step(state);jax.block_until_ready(state);assert np.isfinite(np.asarray(loss)).all();u=int(state[3])
        if u%500==0:
            rows,pred=evaluate(root,state,cfg,time.time()-start);save(root,state,rows,pred);durable=u
            write(root/'training_status.json',dict(status='TRAINING',update=u,horizon=horizon,utc=utc()))
            print('STANDARD_CHECK',u,[(r['key'],round(r['train_accuracy'],4),round(r['heldout_accuracy'],4)) for r in rows],flush=True)
    if int(state[3])!=durable:
        rows,pred=evaluate(root,state,cfg,time.time()-start);save(root,state,rows,pred)
    write(root/'training_status.json',dict(status='HORIZON_COMPLETE' if int(state[3])==horizon else 'UNFINISHED_BUDGET',
        update=int(state[3]),executed_horizon=horizon,requested_horizon=100000,elapsed_training_seconds=time.time()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();run(a.root.resolve(),a.deadline)
