"""Numerical port of the released Power transformer, with source initialization."""
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j

REFERENCE=Path(__file__).parent/'reference_grok_20260929/transformer.py'


def torch_model(vocab,seed=9278700):
    import torch
    # Lightning is imported but unused by this standalone architecture file.
    source=REFERENCE.read_text().replace('import pytorch_lightning as pl','')
    ns={'__name__':'official_grok_reference'};exec(compile(source,str(REFERENCE),'exec'),ns)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model=ns['Transformer'](2,4,128,0.,50,vocab,'relu',weight_noise=0.).float()
    return model


def initialize(vocab,seed=9278700):
    m=torch_model(vocab,seed)
    return {n:p.detach().numpy().copy() for n,p in m.named_parameters()}


def position(length):
    pos=np.arange(length)[:,None];i=np.arange(128)[None,:]
    angles=pos/(10000**((i-i%2)/128))
    return j.asarray(np.where(i%2==0,np.sin(angles),np.cos(angles)),dtype=j.float32)


def norm(p,x,name):
    mu=x.mean(-1,keepdims=True);var=((x-mu)**2).mean(-1,keepdims=True)
    return (x-mu)*jax.lax.rsqrt(var+1e-5)*p[name+'.weight']+p[name+'.bias']


def forward(p,tokens,dense_embedding=False):
    n,length=tokens.shape
    embedding=p['embedding.weight']
    x=(jax.nn.one_hot(tokens,embedding.shape[0])@embedding if dense_embedding else embedding[tokens])+position(length)
    mask=j.arange(length)[:,None]>=j.arange(length)[None,:]
    for layer in range(2):
        b=f'decoder.blocks.{layer}'
        weights=j.concatenate([j.concatenate([p[f'{b}.self_attn.attn_heads.{h}.W{kind}.weight'] for h in range(4)],axis=0) for kind in ['q','k','v']],axis=0)
        q,k,v=j.transpose((x@weights.T).reshape(n,length,3,4,32),(2,0,3,1,4))
        att=jax.nn.softmax(j.where(mask,q@j.swapaxes(k,-1,-2)/(32**.5),-j.inf),axis=-1)
        h=j.transpose(att@v,(0,2,1,3)).reshape(n,length,128)@p[f'{b}.self_attn.Wo.weight'].T
        x=norm(p,x+h,b+'.self_attn_norm')
        h=jax.nn.relu(x@p[f'{b}.ffn.ffn.0.weight'].T)@p[f'{b}.ffn.ffn.2.weight'].T
        x=norm(p,x+h,b+'.ffn_norm')
    return x@p['linear.weight'].T


def loss(p,tokens,labels,mask):
    logits=forward(p,tokens)[:,-2:]
    ce=-j.take_along_axis(jax.nn.log_softmax(logits),labels[...,None],axis=-1)[...,0].mean(-1)
    return j.sum(mask*ce)/j.maximum(j.sum(mask),1.)


def adam(p,m,v,g,step):
    # The initial scheduler step is zero in the released Lightning recipe.
    lr=.001*j.minimum(step/10.,1.)
    mm={};vv={};pp={}
    for n,x in p.items():
        mm[n]=.9*m[n]+.1*g[n];vv[n]=.98*v[n]+.02*g[n]*g[n]
        pp[n]=x*(1-lr)-lr*(mm[n]/(1-.9**(step+1)))/(j.sqrt(vv[n]/(1-.98**(step+1)))+1e-8)
    return pp,mm,vv,step+1
