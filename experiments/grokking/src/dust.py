"""Source-derived Dust adapted to the released transformer and sparse RHS CE.

No autograd in this module. See PROTOCOL.md for deviations from pinned Dust.
Counts include actual full/suffix, head-only, and local attention evaluations.
"""
from dataclasses import dataclass, asdict
import torch
from .model import token_losses

@dataclass
class Settings:
    population:int=4
    nrep:int=4
    direct_sigma:float=.2
    local_sigma:float=.05
    head_sigma:float=.05
    hub_last_sigma:float=.4
    gamma:float=1.
    causal_direct:bool=True
    terminal_tokenwise:bool=True
    center_correction:bool=True
    head_multiplier:int=4

class Dust:
    def __init__(self,model,settings=Settings(),seed=9278703,target_average=None):
        self.model=model; self.s=settings
        self.target_average=target_average
        self.rng=torch.Generator().manual_seed(seed)
        self.counts=dict(clean_forwards=0,suffix_forwards=0,forward_examples=0,
                         direct_draws=0,hub_draws=0,head_draws=0,local_draws=0,
                         head_examples=0,attention_examples=0)

    def noise(self,*shape): return torch.randn(*shape,generator=self.rng)

    def chunks(self,draws):
        for start in range(0,draws,self.s.nrep): yield min(self.s.nrep,draws-start)

    def credit(self,t):
        lag=torch.arange(t)[None,:]-torch.arange(t)[:,None]
        return (lag>=0).float()*self.s.gamma**lag.clamp_min(0)

    @torch.no_grad()
    def capture(self,x,y):
        self.x=x; self.y=y; self.b,self.t=x.shape
        self.clean=token_losses(self.model(x,capture=True),y)
        self.counts['clean_forwards']+=1; self.counts['forward_examples']+=self.b

    @torch.no_grad()
    def direct(self,site,draws=None,sigma=None):
        draws=draws or self.s.population; sigma=sigma or self.s.direct_sigma
        width=self.model.outputs[site].shape[-1]
        error=torch.zeros(self.b,self.t,width)
        credit=self.credit(self.t)
        for n in self.chunks(draws):
            a=self.noise(n,self.b,self.t,width)
            z=self.model.suffix(site,sigma*a.reshape(n*self.b,self.t,width))
            loss=token_losses(z,self.y.repeat(n,1)).reshape(n,self.b,self.t)
            delta=loss-self.clean
            if n>1:
                delta-=delta.mean(0,keepdim=True)
                if self.s.center_correction: delta*=n/(n-1)
            # All final-block direct sites have only tokenwise downstream maps.
            # Earlier block sites and embeddings can influence future RHS losses.
            if self.s.causal_direct and not (self.s.terminal_tokenwise and site.startswith('b1.')):
                delta=torch.einsum('nbs,ts->nbt',delta,credit)
            error+=torch.einsum('nbt,nbtd->btd',delta,a)/sigma
            self.counts['suffix_forwards']+=1
            self.counts['forward_examples']+=n*self.b
            self.counts['hub_draws' if site.endswith('.hub') else 'direct_draws']+=n
        return error/draws

    @torch.no_grad()
    def head(self,draws=None):
        draws=draws or self.s.population*self.s.head_multiplier
        raw=self.model.outputs['head']; error=torch.zeros_like(raw)
        for n in self.chunks(draws):
            a=self.noise(n,*raw.shape)
            z=(raw+ self.s.head_sigma*a).reshape(n*self.b,self.t,-1)
            delta=token_losses(z,self.y.repeat(n,1)).reshape(n,self.b,self.t)-self.clean
            # Faithful to source: batch centering (not population centering).
            if self.b>1:
                delta-=delta.mean(1,keepdim=True)
                if self.s.center_correction: delta*=self.b/(self.b-1)
            error+=torch.einsum('nbt,nbtd->btd',delta,a)/self.s.head_sigma
            self.counts['head_draws']+=n; self.counts['head_examples']+=n*self.b
        return error/draws

    @torch.no_grad()
    def local(self,l,kind,target,draws=None):
        draws=draws or self.s.population; s=f'b{l}'; sigma=self.s.local_sigma
        q,k,v=[self.model.outputs[s+'.'+z] for z in 'qkv']
        baseline=self.model.outputs[s+'.hub']; credit=self.credit(self.t)
        error=torch.zeros_like(q)
        for n in self.chunks(draws):
            a=self.noise(n,self.b,self.t,128)
            qa,ka,va=[z.repeat(n,1,1) for z in (q,k,v)]
            if kind=='q': qa+=sigma*a.reshape_as(qa)
            elif kind=='k': ka+=sigma*a.reshape_as(ka)
            elif kind=='v': va=sigma*a.reshape_as(va)
            out=self.model.attention(qa,ka,va).reshape(n,self.b,self.t,4,32)
            if kind!='v': out-=baseline.reshape(self.b,self.t,4,32)
            score=torch.einsum('bthd,nbthd->nbth',target.reshape(self.b,self.t,4,32),out)
            if kind!='q': score=torch.einsum('nbsh,ts->nbth',score,credit)
            error+=torch.einsum('nbth,nbthd->bthd',score,a.reshape(n,self.b,self.t,4,32)).reshape_as(q)/sigma
            self.counts['local_draws']+=n; self.counts['attention_examples']+=n*self.b
        return error/draws

    @torch.no_grad()
    def gradients(self,x,y):
        self.capture(x,y); gradients={}
        for site in ['embedding']+[f'b{l}.{z}' for l in range(2)
                     for z in ['writer','hidden','ffout','norm1','norm2']]:
            gradients.update(self.model.map_error(site,self.direct(site)))
        gradients.update(self.model.map_error('head',self.head()))
        for l in range(2):
            target=self.direct(f'b{l}.hub',sigma=self.s.direct_sigma if l==0 else self.s.hub_last_sigma)
            if self.target_average is not None:target=self.target_average(target)
            for kind in 'qkv':
                gradients.update(self.model.map_error(f'b{l}.{kind}',self.local(l,kind,target)))
        assert set(gradients)==set(self.model.parameters_named()), 'Missing parameter estimate'
        # Exactly one normalization by number of supervised RHS targets.
        valid=int((y!=-1).sum())
        return {name:g/valid for name,g in gradients.items()},float(self.clean.sum()/valid)

    def metadata(self): return dict(settings=asdict(self.s),counts=dict(self.counts))
