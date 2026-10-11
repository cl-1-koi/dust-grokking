"""Fused-head forward of the released model, with explicit clean activation caches.

Parameters remain in their original names and head layout. No architecture or
initialization changes. Suffix execution uses only immutable clean caches.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .task import reference_model

class GrokTransformer(nn.Module):
    def __init__(self, seed=9278700):
        super().__init__(); self.ref=reference_model(seed)
        self.cache={}; self.inputs={}; self.outputs={}; self.norm_inputs={}

    def parameters_named(self):
        return dict(self.ref.named_parameters())

    def _save(self, site, x, y, capture):
        if capture:
            self.inputs[site]=x.detach().clone(); self.outputs[site]=y.detach().clone()
        return y

    def _qkv(self, block, kind, x):
        weights=torch.cat([getattr(h,'W'+kind).weight for h in block.self_attn.attn_heads])
        return F.linear(x,weights)

    @staticmethod
    def attention(q,k,v):
        b,t,d=q.shape; h=4
        q,k,v=[a.reshape(b,t,h,d//h).transpose(1,2) for a in (q,k,v)]
        mask=torch.ones(t,t,device=q.device,dtype=torch.bool).tril()
        a=(q@k.transpose(-1,-2))/(d//h)**.5
        return (a.masked_fill(~mask,-torch.inf).softmax(-1)@v).transpose(1,2).reshape(b,t,d)

    def _norm(self, x, module, site, capture):
        u=F.layer_norm(x,module.normalized_shape,None,None,module.eps)
        if capture: self.norm_inputs[site]=u.detach().clone()
        return self._save(site,u,u*module.weight+module.bias,capture)

    def _blocks(self, x, first=0, capture=False):
        for l in range(first,2):
            block=self.ref.decoder.blocks[l]; s=f'b{l}'
            if capture: self.cache[s+'.x']=x.detach().clone()
            q,k,v=[self._save(s+'.'+kind,x,self._qkv(block,kind,x),capture) for kind in 'qkv']
            hub=self.attention(q,k,v); self._save(s+'.hub',x,hub,capture)
            writer=self._save(s+'.writer',hub,block.self_attn.Wo(hub),capture)
            x=self._norm(x+writer,block.self_attn_norm,s+'.norm1',capture)
            hidden=self._save(s+'.hidden',x,block.ffn.ffn[0](x),capture)
            ff=self._save(s+'.ffout',hidden.relu(),block.ffn.ffn[2](hidden.relu()),capture)
            x=self._norm(x+ff,block.ffn_norm,s+'.norm2',capture)
        return self._save('head',x,self.ref.linear(x),capture)

    def forward(self, tokens, capture=False):
        if capture:
            self.cache={}; self.inputs={}; self.outputs={}; self.norm_inputs={}
        x=self._save('embedding',tokens,self.ref.embedding(tokens),capture)
        x=x+self.ref.position_encoding[:tokens.shape[1]]
        return self._blocks(x,capture=capture)

    def suffix(self, site, noise):
        """One independently perturbed site, flattened [draw,batch,token,width]."""
        b=self.outputs[site].shape[0]; n=noise.shape[0]//b
        repeat=lambda v:v.repeat(n,1,1)
        x=repeat(self.outputs[site])+noise
        if site=='head': return x
        if site=='embedding':
            return self._blocks(x+self.ref.position_encoding[:x.shape[1]])
        l=int(site[1]); kind=site.split('.')[1]; block=self.ref.decoder.blocks[l]; s=f'b{l}'
        residual=repeat(self.outputs[s+'.norm1'])
        if kind=='hub': x=block.self_attn.Wo(x); kind='writer'
        if kind=='writer':
            x=self._norm(repeat(self.cache[s+'.x'])+x,block.self_attn_norm,s+'.norm1',False)
            kind='norm1'
        if kind=='norm1':
            residual=x
            x=block.ffn.ffn[0](x); kind='hidden'
        if kind=='hidden':
            x=block.ffn.ffn[2](x.relu()); kind='ffout'
        if kind=='ffout':
            x=self._norm(residual+x,block.ffn_norm,s+'.norm2',False)
        return self._blocks(x,first=l+1)

    def map_error(self, site, error):
        """Exact affine reconstruction with no backward invocation."""
        if site=='embedding':
            g=torch.zeros_like(self.ref.embedding.weight)
            g.index_add_(0,self.inputs[site].flatten(),error.reshape(-1,128))
            return {'embedding.weight':g}
        if site=='head': prefix='linear'
        else:
            l=int(site[1]); kind=site.split('.')[1]; prefix=f'decoder.blocks.{l}.'
            if kind in ('norm1','norm2'):
                prefix+= 'self_attn_norm' if kind=='norm1' else 'ffn_norm'
                return {prefix+'.weight':(error*self.norm_inputs[site]).sum((0,1)),
                        prefix+'.bias':error.sum((0,1))}
            if kind in 'qkv':
                g=torch.einsum('bto,bti->oi',error,self.inputs[site])
                return {prefix+f'self_attn.attn_heads.{h}.W{kind}.weight':g[h*32:(h+1)*32]
                        for h in range(4)}
            prefix+={'writer':'self_attn.Wo','hidden':'ffn.ffn.0','ffout':'ffn.ffn.2'}[kind]
        return {prefix+'.weight':torch.einsum('bto,bti->oi',error,self.inputs[site])}

def token_losses(logits, labels):
    return F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-1,
                           reduction='none').reshape_as(labels)
