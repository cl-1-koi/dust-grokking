"""CPU execution of the unchanged Dust estimator for the six-token RHS task."""
from types import MethodType

import torch

from .cpu_suffix_candidate import terminal_suffix
from .dust import Dust
from .model import token_losses


class CpuDust(Dust):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        model.suffix = MethodType(terminal_suffix, model)

    @torch.no_grad()
    def capture(self, x, y):
        if x.shape[1] != 6 or not (y[:, :-2] == -1).all() or not (y[:, -2:] >= 0).all():
            raise ValueError('CPU suffix requires six tokens and exactly two terminal targets')
        return super().capture(x, y)

    @torch.no_grad()
    def direct(self, site, draws=None, sigma=None):
        draws = draws or self.s.population; sigma = sigma or self.s.direct_sigma
        width = self.model.outputs[site].shape[-1]
        error = torch.zeros(self.b, self.t, width)
        credit = self.credit(self.t)
        for n in self.chunks(draws):
            a = self.noise(n, self.b, self.t, width)
            z = self.model.suffix(site, sigma*a.reshape(n*self.b, self.t, width))
            loss = token_losses(z, self.y.repeat(n, 1)).reshape(n, self.b, self.t)
            delta = loss-self.clean
            if n > 1:
                delta -= delta.mean(0, keepdim=True)
                if self.s.center_correction: delta *= n/(n-1)
            if self.s.causal_direct and not (self.s.terminal_tokenwise and site.startswith('b1.')):
                delta = torch.einsum('nbs,ts->nbt', delta, credit)
            error += (delta.unsqueeze(-1)*a).sum(0)/sigma
            self.counts['suffix_forwards'] += 1
            self.counts['forward_examples'] += n*self.b
            self.counts['hub_draws' if site.endswith('.hub') else 'direct_draws'] += n
        return error/draws

    @torch.no_grad()
    def head(self, draws=None):
        draws = draws or self.s.population*self.s.head_multiplier
        raw = self.model.outputs['head']; error = torch.zeros_like(raw)
        for n in self.chunks(draws):
            a = self.noise(n, *raw.shape)
            z = (raw+self.s.head_sigma*a).reshape(n*self.b, self.t, -1)
            delta = token_losses(z, self.y.repeat(n, 1)).reshape(n, self.b, self.t)-self.clean
            if self.b > 1:
                delta -= delta.mean(1, keepdim=True)
                if self.s.center_correction: delta *= self.b/(self.b-1)
            error += (delta.unsqueeze(-1)*a).sum(0)/self.s.head_sigma
            self.counts['head_draws'] += n; self.counts['head_examples'] += n*self.b
        return error/draws

    @torch.no_grad()
    def local(self, l, kind, target, draws=None):
        draws = draws or self.s.population; s = f'b{l}'; sigma = self.s.local_sigma
        q, k, v = [self.model.outputs[s+'.'+z] for z in 'qkv']
        baseline = self.model.outputs[s+'.hub']; credit = self.credit(self.t)
        error = torch.zeros_like(q)
        for n in self.chunks(draws):
            a = self.noise(n, self.b, self.t, 128)
            qa, ka, va = [z.repeat(n, 1, 1) for z in (q, k, v)]
            if kind == 'q': qa += sigma*a.reshape_as(qa)
            elif kind == 'k': ka += sigma*a.reshape_as(ka)
            elif kind == 'v': va = sigma*a.reshape_as(va)
            out = self.model.attention(qa, ka, va).reshape(n, self.b, self.t, 4, 32)
            if kind != 'v': out -= baseline.reshape(self.b, self.t, 4, 32)
            score = torch.einsum('bthd,nbthd->nbth', target.reshape(self.b, self.t, 4, 32), out)
            if kind != 'q': score = torch.einsum('nbsh,ts->nbth', score, credit)
            error += (score.unsqueeze(-1)*a.reshape(n, self.b, self.t, 4, 32)).sum(0).reshape_as(q)/sigma
            self.counts['local_draws'] += n; self.counts['attention_examples'] += n*self.b
        return error/draws
