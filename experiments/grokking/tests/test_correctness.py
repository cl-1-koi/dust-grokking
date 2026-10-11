import copy
import unittest
import torch
from src.task import dataset,reference_model
from src.model import GrokTransformer,token_losses
from src.dust import Dust,Settings
from src.diagnostics import oracle

torch.set_num_threads(2)

class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=dataset(); ix=cls.data['train'][:8]
        cls.x=cls.data['x'][ix];cls.y=cls.data['y'][ix]

    def test_algebra_vocabulary_split(self):
        d=self.data; ids={v:i for i,v in enumerate(d['vocab'])}
        target=(d['pairs'][:,0]+d['pairs'][:,1])%97
        self.assertTrue(torch.equal(d['y'][:,4],torch.tensor([ids[str(c)] for c in target])))
        self.assertEqual(len(d['vocab']),239);self.assertEqual(len(d['train']),2823)
        self.assertEqual(len(set(d['train']) & set(d['held'])),0)
        self.assertEqual(int((d['y']!=-1).sum()),18818)

    def test_reference_forward_gradient_and_causality(self):
        m=GrokTransformer();ref=reference_model()
        z=m(self.x);zr=ref(self.x)[0]
        torch.testing.assert_close(z,zr,atol=2e-6,rtol=2e-5)
        token_losses(z,self.y).sum().backward();token_losses(zr,self.y).sum().backward()
        for n,p in m.parameters_named().items():
            torch.testing.assert_close(p.grad,dict(ref.named_parameters())[n].grad,atol=3e-5,rtol=3e-3)
        future=self.x.clone();future[:,-1]=(future[:,-1]+1)%239
        torch.testing.assert_close(m(future)[:,:-1],z[:,:-1],atol=1e-6,rtol=1e-6)

    def test_cache_suffix_and_affine_reconstruction(self):
        m=GrokTransformer();g,e,z,_=oracle(self.x,self.y)
        with torch.no_grad():
            out=m(self.x,capture=True);cache=copy.deepcopy(m.outputs)
            for site in ['embedding']+[f'b{l}.{s}' for l in range(2)
                         for s in ['hub','writer','hidden','ffout','norm1','norm2']]+['head']:
                torch.testing.assert_close(m.suffix(site,torch.zeros_like(m.outputs[site])),out,atol=2e-6,rtol=1e-5)
            for site in ['embedding','head']+[f'b{l}.{s}' for l in range(2)
                         for s in ['writer','hidden','ffout','norm1','norm2','q','k','v']]:
                for n,value in m.map_error(site,e[site]/16).items():
                    torch.testing.assert_close(value,g[n],atol=3e-6,rtol=2e-3)
            for site in cache: torch.testing.assert_close(m.outputs[site],cache[site],atol=0,rtol=0)

    def test_sparse_prefix_and_parameter_coverage(self):
        m=GrokTransformer().requires_grad_(False)
        _,errors,_,_=oracle(self.x,self.y)
        self.assertGreater(float(errors['embedding'][:,:4].norm()),.01)
        d=Dust(m,Settings(population=4,causal_direct=False));d.capture(self.x,self.y)
        self.assertEqual(float(d.direct('embedding')[:,:4].norm()),0.)
        d.s.causal_direct=True
        self.assertGreater(float(d.direct('embedding')[:,:4].norm()),0.)
        g,_=d.gradients(self.x,self.y)
        self.assertEqual(set(g),set(m.parameters_named()))
        self.assertTrue(all(torch.isfinite(v).all() for v in g.values()))
        self.assertTrue(all(not p.requires_grad for p in m.parameters()))

    def test_sign_sigma_squared_and_centering(self):
        # Antithetic scalar quadratic has exact slope, exposing sign/scale bugs.
        a=torch.tensor([-2.,-1.,1.,2.]);z=torch.tensor(1.7);target=.3
        for sigma in [.2,.02]:
            jitter=sigma*a
            plus=.5*(z+jitter-target)**2;minus=.5*(z-jitter-target)**2
            est=((plus-minus)*jitter/(2*sigma**2)).mean()/a.square().mean()
            torch.testing.assert_close(est,z-target,atol=3e-6,rtol=1e-5)
            self.assertLess(float(.5*((z-.01*est-target)**2)),float(.5*(z-target)**2))
        rng=torch.Generator().manual_seed(14)
        a=torch.randn(30000,4,generator=rng); delta=.2*1.4*a
        centered=delta-delta.mean(1,keepdim=True)
        raw=(centered*a).mean()/.2;corrected=raw*4/3
        self.assertAlmostEqual(float(raw),1.4*.75,delta=.02)
        self.assertAlmostEqual(float(corrected),1.4,delta=.02)

    def test_local_oracle_qkv_alignment_and_scale(self):
        m=GrokTransformer().requires_grad_(False);_,errors,_,_=oracle(self.x,self.y)
        d=Dust(m,Settings(population=2048,nrep=8));d.capture(self.x,self.y)
        for l in range(2):
            for kind in 'qkv':
                truth=errors[f'b{l}.{kind}']; est=d.local(l,kind,errors[f'b{l}.hub'])
                cosine=torch.nn.functional.cosine_similarity(est.flatten(),truth.flatten(),dim=0)
                self.assertGreater(float(cosine),.70,(l,kind,float(cosine)))
                self.assertLess(float(est.norm()/truth.norm()),1.6)

if __name__=='__main__':unittest.main()
