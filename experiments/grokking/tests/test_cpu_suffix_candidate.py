import unittest

import torch

from src.cpu_audit import compare, fresh, supervised_losses, loss_kernel
from src.cpu_suffix_candidate import terminal_suffix
from src.model import token_losses
from src.task import dataset


class SuffixCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = dataset()

    def setUp(self):
        ids = self.data['train'][:7]
        self.x, self.y = self.data['x'][ids], self.data['y'][ids]
        self.model, self.dust = fresh('baseline')
        self.dust.capture(self.x, self.y)

    def test_all_direct_sites_preserve_supervised_suffix(self):
        sites = ['embedding']+[f'b{l}.{s}' for l in range(2)
            for s in ['writer', 'hidden', 'ffout', 'norm1', 'norm2', 'hub']]
        caches = {k:v.clone() for k,v in self.model.outputs.items()}
        rng = torch.Generator().manual_seed(7291)
        for site in sites:
            with self.subTest(site=site):
                width = self.model.outputs[site].shape[-1]
                noise = .002*torch.randn(3*len(self.x), 6, width, generator=rng)
                reference = self.model.suffix(site, noise)
                candidate = terminal_suffix(self.model, site, noise)
                torch.testing.assert_close(reference[:, -2:], candidate[:, -2:], rtol=2e-5, atol=4e-6)
                torch.testing.assert_close(token_losses(reference, self.y.repeat(3, 1)),
                                           token_losses(candidate, self.y.repeat(3, 1)), rtol=2e-5, atol=4e-6)
        for key in caches:
            torch.testing.assert_close(caches[key], self.model.outputs[key], rtol=0, atol=0)

    def test_operand_influence_and_future_invariance(self):
        zero = torch.zeros(len(self.x), 6, 128)
        clean = terminal_suffix(self.model, 'embedding', zero)
        operand = zero.clone(); operand[:, 1, 7] = .1
        changed = terminal_suffix(self.model, 'embedding', operand)
        self.assertGreater(float((clean[:, 4]-changed[:, 4]).abs().max()), 1e-6)
        future = zero.clone(); future[:, 5] = 100
        changed = terminal_suffix(self.model, 'embedding', future)
        torch.testing.assert_close(clean[:, 4], changed[:, 4], rtol=0, atol=0)

    def test_same_noise_full_estimator(self):
        baseline, d0 = fresh('baseline')
        candidate, d1 = fresh('terminal')
        d0.s.population = d1.s.population = 4
        g0, loss0 = d0.gradients(self.x, self.y)
        g1, loss1 = d1.gradients(self.x, self.y)
        self.assertEqual(loss0, loss1)
        self.assertTrue(torch.equal(d0.rng.get_state(), d1.rng.get_state()))
        self.assertEqual(d0.counts, d1.counts)
        self.assertLess(compare(g0, g1)['relative_l2'], .001)
        self.assertTrue(all(p.grad is None for p in candidate.parameters()))

    def test_supervised_only_loss_exact(self):
        logits = self.model(self.x)
        torch.testing.assert_close(token_losses(logits, self.y), supervised_losses(logits, self.y), rtol=0, atol=0)

    def test_weighted_reductions_and_restore(self):
        original = torch.einsum
        for shape, equation in [((4, 7, 6, 128), 'nbt,nbtd->btd'),
                                ((4, 7, 6, 4, 32), 'nbth,nbthd->bthd')]:
            noise = torch.randn(*shape)
            reward = torch.randn(*shape[:-1])
            reference = torch.einsum(equation, reward, noise)
            with loss_kernel(False, reduce=True):
                candidate = torch.einsum(equation, reward, noise)
            torch.testing.assert_close(reference, candidate, rtol=2e-5, atol=2e-6)
            self.assertIs(torch.einsum, original)

    def test_combined_estimator(self):
        baseline, d0 = fresh('baseline')
        candidate, d1 = fresh('terminal')
        d0.s.population = d1.s.population = 4
        g0, _ = d0.gradients(self.x, self.y)
        with loss_kernel(False, reduce=True):
            g1, _ = d1.gradients(self.x, self.y)
        self.assertTrue(torch.equal(d0.rng.get_state(), d1.rng.get_state()))
        self.assertEqual(d0.counts, d1.counts)
        self.assertLess(compare(g0, g1)['relative_l2'], .001)


if __name__ == '__main__':
    unittest.main()
