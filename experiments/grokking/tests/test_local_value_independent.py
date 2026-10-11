"""Deterministic forward-only check against the analytic linear V map.

An orthogonal noise population has covariance I exactly, eliminating sampling
variance so a sign, sigma, causal credit or head-layout bug cannot hide.
"""
import unittest

import torch

from src.dust import Dust, Settings
from src.model import GrokTransformer
from src.task import dataset


class IndependentValue(unittest.TestCase):
    def test_value_map_exact_with_covariance_identity(self):
        torch.set_num_threads(2)
        data = dataset()
        tokens = data['x'][data['train'][:1], :2]
        labels = data['y'][data['train'][:1], :2]
        model = GrokTransformer().requires_grad_(False)
        estimator = Dust(model, Settings(population=64, nrep=8))
        estimator.capture(tokens, labels)
        rng = torch.Generator().manual_seed(48271)
        target = torch.randn(1, 2, 128, generator=rng)
        hadamard = torch.ones(1, 1)
        while hadamard.shape[0] < 64:
            hadamard = torch.cat([
                torch.cat([hadamard, hadamard], dim=1),
                torch.cat([hadamard, -hadamard], dim=1),
            ], dim=0)
        # The score is head-local; each head separately has covariance I.
        population = hadamard.reshape(64, 1, 2, 1, 32).expand(
            -1, -1, -1, 4, -1).reshape(64, 1, 2, 128)
        q = model.outputs['b0.q'].reshape(1, 2, 4, 32).transpose(1, 2)
        k = model.outputs['b0.k'].reshape(1, 2, 4, 32).transpose(1, 2)
        mask = torch.ones(2, 2, dtype=torch.bool).tril()
        weights = ((q @ k.transpose(-1, -2)) / 32**.5).masked_fill(
            ~mask, -torch.inf).softmax(-1)
        for gamma in [0., .98, 1.]:
            credit = torch.tensor([[1., gamma], [0., 1.]])
            expected = torch.einsum(
                'bhst,bshd->bthd',
                weights * credit.T,
                target.reshape(1, 2, 4, 32),
            ).reshape(1, 2, 128)
            for sigma in [.01, .7]:
                with self.subTest(gamma=gamma, sigma=sigma):
                    estimator.s.gamma = gamma
                    estimator.s.local_sigma = sigma
                    cursor = 0

                    def orthogonal_noise(*shape):
                        nonlocal cursor
                        self.assertEqual(shape[1:], (1, 2, 128))
                        noise = population[cursor:cursor + shape[0]]
                        cursor += shape[0]
                        return noise

                    estimator.noise = orthogonal_noise
                    actual = estimator.local(0, 'v', target)
                    self.assertEqual(cursor, 64)
                    torch.testing.assert_close(
                        actual, expected, atol=2e-6, rtol=3e-5)


if __name__ == '__main__':
    unittest.main()
