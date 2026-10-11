"""Compare nonzero cached suffixes with hooks on the unchanged released model.

This checks every residual branch independently of Dust's reward calculation.
"""
import unittest

import torch

from src.model import GrokTransformer
from src.task import dataset


class IndependentSuffix(unittest.TestCase):
    def test_nonzero_suffix_matches_reference_hooks_and_preserves_cache(self):
        torch.set_num_threads(2)
        data = dataset()
        tokens = data['x'][data['train'][:3]]
        model = GrokTransformer().requires_grad_(False)
        rng = torch.Generator().manual_seed(91721)
        sites = ['embedding'] + [
            f'b{layer}.{kind}'
            for layer in range(2)
            for kind in ['hub', 'writer', 'hidden', 'ffout', 'norm1', 'norm2']
        ] + ['head']
        with torch.no_grad():
            model(tokens, capture=True)
            before = {
                name: {site: value.clone() for site, value in cache.items()}
                for name, cache in {
                    'cache': model.cache,
                    'inputs': model.inputs,
                    'outputs': model.outputs,
                    'norm_inputs': model.norm_inputs,
                }.items()
            }
            for site in sites:
                with self.subTest(site=site):
                    shape = model.outputs[site].shape
                    # Two distinct directions test draw/batch repeat ordering.
                    jitter = .07 * torch.randn(
                        2 * shape[0], *shape[1:], generator=rng)
                    if site == 'embedding':
                        module = model.ref.embedding
                    elif site == 'head':
                        module = model.ref.linear
                    else:
                        block = model.ref.decoder.blocks[int(site[1])]
                        module = {
                            'hub': block.self_attn.Wo,
                            'writer': block.self_attn.Wo,
                            'hidden': block.ffn.ffn[0],
                            'ffout': block.ffn.ffn[2],
                            'norm1': block.self_attn_norm,
                            'norm2': block.ffn_norm,
                        }[site.split('.')[1]]
                    if site.endswith('.hub'):
                        handle = module.register_forward_pre_hook(
                            lambda module, args: (args[0] + jitter,))
                    else:
                        handle = module.register_forward_hook(
                            lambda module, args, output: output + jitter)
                    try:
                        expected = model.ref(tokens.repeat(2, 1))[0]
                    finally:
                        handle.remove()
                    actual = model.suffix(site, jitter)
                    torch.testing.assert_close(
                        actual, expected, atol=3e-6, rtol=3e-5)
            for name, snapshot in before.items():
                cache = getattr(model, name)
                self.assertEqual(set(cache), set(snapshot))
                for site, value in snapshot.items():
                    torch.testing.assert_close(cache[site], value, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
