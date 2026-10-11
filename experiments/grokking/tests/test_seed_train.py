import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.seed_train import initialize, masks, save_predictions, score
from src.task import dataset
from src.train import evaluate, update


class SeedReplicaChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data = dataset()

    def small_data(self):
        data = dict(self.data)
        indices = np.array([a*97+b for a in range(3) for b in range(3)])
        data.update(x=data['x'][indices], y=data['y'][indices], pairs=data['pairs'][indices],
                    train=np.array([0, 1, 4, 6]), held=np.array([2, 3, 5, 7, 8]))
        return data

    def test_initialization_and_noise_are_separate_reproducible_factors(self):
        a, oa, da = initialize('dust', 9278700, 9278901)
        b, ob, db = initialize('dust', 9278700, 9278902)
        c, oc, dc = initialize('dust', 9278701, 9278901)
        d, od, _ = initialize('bp', 9278700, 9278901)
        for pa, pb, pd in zip(a.parameters(), b.parameters(), d.parameters()):
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)
            torch.testing.assert_close(pa, pd, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters())))
        first = da.noise(64)
        torch.testing.assert_close(first, dc.noise(64), rtol=0, atol=0)
        self.assertFalse(torch.equal(first, db.noise(64)))
        for opt in (oa, ob, oc, od):
            self.assertEqual(len(opt.state), 0)
        self.assertTrue(all(not p.requires_grad for p in a.parameters()))
        self.assertTrue(all(p.requires_grad for p in d.parameters()))

    def test_unseen_mask_uses_both_orientations_with_diagonals(self):
        train, unseen, reversed_train = masks(self.small_data())
        self.assertEqual(np.flatnonzero(train).tolist(), [0, 1, 4, 6])
        self.assertEqual(np.flatnonzero(unseen).tolist(), [5, 7, 8])
        self.assertEqual(np.flatnonzero(reversed_train).tolist(), [2, 3])
        self.assertFalse(np.any(unseen & reversed_train))
        _, full_unseen, full_reversed = masks(self.data)
        self.assertEqual(int(full_unseen.sum()), 4669)
        self.assertEqual(int(full_reversed.sum()), 1917)

    def test_dust_update_cannot_invoke_backward(self):
        model, opt, dust = initialize('dust', 9278701, 9278902)
        before = [p.detach().clone() for p in model.parameters()]
        with patch('torch.autograd.backward', side_effect=AssertionError('Backprop forbidden')):
            loss = update(model, opt, dust, self.small_data(), 10, 3)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(model.parameters(), before)))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_export_matches_existing_evaluator_and_direct_logits(self):
        model, _, _ = initialize('dust', 9278702, 9278901)
        data = self.small_data()
        row, payload = score(model, data, chunk=4)
        reference = evaluate(model, data, chunk=4)
        for key, value in reference.items():
            self.assertAlmostEqual(row[key], value, places=7)
        with torch.no_grad():
            logits = model(data['x'])[:, 4]
        truth = data['y'][:, 4]
        for i in range(len(truth)):
            self.assertEqual(payload['predicted_token'][i], int(logits[i].argmax()))
            other = torch.cat([logits[i, :truth[i]], logits[i, truth[i]+1:]])
            self.assertAlmostEqual(float(payload['margin'][i]), float(logits[i, truth[i]]-other.max()), places=5)
        self.assertEqual(row['unseen_pair_correct'], int(payload['correct'][[5, 7, 8]].sum()))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_predictions(root, 3177, payload)
            saved = np.load(root/'predictions/step-03177.npz')
            self.assertEqual(int(saved['step']), 3177)
            np.testing.assert_array_equal(saved['unseen_pair'], payload['unseen_pair'])
            np.testing.assert_array_equal(saved['correct'], payload['correct'])
            self.assertEqual(list(root.rglob('*.tmp*')), [])

    def test_prediction_disk_guard_fails_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('src.seed_train.disk_guard', side_effect=OSError('below5GiB')):
                with self.assertRaises(OSError):
                    save_predictions(root, 0, {'correct': np.array([True])})
            self.assertFalse((root/'predictions').exists())


if __name__ == '__main__':
    unittest.main()
