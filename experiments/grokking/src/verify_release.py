"""Verify exported weights, causal predictions, metrics and fixed ensembles."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import GrokTransformer
from .seed_train import masks
from .task import dataset


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data = dataset()
    _, unseen, _ = masks(data)
    truth = data['y'][:, 4].numpy()
    manifest = json.loads(Path('checkpoints/manifest.json').read_text())
    selected = []
    result = {'checkpoints': [], 'primary_endpoint': 3177}
    for entry in manifest:
        path = Path(entry['file'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['training_backprop_examples'] == 0
        model = GrokTransformer().eval().requires_grad_(False)
        model.load_state_dict(saved['model'], strict=True)
        logits = []
        for begin in range(0, len(data['x']), 256):
            batch = data['x'][begin:begin+256]
            z = model(batch[:, :5])[:, 4]
            # All inputs physically omit the answer; compare full-input path too.
            torch.testing.assert_close(z, model(batch)[:, 4], rtol=0, atol=0)
            logits.append(z.numpy())
        z = np.concatenate(logits)
        correct = z.argmax(-1) == truth
        rows = [json.loads(line) for line in
                (Path('results')/entry['run']/'metrics.jsonl').read_text().splitlines()]
        expected = next(row for row in rows if row['step'] == entry['step'])
        assert int(correct[unseen].sum()) == expected['unseen_pair_correct']
        assert abs(float(correct[data['held']].mean())-expected['heldout_accuracy']) < 1e-7
        result['checkpoints'].append(dict(run=entry['run'], step=entry['step'],
            held_correct=int(correct[data['held']].sum()), held_count=len(data['held']),
            unseen_correct=int(correct[unseen].sum()), unseen_count=int(unseen.sum()),
            answer_token_removed=True))
        if entry['step'] == 3177:
            selected.append(z.astype(np.float64))
    assert len(selected) == 6
    z = np.stack(selected)
    probabilities = np.exp(z-z.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    for name, values in [('logit_mean', z.mean(0)), ('probability_mean', probabilities.mean(0))]:
        good = values.argmax(-1) == truth
        result[name] = dict(held_correct=int(good[data['held']].sum()),
                            unseen_correct=int(good[unseen].sum()))
        assert good[data['held']].all()
    result['status'] = 'PASS'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
