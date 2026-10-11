"""Reference equation task and original initialization, without Lightning/JAX."""
import ast
import hashlib
import itertools
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SEED = 9278700

def vocabulary():
    tree = ast.parse((ROOT/'reference/reference_grok_20260929/data.py').read_text())
    ops = next(ast.literal_eval(n.value) for n in tree.body
               if isinstance(n, ast.Assign) and any(isinstance(v, ast.Name)
               and v.id == 'VALID_OPERATORS' for v in n.targets))
    return ['<|eos|>', '='] + sorted(ops) + [str(i) for i in range(97)] + [
        ''.join(map(str, p)) for p in itertools.permutations(range(5))]

def dataset(fraction=.3):
    vocab = vocabulary(); ids = {v:i for i,v in enumerate(vocab)}
    pairs = np.array(list(itertools.product(range(97), repeat=2)), dtype=np.int64)
    c = (pairs[:,0] + pairs[:,1]) % 97
    x = np.array([[ids['<|eos|>'], ids[str(a)], ids['+'], ids[str(b)],
                   ids['='], ids[str(v)]] for (a,b),v in zip(pairs,c)], dtype=np.int64)
    y = np.full_like(x, -1); y[:,4] = x[:,5]; y[:,5] = ids['<|eos|>']
    order = np.random.RandomState(0).permutation(len(x)); n = round(fraction*len(x))
    assert len(vocab)==len(set(vocab))
    if fraction==.3:assert n==2823
    assert not set(order[:n]) & set(order[n:])
    return dict(x=torch.from_numpy(x), y=torch.from_numpy(y), pairs=pairs,
                order=order, train=order[:n], held=order[n:], vocab=vocab)

def reference_model(seed=SEED):
    path=ROOT/'reference/reference_grok_20260929/transformer.py'
    source=path.read_text().replace('import pytorch_lightning as pl','')
    ns={'__name__':'official_grok_reference'}; exec(compile(source,str(path),'exec'), ns)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return ns['Transformer'](2,4,128,0.,50,len(vocabulary()),'relu',weight_noise=0.).float()

def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'reference/reference_grok_20260929').glob('*.py'))}
