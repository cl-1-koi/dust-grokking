"""Released equation vocabulary/format, four full-batch calibration splits."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from ttt_basin_control import write


def setup(root):
    root.mkdir(exist_ok=True,parents=True)
    source=Path('reference_grok_20260929/data.py').read_text();module=ast.parse(source)
    ops=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='VALID_OPERATORS' for x in n.targets))
    vocab=['<|eos|>','=']+sorted(ops)+[str(n) for n in range(97)]+[''.join(map(str,p)) for p in itertools.permutations(range(5))]
    assert len(vocab)==len(set(vocab));ids={s:i for i,s in enumerate(vocab)}
    configs=[];padded=[];masks=[];ys=[]
    for group,order,fractions in [('add97',97,[.3,.5,.7]),('S5',120,[.5])]:
        pair=np.array(list(itertools.product(range(order),repeat=2)));perms=np.array(list(itertools.permutations(range(5))))
        if group=='add97':
            names=[str(i) for i in range(order)];op='+';target=(pair[:,0]+pair[:,1])%97
        else:
            names=[''.join(map(str,p)) for p in perms];lookup={tuple(p):i for i,p in enumerate(perms)};op='s5'
            target=np.array([lookup[tuple(perms[b][perms[a]])] for a,b in pair])
            table=target.reshape(120,120)
            assert np.array_equal(table[0],np.arange(120)) and np.array_equal(table[:,0],np.arange(120))
            for a in range(120):np.testing.assert_array_equal(table[table[a,:,None],np.arange(120)[None]],table[a,table])
        tokens=np.array([[ids['<|eos|>'],ids[names[a]],ids[op],ids[names[b]],ids['='],ids[names[c]]] for (a,b),c in zip(pair,target)],np.int32)
        labels=np.column_stack([tokens[:,-1],np.full(len(pair),ids['<|eos|>'])]).astype(np.int32)
        perm=np.random.RandomState(0).permutation(len(pair))
        np.savez_compressed(root/f'{group}_data.npz',tokens=tokens,labels=labels,permutation=perm,target_class=target)
        for fraction in fractions:
            n=round(fraction*len(pair));train=perm[:n];held=perm[n:]
            assert not set(train)&set(held) and len(train)+len(held)==len(pair)
            configs.append(dict(key=f'{group}_frac{fraction:g}_seed9278700',group=group,fraction=fraction,seed=9278700,
                train_pairs=n,heldout_pairs=len(held),vocab_size=len(vocab)))
            padded.append(tokens[np.pad(train,(0,7200-n),mode='edge')]);ys.append(labels[np.pad(train,(0,7200-n),mode='edge')])
            masks.append(np.arange(7200)<n)
    np.savez_compressed(root/'training_data.npz',tokens=padded,labels=ys,mask=masks)
    write(root/'design.json',dict(configs=configs,seed=9278700,split_seed=0,vocabulary=vocab,requested_horizon=100000,eval_every=500,
        input_format='EOS a op b = c; CE targets c and EOS at last two positions',composition='b[a], official convention',
        architecture='Power official two-layer decoder width128 heads4 ReLU post-LN',optimizer='AdamW lr.001 beta(.9,.98) wd1 warmup10 no clipping',
        batch='full training set; padding has exactly zero loss weight',reference_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('reference_grok_20260929').glob('*.py')}))
    write(root/'data_audit.json',dict(status='PASS',split='nested official RandomState(0) shuffle',s5_associativity='PASS',vocab_size=len(vocab),padding='ZERO_WEIGHT'))


if __name__=='__main__':setup(Path(sys.argv[1]))
