"""Independent released-model forward/target/causality audit of a saved fit."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import torch
from src.task import dataset
from src.model import GrokTransformer
from src.train import disk_guard

@torch.no_grad()
def main(a):
    torch.set_num_threads(2)
    blob=a.checkpoint.read_bytes()
    if a.snapshot:
        if a.snapshot.exists():raise FileExistsError('Refusing existing audit snapshot')
        disk_guard(a.snapshot.parent)
        a.snapshot.write_bytes(blob)
    state=torch.load(io.BytesIO(blob),weights_only=False,map_location='cpu')
    model=GrokTransformer().requires_grad_(False).eval()
    model.load_state_dict(state['model'])
    data=dataset();ids=data['held'];x=data['x'][ids]
    pairs=data['pairs'][ids]
    residue=(pairs[:,0]+pairs[:,1])%97
    lookup=torch.tensor([data['vocab'].index(str(i)) for i in range(97)])
    answer=lookup[torch.as_tensor(residue).long()]
    eos=torch.full_like(answer,data['vocab'].index('<|eos|>'))
    labels=torch.stack([answer,eos],-1)
    assert torch.equal(labels,data['y'][ids,-2:]),'Independent modular-addition target mismatch'
    logits=[];difference=0.
    for start in range(0,len(x),256):
        batch=x[start:start+256]
        original=model.ref(batch)[0]
        fused=model(batch)
        difference=max(difference,float((original-fused).abs().max()))
        logits.append(original[:,-2:])
    logits=torch.cat(logits);ok=logits.argmax(-1)==labels
    # Teacher-forced c is a future input at position5; it must not affect the
    # answer prediction at position4, even at trained weights.
    changed=x[:128].clone()
    changed[:,5]=lookup[(torch.as_tensor(residue[:128]).long()+1)%97]
    delta=float((model.ref(changed)[0][:,4]-model.ref(x[:128])[0][:,4]).abs().max())
    assert difference<3e-5,f'Released/fused logits differ by {difference}'
    assert delta==0,f'Teacher-forced answer leaked into preceding prediction: {delta}'
    result=dict(checkpoint=str(a.checkpoint),archived_checkpoint=str(a.snapshot) if a.snapshot else None,
                checkpoint_sha256=hashlib.sha256(blob).hexdigest(),
                step=state['step'],held_pairs=len(ids),train_pairs=len(data['train']),
                original_held_answer_accuracy=float(ok[:,0].float().mean()),
                original_held_rhs_accuracy=float(ok.all(-1).float().mean()),
                original_held_rhs_ce=float(torch.nn.functional.cross_entropy(logits.reshape(-1,239),labels.reshape(-1))),
                released_fused_max_logit_error=difference,trained_future_token_error=delta,
                source='reference/reference_grok_20260929/transformer.py',
                source_sha256=hashlib.sha256(Path('reference/reference_grok_20260929/transformer.py').read_bytes()).hexdigest(),
                independent_targets='c=(a+b)%97 computed from raw held-out input pairs',
                threads=2,status='PASS')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--snapshot',type=Path)
    main(p.parse_args())
