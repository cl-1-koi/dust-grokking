"""Frozen, forward-only probability ensembles; no training or fitted aggregation."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from .model import GrokTransformer
from .seed_train import masks
from .task import dataset, hashes
from .train import disk_guard, write_json

STEPS = (200, 1000, 2000, 3000, 3177, 3200)
NAMES = tuple(f'i{i}-n{j}' for i in range(3) for j in range(2))
GROUPS = {'all_six': NAMES,
          **{f'noise{j}': tuple(f'i{i}-n{j}' for i in range(3)) for j in range(2)},
          **{f'init{i}': tuple(f'i{i}-n{j}' for j in range(2)) for i in range(3)}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim < 2 or not np.isfinite(z).all():
        raise ValueError('Expected finite logits with a class axis')
    shifted = z-z.max(axis=-1, keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def combine(logits):
    """No labels enter any combination rule. Ties use lowest vocabulary ID."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 3 or not z.shape[0]:
        raise ValueError('Expected nonempty [members, examples, vocabulary]')
    member_logp = log_softmax(z)
    member_p = np.exp(member_logp)
    # Log-sum-exp across members preserves CE even if true probability underflows.
    peak = member_logp.max(axis=0)
    mixture_logp = peak+np.log(np.exp(member_logp-peak).sum(axis=0))-np.log(len(z))
    logits_logp = log_softmax(z.mean(axis=0))
    member_pred = z.argmax(axis=-1)
    votes = np.zeros(z.shape[1:], dtype=np.int64)
    for p in member_pred:
        votes[np.arange(len(p)), p] += 1
    return dict(member_logp=member_logp, member_probability=member_p,
                member_pred=member_pred, probability_logp=mixture_logp,
                logit_mean_logp=logits_logp, votes=votes,
                probability_pred=mixture_logp.argmax(-1),
                logit_mean_pred=logits_logp.argmax(-1), plurality_pred=votes.argmax(-1),
                vote_tie=(votes == votes.max(-1, keepdims=True)).sum(-1) > 1)


def per_example(logp, truth):
    truth = np.asarray(truth, dtype=np.int64)
    if logp.ndim != 2 or truth.shape != (len(logp),):
        raise ValueError('Mismatched truth/log-probability shapes')
    p = np.exp(logp); rows = np.arange(len(p)); pred = logp.argmax(-1)
    true_logp = logp[rows, truth]; true_p = p[rows, truth]
    rank = 1+(logp > true_logp[:, None]).sum(-1)
    rank += ((logp == true_logp[:, None]) & (np.arange(p.shape[1]) < truth[:, None])).sum(-1)
    top = np.partition(p, -2, axis=-1)[:, -2:]
    return dict(prediction=pred, correct=pred == truth, ce=-true_logp,
                true_probability=true_p, true_rank=rank,
                brier=np.square(p).sum(-1)-2*true_p+1,
                entropy=-(p*logp).sum(-1), top1_probability_gap=top[:, 1]-top[:, 0])


def summarize(values, selected, residue_tokens):
    n = int(selected.sum())
    out = dict(count=n, correct=int(values['correct'][selected].sum()))
    out['accuracy'] = out['correct']/n
    out['invalid_token_predictions'] = int((~np.isin(values['prediction'][selected], residue_tokens)).sum())
    for key in ('ce', 'true_probability', 'true_rank', 'brier', 'entropy', 'top1_probability_gap'):
        if key in values:
            out['mean_'+key] = float(values[key][selected].mean())
    if 'true_rank' in values:
        out['median_true_rank'] = float(np.median(values['true_rank'][selected]))
        out['true_rank_le5'] = int((values['true_rank'][selected] <= 5).sum())
    return out


def validate_config(config, run, data):
    match = re.fullmatch(r'i([0-2])-n([0-1])', run)
    if not match:
        raise ValueError('Unknown replica')
    i, j = map(int, match.groups())
    expected = dict(method='dust', model_seed=9278700+i, noise_seed=9278901+j,
                    inherited_backprop_steps=0, zeroth_order_only=True,
                    split_sha256=hashlib.sha256(data['order'].tobytes()).hexdigest(),
                    vocabulary=data['vocab'])
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError('Replica provenance mismatch: '+key)
    for path in ('src/model.py', 'src/task.py', 'src/dust.py', 'src/train.py'):
        if config['source_hashes'][path] != sha(path):
            raise ValueError('Model/task source changed: '+path)
    if config['reference_hashes'] != hashes():
        raise ValueError('Reference source changed')
    migration = config.get('execution_migration')
    if migration:
        for path, digest in migration['execution_source_sha256'].items():
            if sha(path) != digest:
                raise ValueError('CPU execution source changed: '+path)
    settings = config['dust_settings']
    for key, value in dict(population=16, nrep=4, direct_sigma=.002, hub_last_sigma=.004,
                           local_sigma=.05, head_sigma=.05).items():
        if settings[key] != value:
            raise ValueError('Dust recipe mismatch: '+key)


@torch.no_grad()
def forward_logits(model, data, chunk=256):
    model.eval(); model.requires_grad_(False)
    out = []
    for start in range(0, len(data['x']), chunk):
        x = data['x'][start:start+chunk]
        z = model(x)[:, 4]
        if start == 0:
            altered = x.clone(); altered[:, 5] = data['vocab'].index('<|eos|>')
            if not torch.equal(z, model(altered)[:, 4]):
                raise AssertionError('Future answer input affected causal prediction')
        out.append(z.cpu().numpy())
    return np.concatenate(out), len(data['x'])+min(chunk, len(data['x']))


def extract(root, run, step, threads=2):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    start = time.monotonic(); data = dataset()
    source = root/'checkpoints'/run
    config_path = source/f'config-step-{step:05d}.json'
    if not config_path.exists(): config_path = source/'config.json'
    config = json.loads(config_path.read_text())
    validate_config(config, run, data)
    with np.load(source/'split.npz') as split:
        for k, value in [('pairs', data['pairs']), ('train', data['train']), ('held', data['held']),
                         ('tokens', data['x'].numpy()), ('labels', data['y'].numpy())]:
            np.testing.assert_array_equal(split[k], value)
    checkpoint = source/f'step-{step:05d}.pt'; checksum = sha(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['step'] != step or step not in STEPS:
        raise ValueError('Checkpoint step mismatch')
    migration = config.get('execution_migration')
    if migration and step > migration['cutover_step']:
        if state.get('execution', {}).get('path') != 'cpu' or state['execution']['cutover_step'] != migration['cutover_step']:
            raise ValueError('Checkpoint execution lineage mismatch')
    output = root/'distributions'/run; output.mkdir(parents=True, exist_ok=True)
    destination = output/f'step-{step:05d}.npz'
    if destination.exists() or destination.with_suffix('.json').exists():
        raise FileExistsError(destination)
    model = GrokTransformer(seed=config['model_seed'])
    model.load_state_dict(state['model'], strict=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    logits, forwards = forward_logits(model, data)
    lp = log_softmax(logits); values = per_example(lp, data['y'][:, 4].numpy())
    with np.load(source/f'reference-step-{step:05d}.npz') as ref:
        if int(ref['step']) != step:
            raise ValueError('Prediction reference step mismatch')
        np.testing.assert_array_equal(ref['pairs'], data['pairs'])
        np.testing.assert_array_equal(ref['predicted_token'], values['prediction'])
        np.testing.assert_allclose(ref['answer_ce'], values['ce'], atol=3e-5, rtol=2e-5)
        max_ce_difference = float(np.max(np.abs(ref['answer_ce']-values['ce'])))
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert sha(checkpoint) == checksum
    train, unseen, reverse = masks(data)
    disk_guard(root)
    temporary = destination.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, answer_logits=logits, probability=np.exp(lp),
                        target_token=data['y'][:, 4].numpy(), pairs=data['pairs'],
                        train=train, unseen_pair=unseen, reversed_train=reverse,
                        step=np.array(step), **values)
    temporary.replace(destination)
    meta = dict(run=run, step=step, method='dust', inherited_backprop_steps=0,
                current_backprop_steps=0, retained_updates=0, model_seed=config['model_seed'],
                noise_seed=config['noise_seed'], checkpoint_sha256=checksum,
                split_sha256=config['split_sha256'], vocabulary=config['vocabulary'],
                source_hashes={p: sha(p) for p in ('src/ensemble_audit.py', 'src/model.py', 'src/task.py')},
                protocol_sha256=sha('PROTOCOL.md'), forward_examples=forwards, threads=threads,
                seconds=time.monotonic()-start, future_invariance=True,
                reference_argmax_exact=True, reference_max_ce_difference=max_ce_difference,
                checkpoint_unchanged=True, artifact_sha256=sha(destination))
    if migration:
        meta['execution_migration'] = migration
        meta['checkpoint_execution'] = state.get('execution')
        meta['config_snapshot_sha256'] = sha(config_path)
    write_json(destination.with_suffix('.json'), meta)
    return meta


def validate_members(names, metas, arrays, step):
    if len(names) != len(set(names)) or not names or len(names) != len(metas) or len(names) != len(arrays):
        raise ValueError('Empty or duplicate member names')
    reference = metas[0]
    for name, meta, arr in zip(names, metas, arrays):
        if meta['run'] != name or meta['step'] != step or int(arr['step']) != step:
            raise ValueError('Member/step mismatch')
        match = re.fullmatch(r'i([0-2])-n([0-1])', name)
        if match is None or meta.get('model_seed') != 9278700+int(match[1]) or meta.get('noise_seed') != 9278901+int(match[2]):
            raise ValueError('Member seed mismatch')
        if meta.get('method') != 'dust' or meta.get('inherited_backprop_steps') != 0:
            raise ValueError('Not a pure Dust member')
        for key in ('split_sha256', 'vocabulary'):
            if meta[key] != reference[key]:
                raise ValueError('Member data mismatch: '+key)
        if meta.get('execution_migration', {}).get('cutover_step') != reference.get('execution_migration', {}).get('cutover_step'):
            raise ValueError('Member execution schedule mismatch')
        for key in ('target_token', 'pairs', 'train', 'unseen_pair', 'reversed_train'):
            if not np.array_equal(arr[key], arrays[0][key]):
                raise ValueError('Member array mismatch: '+key)
        if arr['answer_logits'].shape != (len(arr['pairs']), len(meta['vocabulary'])):
            raise ValueError('Logit shape mismatch')


def reversal_summary(correct, selected, pairs):
    blocks = {}
    for ix in np.flatnonzero(selected):
        a, b = map(int, pairs[ix]); key = (min(a, b), max(a, b))
        blocks.setdefault(key, []).append(bool(correct[ix]))
    return dict(unordered_groups=len(blocks), all_observed_orientations_correct=sum(all(v) for v in blocks.values()),
                some_but_not_all_correct=sum(any(v) and not all(v) for v in blocks.values()),
                none_correct=sum(not any(v) for v in blocks.values()),
                one_observed_orientation=sum(len(v) == 1 for v in blocks.values()))


def analyze_group(root, step, group, names):
    output = root/'reports'/f'step-{step:05d}'/group
    if (output/'summary.json').exists():
        return json.loads((output/'summary.json').read_text())
    arrays, metas = [], []
    for name in names:
        path = root/'distributions'/name/f'step-{step:05d}.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        if sha(path) != meta['artifact_sha256']:
            raise ValueError('Distribution artifact hash mismatch')
        arrays.append(dict(np.load(path))); metas.append(meta)
    validate_members(names, metas, arrays, step)
    truth = arrays[0]['target_token']; pair = arrays[0]['pairs']; vocab = metas[0]['vocabulary']
    subsets = {'train': arrays[0]['train'], 'held': ~arrays[0]['train'],
               'unseen_pair': arrays[0]['unseen_pair'], 'reversed_train': arrays[0]['reversed_train']}
    if {k: int(v.sum()) for k, v in subsets.items()} != dict(train=2823, held=6586, unseen_pair=4669, reversed_train=1917):
        raise ValueError('Evaluation universe changed')
    out = combine(np.stack([a['answer_logits'] for a in arrays]))
    individual = [per_example(lp, truth) for lp in out['member_logp']]
    selected_index = min(range(len(names)), key=lambda k: (individual[k]['ce'][subsets['train']].mean(), names[k]))
    baseline = individual[selected_index]['correct']
    member_correct = out['member_pred'] == truth[None]
    union = member_correct.any(0)
    residue_tokens = np.array([vocab.index(str(i)) for i in range(97)])
    values = {name: v for name, v in zip(names, individual)}
    values['probability_mean'] = per_example(out['probability_logp'], truth)
    values['logit_mean'] = per_example(out['logit_mean_logp'], truth)
    values['plurality'] = dict(prediction=out['plurality_pred'], correct=out['plurality_pred'] == truth)
    summary = dict(step=step, group=group, members=list(names), train_selected_member=names[selected_index],
                   selection_rule='Lowest full training answer CE; ties lexicographic member ID',
                   probability_rule='Uniform softmax probability mean, temperature1, all239classes',
                   plurality_tie_rule='Lowest vocabulary token ID', individual_inference_models=1,
                   ensemble_inference_models=len(names), subsets={}, provenance=metas)
    for label, mask in subsets.items():
        row = dict(metrics={name: summarize(v, mask, residue_tokens) for name, v in values.items()},
                   top1_union_count=int(union[mask].sum()), vote_ties=int(out['vote_tie'][mask].sum()),
                   member_correct_count_histogram=np.bincount(member_correct[:, mask].sum(0), minlength=len(names)+1).tolist(),
                   descriptive_best_member_correct=max(int(v['correct'][mask].sum()) for v in individual),
                   ensembles={})
        for name in ('probability_mean', 'logit_mean', 'plurality'):
            correct = values[name]['correct']
            row['ensembles'][name] = dict(correct_outside_top1_union=int((correct & ~union & mask).sum()),
                  lost_from_top1_union=int((~correct & union & mask).sum()),
                  gained_vs_train_selected=int((correct & ~baseline & mask).sum()),
                  lost_vs_train_selected=int((~correct & baseline & mask).sum()),
                  reversal_blocks=reversal_summary(correct, mask, pair))
        summary['subsets'][label] = row
    output.mkdir(parents=True, exist_ok=True); disk_guard(root)
    temporary = output/'pairs.tmp.npz'
    np.savez_compressed(temporary, pairs=pair, target_token=truth,
                        member_predictions=out['member_pred'], member_correct=member_correct,
                        member_true_probability=np.stack([v['true_probability'] for v in individual]),
                        member_true_rank=np.stack([v['true_rank'] for v in individual]),
                        probability_mean=np.exp(out['probability_logp']),
                        logit_mean_probability=np.exp(out['logit_mean_logp']),
                        probability_prediction=out['probability_pred'], logit_mean_prediction=out['logit_mean_pred'],
                        plurality_prediction=out['plurality_pred'], top1_union=union, **subsets)
    temporary.replace(output/'pairs.npz')
    with (output/'pairs.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['pair_id', 'a', 'b', 'truth', 'unseen_pair', 'probability_prediction',
                         'ensemble_correct', 'members_correct', 'correct_outside_union',
                         'ensemble_true_probability']+[n+s for n in names for s in ('_prediction', '_true_probability', '_true_rank')])
        for ix, (a, b) in enumerate(pair):
            correct = values['probability_mean']['correct'][ix]
            writer.writerow([ix, a, b, vocab[truth[ix]], int(subsets['unseen_pair'][ix]), vocab[out['probability_pred'][ix]],
                             int(correct), int(member_correct[:, ix].sum()), int(correct and not union[ix]),
                             values['probability_mean']['true_probability'][ix]]+
                            [value for j in range(len(names)) for value in
                             (vocab[out['member_pred'][j, ix]], individual[j]['true_probability'][ix], individual[j]['true_rank'][ix])])
    write_json(output/'summary.json', summary)
    return summary


def report(root, destination):
    torch.set_num_threads(1)
    summaries = []
    for step in STEPS:
        for group, names in GROUPS.items():
            if all((root/'distributions'/name/f'step-{step:05d}.json').exists() for name in names):
                summaries.append(analyze_group(root, step, group, names))
    text = ['# Frozen probability-ensemble diagnostic', '', 'Updated '+datetime.now(timezone.utc).isoformat()+'.', '',
            'Pure Dust only; frozen models, zero backward passes or retained updates. Primary endpoint: all six at3177. '
            'Early rows do not answer the trained9% question. Uniform rules were fixed before scoring; no held-label fitting.', '',
            '| Step | Group | Members | Train-selected single on U | Probability mean on U | Logit mean on U | Plurality on U | Soft gains outside top1 union | Union count |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        row = s['subsets']['unseen_pair']; m = row['metrics']
        text.append(f"| {s['step']} | {s['group']} | {len(s['members'])} | "
                    f"{m[s['train_selected_member']]['accuracy']:.2%} | {m['probability_mean']['accuracy']:.2%} | "
                    f"{m['logit_mean']['accuracy']:.2%} | {m['plurality']['accuracy']:.2%} | "
                    f"{row['ensembles']['probability_mean']['correct_outside_top1_union']} | {row['top1_union_count']} |")
    if not summaries:
        text += ['', 'Waiting for matched-step frozen probability exports.']
    for s in summaries:
        text += ['', f"## Step{s['step']} / {s['group']}", '',
                 'Members: '+', '.join(s['members'])+'. Train-selected member: '+s['train_selected_member']+'.', '',
                 '| Method | U correct/4669 | U answer CE | U mean true rank | U mean true probability | Train accuracy | All-held accuracy |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for method in list(s['members'])+['probability_mean', 'logit_mean', 'plurality']:
            u = s['subsets']['unseen_pair']['metrics'][method]
            ce = f"{u['mean_ce']:.6f}" if 'mean_ce' in u else '—'
            rank = f"{u['mean_true_rank']:.2f}" if 'mean_true_rank' in u else '—'
            prob = f"{u['mean_true_probability']:.6f}" if 'mean_true_probability' in u else '—'
            text.append(f"| {method} | {u['correct']} | {ce} | {rank} | {prob} | "
                        f"{s['subsets']['train']['metrics'][method]['accuracy']:.2%} | "
                        f"{s['subsets']['held']['metrics'][method]['accuracy']:.2%} |")
        artifact = root/'reports'/f"step-{s['step']:05d}"/s['group']
        text += ['', f'All per-equation distributions, member ranks and transitions: `{artifact}`.']
    text += ['', 'U is the fixed4669 ordered equations with neither orientation trained. Reversed pairs are dependent; '
             'group counts are saved without independent-example significance claims. The top1 union is not an upper bound '
             'on probability ensembling. Ensemble inference evaluates every member; it is not a single improved transformer.', '',
             'Plurality ties use lowest vocabulary ID and may therefore predict a nonresidue token. '
             'No accuracy-dependent stopping, member selection, temperature or weight fitting.', '']
    disk_guard(root)
    temp = destination.with_suffix(destination.suffix+'.tmp'); temp.write_text('\n'.join(text)); temp.replace(destination)
    write_json(root/'summary.json', dict(reports=summaries))
    return len(summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['extract', 'report'])
    parser.add_argument('--root', type=Path, default=Path('runs/ensemble-diagnostic-20261011'))
    parser.add_argument('--run'); parser.add_argument('--step', type=int)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--report', type=Path, default=Path('ENSEMBLE_RESULTS.md'))
    args = parser.parse_args()
    result = extract(args.root, args.run, args.step, args.threads) if args.action == 'extract' else report(args.root, args.report)
    print(json.dumps(result), flush=True)
