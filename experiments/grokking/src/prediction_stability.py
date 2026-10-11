"""Frozen, label-free prediction selectors; correctness is audited after commit.

Reads cached predictions only. No model, optimizer, training or gradients.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np


ENDPOINT = 3177
FUTURE = 3200
HISTORY = (2700, 2800, 2900, 3000, 3100, ENDPOINT)
RUNS = tuple(f'i{i}-n{j}' for i in range(3) for j in range(2))
COUNTS = (64, 256, 512)
RULES = ('confidence', 'temporal', 'replica')
MIN_FREE = 5*1024**3


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''): digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def disk_guard(path):
    if shutil.disk_usage(path).free < MIN_FREE: raise OSError('Artifact refused: free disk below5GiB')


def write_json(path, value):
    disk_guard(path.parent)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read_partitions(path):
    """Deserialize only the label-free top-level partitions object of the plan."""
    text = Path(path).read_text()
    match = re.search(r'(?m)^  "partitions":\s*', text)
    if match is None: raise ValueError('Missing top-level partitions in frozen indented plan')
    value, _ = json.JSONDecoder().raw_decode(text, match.end())
    if not isinstance(value, dict): raise ValueError('Partitions must be an object')
    return {key: value[key] for key in ('U', 'pseudo_pool', 'clean_eval')}


def group(pair):
    a, b = map(int, pair)
    return min(a, b), max(a, b)


def top1_gap(logits):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError('Expected finite full-vocabulary logits')
    top = np.partition(z, -2, axis=1)[:, -2:]
    return z.argmax(1), top.max(1)-top.min(1)


def select_pseudo_labels(*, pairs, candidate_indices, clean_eval_indices,
                        valid_token_ids, predicted_token, confidence_gap,
                        current_step, rule, count, historical_predictions=None,
                        historical_steps=(), peer_predictions=None, peer_step=None,
                        selection_step=ENDPOINT):
    """Select without truth, margins/CE, arithmetic checks or future inputs.

    Rankings: confidence; temporal→confidence; replica→temporal→confidence;
    ensemble: replica→confidence. Last tie is ascending original pair ID.
    """
    if current_step != selection_step or any(step > selection_step for step in historical_steps):
        raise ValueError('Selection time cutoff violated')
    if peer_predictions is not None and peer_step != selection_step:
        raise ValueError('Peer time cutoff violated')
    if rule not in (*RULES, 'ensemble') or not isinstance(count, int) or count < 1:
        raise ValueError('Unknown rule or invalid requested count')
    pairs = np.asarray(pairs, dtype=np.int64); n = len(pairs)
    predicted = np.asarray(predicted_token, dtype=np.int64)
    confidence = np.asarray(confidence_gap, dtype=np.float64)
    if pairs.shape != (n, 2) or predicted.shape != (n,) or confidence.shape != (n,):
        raise ValueError('Selection feature dimensions differ')
    if not np.isfinite(confidence).all() or np.any(confidence < 0):
        raise ValueError('Invalid predicted confidence gaps')
    pool = np.asarray(candidate_indices, dtype=np.int64)
    clean = np.asarray(clean_eval_indices, dtype=np.int64)
    for ids in (pool, clean):
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= n)):
            raise ValueError('Invalid partition indices')
    clean_groups = {group(pairs[i]) for i in clean}
    if any(group(pairs[i]) in clean_groups for i in pool):
        raise ValueError('Pseudo pool overlaps reserved evaluation reversal groups')
    temporal = np.zeros(n); replica = np.zeros(n)
    if historical_predictions is not None:
        history = np.asarray(historical_predictions, dtype=np.int64)
        if history.shape != (len(historical_steps), n) or len(set(historical_steps)) != len(historical_steps):
            raise ValueError('Historical dimensions or duplicate times differ')
        if not len(history): raise ValueError('Empty supplied history')
        if selection_step in historical_steps:
            if not np.array_equal(history[list(historical_steps).index(selection_step)], predicted):
                raise ValueError('Current top1 differs from endpoint history')
        temporal = (history == predicted[None]).mean(0)
    if rule in ('temporal', 'replica') and historical_predictions is None:
        raise ValueError('Rule requires pre-endpoint history')
    if peer_predictions is not None:
        peers = np.asarray(peer_predictions, dtype=np.int64)
        if peers.ndim != 2 or peers.shape[1] != n or not len(peers):
            raise ValueError('Peer prediction dimensions differ')
        replica = (peers == predicted[None]).mean(0)
    if rule in ('replica', 'ensemble') and peer_predictions is None:
        raise ValueError('Rule requires current replica predictions')
    valid = set(valid_token_ids)
    eligible = [int(i) for i in pool if predicted[i] in valid]
    if rule == 'confidence': priority = lambda i: (-confidence[i], i)
    elif rule == 'temporal': priority = lambda i: (-temporal[i], -confidence[i], i)
    elif rule == 'replica': priority = lambda i: (-replica[i], -temporal[i], -confidence[i], i)
    else: priority = lambda i: (-replica[i], -confidence[i], i)
    ordered = sorted(eligible, key=priority)
    unique = []; seen = set()
    for i in ordered:
        key = group(pairs[i])
        if key not in seen: unique.append(i); seen.add(key)
    enough = len(unique) >= count
    ids = unique[:count] if enough else []
    return dict(rule=rule, requested_count=count, status='SELECTED' if enough else 'INSUFFICIENT_POOL',
                eligible_ordered=len(eligible), eligible_unordered=len(unique),
                selected_indices=ids, pseudo_tokens=predicted[ids].tolist(),
                selected_features=[dict(confidence_gap=float(confidence[i]),
                                        temporal_agreement=float(temporal[i]),
                                        replica_agreement=float(replica[i])) for i in ids])


def read_top1(path, expected_step):
    """Access only past predicted tokens, inputs and label-independent masks."""
    with np.load(path, allow_pickle=False) as saved:
        if int(saved['step']) != expected_step: raise ValueError('Historical snapshot step mismatch')
        return {key: saved[key].copy() for key in
                ('predicted_token', 'pairs', 'train', 'unseen_pair')}


def read_distribution_inputs(path, expected_step):
    with np.load(path, allow_pickle=False) as saved:
        if int(saved['step']) != expected_step: raise ValueError('Distribution step mismatch')
        return {key: saved[key].copy() for key in ('answer_logits', 'pairs', 'train', 'unseen_pair')}


def validate_current_meta(meta, run):
    i, j = int(run[1]), int(run[-1])
    expected = dict(run=run, step=ENDPOINT, method='dust', inherited_backprop_steps=0,
                    current_backprop_steps=0, model_seed=9278700+i, noise_seed=9278901+j)
    if any(meta.get(k) != v for k, v in expected.items()): raise ValueError('Current replica provenance mismatch')


def load_selection_inputs(predictions, distributions, plan_path):
    partitions = read_partitions(plan_path)
    sources = {}; current = {}; histories = {}; metas = {}; reference = None
    for run in RUNS:
        path = distributions/run/f'step-{ENDPOINT:05d}.npz'
        meta_path = path.with_suffix('.json'); meta = json.loads(meta_path.read_text())
        validate_current_meta(meta, run)
        if sha(path) != meta['artifact_sha256']: raise ValueError('Distribution checksum mismatch')
        now = read_distribution_inputs(path, ENDPOINT)
        if reference is None: reference = now
        else:
            for key in ('pairs', 'train', 'unseen_pair'):
                np.testing.assert_array_equal(now[key], reference[key])
        if now['answer_logits'].shape != (9409, 239): raise ValueError('Frozen task/vocabulary dimensions changed')
        if metas and (meta['vocabulary'] != next(iter(metas.values()))['vocabulary'] or
                      meta['split_sha256'] != next(iter(metas.values()))['split_sha256']):
            raise ValueError('Replica vocabulary/split mismatch')
        snapshots = []
        for step in HISTORY:
            historic_path = predictions/run/'predictions'/f'step-{step:05d}.npz'
            historic = read_top1(historic_path, step)
            for key in ('pairs', 'train', 'unseen_pair'):
                np.testing.assert_array_equal(historic[key], reference[key])
            snapshots.append(historic['predicted_token'])
            sources[str(historic_path)] = sha(historic_path)
        pred, _ = top1_gap(now['answer_logits'])
        np.testing.assert_array_equal(pred, snapshots[-1])
        # Preserve only label-independent provenance; do not pass the extractor's
        # true-CE parity audit scalar to the selection manifest.
        safe_keys = ('run', 'step', 'method', 'inherited_backprop_steps', 'current_backprop_steps',
                     'model_seed', 'noise_seed', 'checkpoint_sha256', 'split_sha256', 'vocabulary',
                     'source_hashes', 'artifact_sha256', 'execution_migration')
        current[run] = now['answer_logits']; histories[run] = np.stack(snapshots)
        metas[run] = {key:meta[key] for key in safe_keys if key in meta}
        sources[str(path)] = sha(path); sources[str(meta_path)] = sha(meta_path)
    pairs = reference['pairs']; U = np.flatnonzero(reference['unseen_pair']).tolist()
    if len(U) != 4669 or set(U) != set(partitions['U']): raise ValueError('Original unseen universe changed')
    if set(partitions['pseudo_pool']) & set(partitions['clean_eval']) or \
       set(partitions['pseudo_pool']) | set(partitions['clean_eval']) != set(U):
        raise ValueError('Frozen pool/evaluation partition changed')
    if len(metas[RUNS[0]]['vocabulary']) != 239: raise ValueError('Vocabulary changed')
    sources[str(plan_path)] = sha(plan_path)
    return pairs, partitions, current, histories, metas, sources


def build_manifest(pairs, partitions, current, histories, metas, sources):
    vocabulary = metas[RUNS[0]]['vocabulary']
    valid = [vocabulary.index(str(i)) for i in range(97)]
    predicted = {}; confidence = {}
    for run in RUNS: predicted[run], confidence[run] = top1_gap(current[run])
    peer = np.stack([predicted[run] for run in RUNS])
    selections = []
    shared = dict(pairs=pairs, candidate_indices=partitions['pseudo_pool'],
                  clean_eval_indices=partitions['clean_eval'], valid_token_ids=valid,
                  current_step=ENDPOINT, peer_predictions=peer, peer_step=ENDPOINT)
    for run in RUNS:
        for rule in RULES:
            for count in COUNTS:
                chosen = select_pseudo_labels(**shared, predicted_token=predicted[run],
                    confidence_gap=confidence[run], historical_predictions=histories[run],
                    historical_steps=HISTORY, rule=rule, count=count)
                selections.append(dict(teacher=run, **chosen))
    mean_logits = np.mean(np.stack([current[r] for r in RUNS]).astype(np.float64), axis=0)
    ensemble_pred, ensemble_gap = top1_gap(mean_logits)
    for count in COUNTS:
        selections.append(dict(teacher='six_logit_mean', **select_pseudo_labels(**shared,
            predicted_token=ensemble_pred, confidence_gap=ensemble_gap, rule='ensemble', count=count)))
    manifest = dict(format_version=1, selection_step=ENDPOINT, history_steps=list(HISTORY),
        future_inputs_used=False, target_margin_ce_correctness_used=False,
        training_updates=0, new_model_forwards=0, exploratory_not_blinded=True,
        rules=dict(confidence='current full239 top1-top2 gap',
                   temporal='historical agreement then current gap',
                   replica='current six-replica vote agreement then temporal agreement then gap',
                   ensemble='uniform six-member raw-logit mean labels; replica vote agreement then mean-logit gap'),
        final_tie='ascending global pair index', one_ordering_per_unordered_group=True,
        count_requests=list(COUNTS), partition_sizes={k:len(v) for k,v in partitions.items()},
        input_sha256=sources, source_sha256=sha(__file__), protocol_sha256=sha('PROTOCOL.md'),
        replica_provenance=metas, selections=selections,
        teacher_cost=dict(confidence_model_states=1, temporal_model_states=6,
                          replica_model_states=11, ensemble_model_states=6,
                          note='Existing cached states; six replicas share a split; no new inference/training. Replica rule needs six current states and five earlier states of the named model.'))
    return manifest, predicted, ensemble_pred


def commit_selection(output, manifest):
    path = output/'selection_manifest.json'; write_json(path, manifest)
    commit = dict(selection_manifest_sha256=sha(path), canonical_manifest_sha256=canonical_hash(manifest),
                  committed_utc=datetime.now(timezone.utc).isoformat(),
                  future_step=FUTURE, future_predictions_opened=False, labels_opened=False)
    write_json(output/'selection_commit.json', commit)
    return commit


def label_distribution(tokens, vocabulary):
    return {vocabulary[k]:int(v) for k,v in sorted(Counter(map(int, tokens)).items())}


def audit_transition(pseudo, future, truth):
    correct = pseudo == truth; later_correct = future == truth; retained = pseudo == future
    return dict(count=len(pseudo), pseudo_correct=int(correct.sum()),
        pseudo_accuracy=float(correct.mean()) if len(pseudo) else None,
        retained=int(retained.sum()), retention=float(retained.mean()) if len(pseudo) else None,
        future_correct=int(later_correct.sum()), future_accuracy=float(later_correct.mean()) if len(pseudo) else None,
        stable_correct=int((correct & retained).sum()), stable_wrong=int((~correct & retained).sum()),
        correct_lost=int((correct & ~retained).sum()), wrong_corrected=int((~correct & later_correct).sum()),
        wrong_to_different_wrong=int((~correct & ~retained & ~later_correct).sum()),
        wrong_label_retention=int((~correct & retained).sum())/int((~correct).sum()) if (~correct).any() else None)


def evaluate_after_commit(output, distributions, pairs, partitions, current, histories, metas,
                          manifest, commit, predicted, ensemble_pred):
    manifest_path = output/'selection_manifest.json'; commit_path = output/'selection_commit.json'
    if not manifest_path.exists() or not commit_path.exists(): raise ValueError('Selection is not committed')
    if json.loads(commit_path.read_text()) != commit or \
       sha(manifest_path) != commit['selection_manifest_sha256'] or \
       canonical_hash(manifest) != commit['canonical_manifest_sha256']:
        raise ValueError('Frozen selection object or commit differs')
    truth = None; future_logits = {}; future_predictions = {}; future_sources = {}
    for run in RUNS:
        current_path = distributions/run/f'step-{ENDPOINT:05d}.npz'
        with np.load(current_path, allow_pickle=False) as saved:
            targets = saved['target_token'].copy()
        if truth is None: truth = targets
        else: np.testing.assert_array_equal(truth, targets)
        path = distributions/run/f'step-{FUTURE:05d}.npz'; meta_path = path.with_suffix('.json')
        meta = json.loads(meta_path.read_text())
        expected = dict(run=run, step=FUTURE, method='dust', inherited_backprop_steps=0,
                        current_backprop_steps=0, model_seed=metas[run]['model_seed'],
                        noise_seed=metas[run]['noise_seed'], vocabulary=metas[run]['vocabulary'],
                        split_sha256=metas[run]['split_sha256'])
        if any(meta.get(k) != v for k,v in expected.items()) or sha(path) != meta['artifact_sha256']:
            raise ValueError('Future artifact provenance mismatch')
        with np.load(path, allow_pickle=False) as saved:
            if int(saved['step']) != FUTURE: raise ValueError('Future step mismatch')
            np.testing.assert_array_equal(saved['pairs'], pairs)
            np.testing.assert_array_equal(saved['target_token'], truth)
            future_logits[run] = saved['answer_logits'].copy()
        future_predictions[run], _ = top1_gap(future_logits[run])
        future_sources[str(path)] = sha(path); future_sources[str(meta_path)] = sha(meta_path)
    vocabulary = metas[RUNS[0]]['vocabulary']
    future_ensemble, _ = top1_gap(np.mean(np.stack([future_logits[r] for r in RUNS]).astype(np.float64), axis=0))
    rows = []; details = []
    for selection in manifest['selections']:
        row = {k:selection[k] for k in ('teacher', 'rule', 'requested_count', 'status', 'eligible_unordered')}
        ids = np.asarray(selection['selected_indices'], dtype=np.int64)
        pseudo = np.asarray(selection['pseudo_tokens'], dtype=np.int64)
        later = future_ensemble if selection['teacher'] == 'six_logit_mean' else future_predictions[selection['teacher']]
        row.update(audit_transition(pseudo, later[ids], truth[ids]))
        row.update(pool_ordered_coverage=len(ids)/len(partitions['pseudo_pool']),
                   pool_unordered_coverage=len(ids)/len({group(pairs[i]) for i in partitions['pseudo_pool']}),
                   pseudo_class_distribution=label_distribution(pseudo, vocabulary),
                   mean_temporal_agreement=float(np.mean([f['temporal_agreement'] for f in selection['selected_features']])) if len(ids) else None,
                   mean_replica_agreement=float(np.mean([f['replica_agreement'] for f in selection['selected_features']])) if len(ids) else None)
        rows.append(row)
        for j, i in enumerate(ids):
            details.append(dict(teacher=row['teacher'], rule=row['rule'], requested_count=row['requested_count'],
                pair_id=int(i), a=int(pairs[i,0]), b=int(pairs[i,1]), pseudo_token=vocabulary[pseudo[j]],
                true_token=vocabulary[truth[i]], future_token=vocabulary[later[i]],
                pseudo_correct=bool(pseudo[j] == truth[i]), retained=bool(pseudo[j] == later[i]),
                future_correct=bool(later[i] == truth[i]), **selection['selected_features'][j]))
    baselines = {}
    for run in RUNS:
        stats = {}
        for label in ('U', 'pseudo_pool', 'clean_eval'):
            ids = np.asarray(partitions[label], dtype=np.int64)
            stats[label] = audit_transition(predicted[run][ids], future_predictions[run][ids], truth[ids])
        temporal = (histories[run] == predicted[run][None]).all(0)
        replica = (np.stack([predicted[r] for r in RUNS]) == predicted[run][None]).all(0)
        for label, mask in [('all_six_times_agree', temporal), ('all_six_replicas_agree', replica)]:
            ids = np.asarray([i for i in partitions['U'] if mask[i]], dtype=np.int64)
            stats[label] = audit_transition(predicted[run][ids], future_predictions[run][ids], truth[ids])
        baselines[run] = stats
    U = np.asarray(partitions['U'], dtype=np.int64)
    baselines['six_logit_mean'] = dict(U=audit_transition(ensemble_pred[U], future_ensemble[U], truth[U]))
    for path, expected in {**manifest['input_sha256'], **future_sources}.items():
        if sha(path) != expected: raise ValueError('Input artifact changed during diagnostic')
    if sha(output/'selection_manifest.json') != commit['selection_manifest_sha256']:
        raise ValueError('Frozen selection manifest changed')
    result = dict(selection_commit=commit, evaluation_future_step=FUTURE,
        truth_used_only_after_commit=True, future_used_only_after_commit=True,
        new_model_forwards=0, training_updates=0, source_artifacts_unchanged=True,
        rows=rows, baselines=baselines, future_input_sha256=future_sources,
        caution='Exploratory after observed pilot outcomes; ordered reversal pairs and shared-split replicas are dependent. Confidence/stability are selection features, not correctness guarantees. No self-training or MDL result.')
    write_json(output/'results.json', result)
    disk_guard(output)
    with (output/'selected_examples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]) if details else ['teacher'])
        writer.writeheader(); writer.writerows(details)
    disk_guard(output)
    with (output/'summary.csv').open('w', newline='') as stream:
        fields = [k for k in rows[0] if k != 'pseudo_class_distribution']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    return result


def main(args):
    start = time.monotonic(); args.output.mkdir(parents=True, exist_ok=False); disk_guard(args.output)
    inputs = load_selection_inputs(args.predictions, args.distributions, args.plan)
    pairs, partitions, current, histories, metas, sources = inputs
    manifest, predicted, ensemble = build_manifest(*inputs)
    commit = commit_selection(args.output, manifest)
    result = evaluate_after_commit(args.output, args.distributions, pairs, partitions, current, histories,
                                   metas, manifest, commit, predicted, ensemble)
    write_json(args.output/'completion.json', dict(state='FROZEN_STABILITY_AUDIT_COMPLETE',
        seconds=time.monotonic()-start, selection_manifest_sha256=commit['selection_manifest_sha256'],
        selections=len(manifest['selections']), source_artifacts_unchanged=True,
        new_model_forwards=0, training_updates=0))
    print(json.dumps(dict(output=str(args.output), seconds=time.monotonic()-start,
        i1_n1=[r for r in result['rows'] if r['teacher']=='i1-n1'],
        ensemble=[r for r in result['rows'] if r['teacher']=='six_logit_mean']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/prediction-stability-20261011'))
    parser.add_argument('--predictions', type=Path, default=Path('runs/seed-fast-collected-20261011'))
    parser.add_argument('--distributions', type=Path, default=Path('runs/ensemble-diagnostic-20261011/distributions'))
    parser.add_argument('--plan', type=Path, default=Path('runs/followup-data-20261011/plan.json'))
    main(parser.parse_args())
