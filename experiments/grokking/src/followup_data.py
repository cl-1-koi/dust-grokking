"""Predeclared nested subsets and frozen-teacher reconstruction data.

Selection reads only original input coordinates/split and 239-way teacher
logits. Ground-truth targets are constructed AFTER fixing pseudo selection,
for retained original training labels, the explicit oracle arm, and audits.
No model, optimizer, gradient, or training execution occurs in this module.
"""
import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .task import dataset
from .train import disk_guard, write_json


GROUP_SEED = 9278910
POOL_SEED = 9278911
SHUFFLE_SEED = 9278912
TEACHER_SHA256 = '35ba0b317bee0976c755cf7a535a623f1267290ef62a1e279832ace7f38432c0'


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def teacher_answer_logits(teacher):
    """Ignore saved true-answer margins, CE, correctness and target metadata."""
    if isinstance(teacher, Mapping) or hasattr(teacher, 'files'):
        teacher = teacher['answer_logits']
    logits = np.asarray(teacher)
    if logits.ndim != 2 or logits.shape[1] < 2 or not np.isfinite(logits).all():
        raise ValueError('Teacher logits must be finite [pairs, vocabulary]')
    return logits


def select_pseudo_indices(teacher, candidate_indices, count, valid_token_ids):
    """Top1-top2 over ALL tokens; valid-residue top1; gap↓ then global ID↑."""
    logits = teacher_answer_logits(teacher)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.ndim != 1 or len(np.unique(candidates)) != len(candidates):
        raise ValueError('Candidate indices must be unique and one-dimensional')
    if np.any((candidates < 0) | (candidates >= len(logits))) or count < 0:
        raise ValueError('Invalid candidate indices or requested count')
    predicted = logits.argmax(1)
    top_two = np.partition(logits.astype(np.float64), -2, axis=1)[:, -2:]
    gap = top_two.max(1)-top_two.min(1)
    eligible = candidates[np.isin(predicted[candidates], np.asarray(valid_token_ids))]
    if count > len(eligible):
        raise ValueError(f'Requested {count} pseudo examples, only {len(eligible)} eligible')
    ordered = eligible[np.lexsort((eligible, -gap[eligible]))]
    return ordered[:count], predicted, gap, eligible


def _group_key(pair):
    a, b = map(int, pair)
    return min(a, b), max(a, b)


def _group_keys(indices, pairs):
    return sorted({_group_key(pairs[int(index)]) for index in indices})


def _members(indices, pairs, chosen_groups):
    chosen_groups = set(chosen_groups)
    return np.array([int(index) for index in indices
                     if _group_key(pairs[int(index)]) in chosen_groups], dtype=np.int64)


def _variant(indices, target_tokens, label_source):
    indices = np.asarray(indices, dtype=np.int64)
    targets = np.asarray(target_tokens, dtype=np.int64)
    if len(indices) != len(targets) or len(np.unique(indices)) != len(indices):
        raise ValueError('Variant needs unique indices with one target each')
    return dict(train_indices=indices.tolist(), target_tokens=targets.tolist(),
                label_source=label_source)


def prepare_plan(data, answer_logits, teacher_provenance=None, modulus=97,
                 group_seed=GROUP_SEED, pool_seed=POOL_SEED,
                 shuffle_seed=SHUFFLE_SEED):
    """Return serializable plan, preserving original global pair IDs.

    Canonical groups are sorted (min(a,b),max(a,b)) tuples before the seeded
    permutations. round() uses Python's nearest-even convention. R/S50 retain
    the original S iteration order. P uses confidence order and is appended to R.
    """
    pairs = np.asarray(data['pairs'], dtype=np.int64)
    n = len(pairs); vocabulary = list(data['vocab'])
    if pairs.shape != (modulus**2, 2) or np.any((pairs < 0) | (pairs >= modulus)):
        raise ValueError('Expected the complete modular pair grid')
    lookup = {tuple(pair): i for i, pair in enumerate(pairs)}
    if len(lookup) != n or len(vocabulary) != len(set(vocabulary)):
        raise ValueError('Duplicate input pair or vocabulary token')
    S = np.asarray(data['train'], dtype=np.int64)
    if S.ndim != 1 or len(np.unique(S)) != len(S) or np.any((S < 0) | (S >= n)):
        raise ValueError('Invalid original training indices')
    residues = np.array([vocabulary.index(str(value)) for value in range(modulus)])
    logits = teacher_answer_logits(answer_logits)
    if logits.shape != (n, len(vocabulary)):
        raise ValueError('Teacher logits do not match original pair/vocabulary dimensions')

    train = np.zeros(n, dtype=bool); train[S] = True
    reverse = np.array([lookup[(int(b), int(a))] for a, b in pairs])
    U = np.flatnonzero(~train & ~train[reverse])
    training_groups = _group_keys(S, pairs)
    training_order = np.random.RandomState(group_seed).permutation(len(training_groups))
    count75 = round(.75*len(training_groups)); count50 = round(.50*len(training_groups))
    selected75 = [training_groups[i] for i in training_order[:count75]]
    selected50 = [training_groups[i] for i in training_order[:count50]]
    R = _members(S, pairs, selected75); S50 = _members(S, pairs, selected50)
    H = _members(S, pairs, set(training_groups)-set(selected75))

    unseen_groups = _group_keys(U, pairs)
    unseen_order = np.random.RandomState(pool_seed).permutation(len(unseen_groups))
    candidate_group_count = round(.50*len(unseen_groups))
    candidate_groups = [unseen_groups[i] for i in unseen_order[:candidate_group_count]]
    evaluation_groups = [unseen_groups[i] for i in unseen_order[candidate_group_count:]]
    pool = _members(U, pairs, candidate_groups)
    clean_eval = _members(U, pairs, evaluation_groups)
    # Only split coordinates, vocabulary domain, and predicted confidence enter selection.
    P, predicted, gap, eligible = select_pseudo_indices(logits, pool, len(H), residues)

    # Arithmetic truth is deliberately evaluated only after the selection is frozen.
    true_tokens = residues[pairs.sum(1) % modulus]
    teacher_labels = predicted[P]
    shuffled_labels = np.random.RandomState(shuffle_seed).permutation(teacher_labels)
    mixed_indices = np.concatenate([R, P])
    variants = dict(
        subset75=_variant(R, true_tokens[R], 'Original true labels: retained75% training groups'),
        subset50=_variant(S50, true_tokens[S50], 'Original true labels: retained50% training groups'),
        teacher_mix=_variant(mixed_indices, np.concatenate([true_tokens[R], teacher_labels]),
                             'R original true labels; P frozen-teacher top1 predictions'),
        pseudo_only=_variant(P, teacher_labels, 'P frozen-teacher top1 predictions only'),
        shuffled_mix=_variant(mixed_indices, np.concatenate([true_tokens[R], shuffled_labels]),
                              'R original true labels; P permuted teacher labels, histogram preserved'),
        oracle_mix=_variant(mixed_indices, true_tokens[mixed_indices],
                            'R original true labels; P explicit oracle true labels: synthetic-task upper control'))
    partitions = {name: values.tolist() for name, values in
                  [('S', S), ('R', R), ('H', H), ('U', U), ('S50', S50),
                   ('pseudo_pool', pool), ('clean_eval', clean_eval), ('P', P)]}
    provenance = dict(teacher=teacher_provenance or {},
                      answer_logits_sha256=hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest(),
                      original_train_sha256=hashlib.sha256(S.tobytes()).hexdigest(),
                      input_pair_sha256=hashlib.sha256(pairs.tobytes()).hexdigest(),
                      vocabulary_sha256=canonical_hash(vocabulary),
                      algorithm_sha256=sha256_file(__file__),
                      seeds=dict(training_groups=group_seed, unseen_groups=pool_seed, shuffled_labels=shuffle_seed))
    selected_correct = int((teacher_labels == true_tokens[P]).sum())
    plan = dict(format_version=1, modulus=modulus, vocabulary=vocabulary,
                original_pair_count=n, original_train_count=len(S),
                model_seed=9278700, perturbation_seed=9278902,
                partitions=partitions, teacher_predictions=predicted.tolist(), variants=variants,
                group_counts=dict(original_training=len(training_groups), subset75=count75,
                                  subset50=count50, original_unseen=len(unseen_groups),
                                  pseudo_candidate=candidate_group_count,
                                  clean_evaluation=len(evaluation_groups)),
                partition_sizes={name: len(values) for name, values in partitions.items()},
                ordering='Canonical sorted unordered groups, seeded permutation; original S order for R/S50; confidence-ordered P appended to R',
                group_rounding='Python round: nearest integer, ties to even; pseudo candidate gets rounded half',
                selection=dict(count=len(P), eligible_pool_count=len(eligible),
                               nonresidue_pool_predictions=len(pool)-len(eligible),
                               confidence='top1-minus-top2 logit gap over full vocabulary',
                               domain='top1 token must belong to known residue vocabulary',
                               tie_break='ascending global pair index',
                               truth_margin_ce_correctness_used=False,
                               selected_predicted_gaps=gap[P].tolist()),
                selected_teacher_audit=dict(correct=selected_correct, count=len(P),
                                            accuracy=selected_correct/len(P) if len(P) else None,
                                            audit_after_selection=True,
                                            selected_true_tokens=true_tokens[P].tolist(),
                                            teacher_label_histogram=np.bincount(teacher_labels, minlength=len(vocabulary)).tolist()),
                provenance=provenance, provenance_sha256=canonical_hash(provenance),
                trained=False, new_model_forwards=0)
    plan['plan_sha256'] = canonical_hash(plan)
    return plan


def build_training_data(data, variant_dict):
    """Return ONLY selected rows, fresh targets, and contiguous local train IDs.

    Global indices remain in variant_dict. Evaluate true labels using a separate
    untouched original dataset; the optimizer receives no withheld rows/labels.
    """
    indices = np.asarray(variant_dict['train_indices'], dtype=np.int64)
    targets = np.asarray(variant_dict['target_tokens'], dtype=np.int64)
    if indices.ndim != 1 or targets.shape != indices.shape or not len(indices):
        raise ValueError('Need a nonempty one-dimensional training index/target list')
    if len(np.unique(indices)) != len(indices) or np.any((indices < 0) | (indices >= len(data['pairs']))):
        raise ValueError('Training global indices must be unique and in range')
    if np.any((targets < 0) | (targets >= len(data['vocab']))):
        raise ValueError('Target token outside original vocabulary')
    x = data['x'][torch.from_numpy(indices)].detach().clone()
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError('Expected the frozen six-token equation inputs')
    y = torch.full_like(x, -1)
    target = torch.as_tensor(targets, dtype=x.dtype, device=x.device)
    x[:, 5] = target; y[:, 4] = target
    y[:, 5] = list(data['vocab']).index('<|eos|>')
    return dict(x=x, y=y, train=np.arange(len(indices), dtype=np.int64),
                pairs=np.asarray(data['pairs'])[indices].copy())


def main(args):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if sha256_file(args.teacher_checkpoint) != TEACHER_SHA256:
        raise ValueError('Frozen teacher checkpoint does not match predeclared SHA256')
    manifest = json.loads(args.teacher_manifest.read_text())
    entry = next(row for row in manifest['checkpoints'] if row['label'] == 'dust3177')
    if entry['sha256'] != TEACHER_SHA256:
        raise ValueError('Saved-logit audit manifest names a different teacher')
    args.output.mkdir(parents=True, exist_ok=False); disk_guard(args.output)
    data = dataset()
    source_sha = sha256_file(args.teacher_npz)
    with np.load(args.teacher_npz) as archive:
        logits = teacher_answer_logits(archive).copy()
        np.testing.assert_array_equal(archive['a'], data['pairs'][:, 0])
        np.testing.assert_array_equal(archive['b'], data['pairs'][:, 1])
        np.testing.assert_array_equal(archive['predicted_token_id'], logits.argmax(1))
    provenance = dict(checkpoint_path=str(args.teacher_checkpoint), checkpoint_sha256=TEACHER_SHA256,
                      predictions_path=str(args.teacher_npz), predictions_sha256=source_sha,
                      prediction_manifest_path=str(args.teacher_manifest),
                      prediction_manifest_sha256=sha256_file(args.teacher_manifest),
                      protocol_sha256=sha256_file('PROTOCOL.md'))
    plan = prepare_plan(data, logits, provenance)
    for variant in plan['variants'].values():
        compact = build_training_data(data, variant)
        assert len(compact['x']) == len(variant['train_indices'])
    disk_guard(args.output); write_json(args.output/'plan.json', plan)
    write_json(args.output/'completion.json', dict(state='FOLLOWUP_DATA_READY', trained=False,
               new_model_forwards=0, partition_sizes=plan['partition_sizes'],
               group_counts=plan['group_counts'], teacher_selection_audit=plan['selected_teacher_audit'],
               provenance_sha256=plan['provenance_sha256'], plan_sha256=plan['plan_sha256']))
    if sha256_file(args.teacher_npz) != source_sha or sha256_file(args.teacher_checkpoint) != TEACHER_SHA256:
        raise AssertionError('Teacher source artifact changed during preparation')
    print(json.dumps(dict(output=str(args.output), partition_sizes=plan['partition_sizes'],
                         group_counts=plan['group_counts'],
                         selected_teacher_correct=plan['selected_teacher_audit']['correct'],
                         selected_teacher_count=plan['selected_teacher_audit']['count'],
                         selected_teacher_accuracy=plan['selected_teacher_audit']['accuracy'],
                         provenance_sha256=plan['provenance_sha256'], plan_sha256=plan['plan_sha256']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/followup-data-20261011'))
    parser.add_argument('--teacher-npz', type=Path, default=Path('runs/example-audit-20261011/dust3177_pairs.npz'))
    parser.add_argument('--teacher-manifest', type=Path, default=Path('runs/example-audit-20261011/manifest.json'))
    parser.add_argument('--teacher-checkpoint', type=Path, default=Path('runs/component-final-states-20261011/dust.pt'))
    main(parser.parse_args())
