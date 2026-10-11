"""Bounded pure-Dust continuations with frozen prediction-selected labels.

Each process independently restores the same i1-n1@3177 checkpoint. Selection
and its commit record are read-only inputs; evaluation never changes labels.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .followup_data import canonical_hash
from .model import token_losses
from .seed_continue import restore, activate_cpu, adam_sha, tensor_sha
from .seed_train import masks, score
from .task import dataset, hashes
from .train import checkpoint, disk_guard, source_hashes, update, write_json

ARMS = ('baseline', 'confidence', 'stability')
INPUT_FILES = ('checkpoint.pt', 'parent_config.json', 'selection_manifest.json',
               'selection_commit.json', 'followup_plan.json')
START_STEP = 3177


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_keys(indices, pairs):
    return {tuple(sorted(map(int, pairs[int(i)]))) for i in indices}


def select_entry(manifest, arm):
    rule = {'confidence': 'confidence', 'stability': 'replica'}[arm]
    found = [row for row in manifest['selections'] if row['teacher'] == 'i1-n1'
             and row['rule'] == rule and row['requested_count'] == 256]
    if len(found) != 1 or found[0]['status'] != 'SELECTED':
        raise ValueError('Need exactly one successful frozen 256-example selection: '+arm)
    return found[0]


def validate_manifest(manifest, commit, plan, source_config, checkpoint_sha, manifest_sha, data):
    if manifest.get('selection_step') != START_STEP or manifest.get('future_inputs_used') is not False \
       or manifest.get('target_margin_ce_correctness_used') is not False:
        raise ValueError('Selection must exclude future predictions and evaluation labels')
    if commit.get('selection_manifest_sha256') != manifest_sha \
       or commit.get('canonical_manifest_sha256') != canonical_hash(manifest) \
       or commit.get('future_predictions_opened') is not False or commit.get('labels_opened') is not False:
        raise ValueError('Selection freeze/hash/chronology receipt is invalid')
    committed = datetime.fromisoformat(commit['committed_utc'])
    if committed.tzinfo is None or committed.timestamp() > time.time()+1:
        raise ValueError('Selection commit must have a past timezone-aware timestamp')
    meta = manifest['replica_provenance']['i1-n1']
    expected = dict(run='i1-n1', step=START_STEP, method='dust', model_seed=9278701,
                    noise_seed=9278902, inherited_backprop_steps=0, current_backprop_steps=0,
                    checkpoint_sha256=checkpoint_sha)
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen selector is not the specified original i1-n1@3177 teacher')
    if source_config['model_seed'] != 9278701 or source_config['noise_seed'] != 9278902:
        raise ValueError('Parent initialization/noise identity changed')
    split_sha = hashlib.sha256(data['order'].tobytes()).hexdigest()
    if source_config['split_sha256'] != split_sha or meta['split_sha256'] != split_sha \
       or source_config['vocabulary'] != data['vocab'] or meta['vocabulary'] != data['vocab']:
        raise ValueError('Vocabulary or original split changed')
    if source_config['reference_hashes'] != hashes():
        raise ValueError('Released reference source changed')
    canonical = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    if canonical_hash(canonical) != plan['plan_sha256']:
        raise ValueError('Original follow-up partition plan hash mismatch')
    if canonical_hash(plan['provenance']) != plan['provenance_sha256']:
        raise ValueError('Original follow-up provenance hash mismatch')
    if plan['vocabulary'] != data['vocab'] or plan['partitions']['S'] != data['train'].tolist():
        raise ValueError('Original full training set/order changed')
    pool = np.asarray(plan['partitions']['pseudo_pool'], dtype=np.int64)
    clean = np.asarray(plan['partitions']['clean_eval'], dtype=np.int64)
    unseen = set(np.flatnonzero(masks(data)[1]).tolist())
    for ids in (pool, clean):
        if len(set(ids.tolist())) != len(ids) or not len(ids):
            raise ValueError('Partition indices must be unique and nonempty')
    if set(pool) & set(clean) or set(pool) | set(clean) != unseen:
        raise ValueError('Pseudo pool and clean evaluation must partition the original unseen pairs')
    if group_keys(pool, data['pairs']) & group_keys(clean, data['pairs']):
        raise ValueError('Opposite orientations cross pseudo-pool/clean-evaluation boundary')
    valid_tokens = {data['vocab'].index(str(i)) for i in range(97)}
    for arm in ('confidence', 'stability'):
        entry = select_entry(manifest, arm)
        ids = entry['selected_indices']; tokens = entry['pseudo_tokens']
        if len(ids) != 256 or len(tokens) != 256 or len(set(ids)) != 256:
            raise ValueError('Frozen augmentation must contain exactly 256 unique indices')
        if not set(ids).issubset(set(pool)) or not set(tokens).issubset(valid_tokens):
            raise ValueError('Pseudo examples/labels must be valid pool indices/residue tokens')
        if len(group_keys(ids, data['pairs'])) != 256:
            raise ValueError('Pseudo selection contains both orientations of one group')
    return committed.timestamp()


def load_inputs(root):
    """Verify immutable package before a trainer creates any output."""
    root = Path(root).resolve()
    package_path = root/'inputs.json'; package = json.loads(package_path.read_text())
    if package.get('format_version') != 1 or set(package['files']) != set(INPUT_FILES):
        raise ValueError('Unexpected input-package version or file inventory')
    for name in INPUT_FILES:
        if sha(root/name) != package['files'][name]:
            raise ValueError('Input-package hash mismatch: '+name)
    config = json.loads((root/'parent_config.json').read_text())
    manifest = json.loads((root/'selection_manifest.json').read_text())
    commit = json.loads((root/'selection_commit.json').read_text())
    plan = json.loads((root/'followup_plan.json').read_text())
    data = dataset()
    committed_unix = validate_manifest(manifest, commit, plan, config, package['files']['checkpoint.pt'],
                                      package['files']['selection_manifest.json'], data)
    created = float(package['created_unix'])
    if not committed_unix <= created <= time.time()+1:
        raise ValueError('Input package must be created after selection was frozen and before launch')
    if package['files']['followup_plan.json'] not in manifest['input_sha256'].values():
        raise ValueError('Selection did not use the supplied original partition plan')
    saved = torch.load(root/'checkpoint.pt', map_location='cpu', weights_only=False)
    if int(saved['step']) != START_STEP or saved.get('dust_rng') is None:
        raise ValueError('A complete original3177 checkpoint and perturbation RNG are required')
    return dict(root=root, package=package, package_sha256=sha(package_path), config=config,
                manifest=manifest, commit=commit, plan=plan, data=data, saved=saved)


def build_data(data, manifest, plan, arm):
    if arm not in ARMS: raise ValueError('Unknown continuation arm')
    entry = None if arm == 'baseline' else select_entry(manifest, arm)
    pseudo = np.asarray([] if entry is None else entry['selected_indices'], dtype=np.int64)
    tokens = torch.tensor([] if entry is None else entry['pseudo_tokens'], dtype=torch.long)
    result = dict(data, x=data['x'].clone(), y=data['y'].clone(),
                  train=np.concatenate([data['train'].copy(), pseudo]))
    if len(pseudo):
        result['x'][pseudo, 5] = tokens
        result['y'][pseudo, 4] = tokens
    original = data['train']
    if not np.array_equal(result['train'][:len(original)], original) \
       or not torch.equal(result['x'][original], data['x'][original]) \
       or not torch.equal(result['y'][original], data['y'][original]):
        raise AssertionError('Original true-training equations changed')
    if group_keys(result['train'], data['pairs']) & group_keys(plan['partitions']['clean_eval'], data['pairs']):
        raise ValueError('Training overlaps clean evaluation, including reversed operands')
    if len(pseudo) and (not torch.equal(result['x'][pseudo, 5], result['y'][pseudo, 4])
                        or not torch.equal(result['y'][pseudo, 5], data['y'][pseudo, 5])):
        raise AssertionError('Pseudo teacher forcing/answer/EOS targets disagree')
    return result, pseudo


def initialize_parent(saved, config):
    model, opt, dust = restore(saved, config)
    dust = activate_cpu(model, dust)
    group = opt.param_groups[0]
    if group['lr'] != .001 or tuple(group['betas']) != (.9, .98) \
       or group['eps'] != 1e-8 or group['weight_decay'] != 1.:
        raise ValueError('Matched optimizer recipe changed')
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError('Dust continuation must disable autograd')
    return model, opt, dust


def validate_schedule(steps, seconds, threads, deadline_unix=None):
    if not 1 <= steps <= 200 or not 90 <= seconds <= 3540 or not 1 <= threads <= 8:
        raise ValueError('Continuation exceeds bounded steps/runtime/CPU settings')
    now = time.time(); deadline = now+seconds if deadline_unix is None else float(deadline_unix)
    if not now+45 < deadline <= now+seconds+1:
        raise ValueError('Continuation deadline is exhausted or exceeds the internal budget')
    return deadline


def create_output(root, inputs_root):
    root = Path(root).resolve()
    if root.exists() or root.is_relative_to(Path(inputs_root).resolve()):
        raise FileExistsError('Refusing existing output or output inside immutable inputs')
    root.parent.mkdir(parents=True, exist_ok=True)
    disk_guard(root.parent)
    root.mkdir(exist_ok=False)
    return root


@torch.no_grad()
def evaluate_arm(model, truth, training, plan, pseudo, initial=None, previous=None):
    measured, payload = score(model, truth)
    clean = np.asarray(plan['partitions']['clean_eval'], dtype=np.int64)
    prediction = payload['predicted_token'][clean].copy()
    correct = payload['correct'][clean].copy()
    current = dict(indices=clean, predicted_token=prediction, correct=correct,
                   answer_ce=payload['answer_ce'][clean].copy(), margin=payload['margin'][clean].copy())
    if initial is None: initial = current
    if previous is None: previous = current
    start_correct = initial['correct']; prev_correct = previous['correct']
    row = dict(original_train_accuracy=measured['train_accuracy'],
               original_train_rhs_loss=measured['train_loss'],
               original_train_answer_ce=measured['train_product_loss'],
               original_all_held_accuracy=measured['heldout_accuracy'],
               original_all_held_answer_ce=measured['heldout_product_loss'],
               ordinary_all_held_contaminated=bool(len(pseudo)),
               clean_eval_count=len(clean), clean_eval_accuracy=float(correct.mean()),
               clean_eval_answer_ce=float(current['answer_ce'].mean()),
               clean_eval_prediction_retention=float((prediction == initial['predicted_token']).mean()),
               clean_eval_initial_correct=int(start_correct.sum()),
               clean_eval_retained_correct=int((correct & start_correct).sum()),
               clean_eval_lost_initial_correct=int((~correct & start_correct).sum()),
               clean_eval_recovered_initial_wrong=int((correct & ~start_correct).sum()),
               clean_eval_recovered_since_previous=int((correct & ~prev_correct).sum()),
               clean_eval_lost_since_previous=int((~correct & prev_correct).sum()),
               weight_norm=measured['weight_norm'], pseudo_count=len(pseudo),
               pseudo_target_accuracy=None, pseudo_answer_ce=None, pseudo_rhs_loss=None,
               evaluation_forward_examples=measured['evaluation_forward_examples'])
    if len(pseudo):
        y = training['y'][pseudo]; z = model(training['x'][pseudo]); loss = token_losses(z, y)[:, 4:6]
        row.update(pseudo_target_accuracy=float((z[:, 4].argmax(-1) == y[:, 4]).float().mean()),
                   pseudo_answer_ce=float(loss[:, 0].mean()), pseudo_rhs_loss=float(loss.mean()),
                   evaluation_forward_examples=row['evaluation_forward_examples']+len(pseudo))
    current.update(initial_predicted_token=initial['predicted_token'], initial_correct=start_correct,
                   retained_initial_correct=correct & start_correct,
                   lost_initial_correct=~correct & start_correct,
                   recovered_initial_wrong=correct & ~start_correct,
                   recovered_since_previous=correct & ~prev_correct)
    return row, current


def trajectory_summary(rows):
    later = [r for r in rows if 100 <= r['step'] <= 200]
    return dict(final_step=rows[-1]['step'], final_clean_eval_accuracy=rows[-1]['clean_eval_accuracy'],
        minimum_clean_eval_accuracy=min(r['clean_eval_accuracy'] for r in rows),
        mean_clean_eval_accuracy_updates100_to200=float(np.mean([r['clean_eval_accuracy'] for r in later])) if later else None,
        mean_clean_eval_answer_ce_updates100_to200=float(np.mean([r['clean_eval_answer_ce'] for r in later])) if later else None,
        late_evaluation_steps=[r['step'] for r in later], late_window_complete=all(s in [r['step'] for r in later] for s in range(100, 201, 10)),
        minimum_scope='all recorded evaluations including update0; no best-checkpoint selection')


def run(a):
    deadline = validate_schedule(a.steps, a.seconds, a.threads, a.deadline_unix)
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    inputs = load_inputs(a.inputs); root = create_output(a.output, a.inputs)
    if time.time() >= deadline-45: raise ValueError('Input preflight consumed the remaining runtime')
    truth = inputs['data']; saved = inputs['saved']; source_config = inputs['config']
    training, pseudo = build_data(truth, inputs['manifest'], inputs['plan'], a.arm)
    model, opt, dust = initialize_parent(saved, source_config)
    inherited_counts = dict(dust.counts); counters = dict(saved['counters']); inherited_counters = dict(counters)
    counters.setdefault('eval_forward_examples', 0)
    config = dict(arm=a.arm, method='dust', source_step=START_STEP, inputs_root=str(inputs['root']),
        input_package_sha256=inputs['package_sha256'], input_sha256=inputs['package']['files'],
        selection_commit=inputs['commit'], selection_rule=None if a.arm == 'baseline' else ('confidence' if a.arm == 'confidence' else 'replica'),
        model_seed=source_config['model_seed'], noise_seed=source_config['noise_seed'],
        initial_parameter_sha256=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest(),
        initial_adam_sha256=adam_sha(opt), initial_rng_sha256=tensor_sha(dust.rng.get_state()),
        inherited_counters=inherited_counters, inherited_dust_counts=inherited_counts,
        inherited_backprop_steps=0, zeroth_order_only=True, execution='CpuDust',
        original_training_examples=len(truth['train']), pseudo_examples=len(pseudo), training_examples=len(training['train']),
        pseudo_loss_weight=len(pseudo)/len(training['train']), original_loss_weight=len(truth['train'])/len(training['train']),
        original_training_order_preserved=True, pseudo_teacher_forced_answer_updated=True,
        train_indices=training['train'].tolist(), pseudo_indices=pseudo.tolist(),
        clean_eval_indices=inputs['plan']['partitions']['clean_eval'],
        primary_metric='reserved clean_eval; both orientations excluded from training',
        ordinary_all_held_contaminated=bool(len(pseudo)),
        noise_pairing='Identical starting RNG; augmented batch shapes change subsequent draw trajectories',
        selection_teacher_cost=inputs['manifest'].get('teacher_cost'),
        steps=a.steps, internal_seconds=a.seconds, internal_deadline_unix=deadline, threads=a.threads, chunk=256,
        eval_every=10, special_evaluation_steps=[0, 23], start_unix=time.time(),
        optimizer={k:opt.param_groups[0][k] for k in ('lr','betas','eps','weight_decay')},
        dust_settings=dust.metadata()['settings'], source_hashes=source_hashes(),
        protocol_sha256=sha('PROTOCOL.md'), checkpoint_sha256=inputs['package']['files']['checkpoint.pt'])
    write_json(root/'config.json', config)
    stop = [False]; old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    start = time.monotonic(); step = 0; rows = []; initial = None; previous = None; update_seconds = 0.

    def record():
        nonlocal initial, previous
        row, payload = evaluate_arm(model, truth, training, inputs['plan'], pseudo, initial, previous)
        if initial is None: initial = {k:v.copy() for k,v in payload.items()}
        previous = {k:v.copy() for k,v in payload.items()}
        counters['eval_forward_examples'] += row.pop('evaluation_forward_examples')
        row.update(step=step, model_step=START_STEP+step, arm=a.arm, wall_seconds=time.monotonic()-start,
                   update_seconds=update_seconds, counters=dict(counters),
                   incremental_counters={k:v-inherited_counters.get(k,0) for k,v in counters.items()},
                   dust_counts=dict(dust.counts), incremental_dust_counts={k:v-inherited_counts.get(k,0) for k,v in dust.counts.items()})
        disk_guard(root)
        pred_dir = root/'clean_predictions'; pred_dir.mkdir(exist_ok=True)
        target = pred_dir/f'update-{step:05d}.npz'
        if target.exists(): raise FileExistsError(target)
        temp = target.with_suffix('.tmp.npz'); np.savez_compressed(temp, step=np.array(step), **payload); temp.replace(target)
        row.update(checkpoint(root, model, opt, dust, START_STEP+step, counters))
        rows.append(row); write_json(root/'metrics.json', rows)
        write_json(root/'trajectory_summary.json', trajectory_summary(rows))
        write_json(root/'status.json', dict(state='TRAINING', **row)); print(json.dumps(row), flush=True)

    try:
        record()
        while step < a.steps and not stop[0] and time.time() < deadline-45:
            t = time.monotonic()
            loss = update(model, opt, dust, training, START_STEP+step, 256)
            update_seconds += time.monotonic()-t; step += 1
            counters['supervised_token_presentations'] += 2*len(training['train'])
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite parameter after Dust update')
            if step % 10 == 0 or step == 23: record()
            elif step % 5 == 0:
                write_json(root/'progress.json', dict(step=step, loss_before_update=loss,
                           seconds_per_update=update_seconds/step, wall_seconds=time.monotonic()-start))
        if rows[-1]['step'] != step: record()
        if sha(inputs['root']/'inputs.json') != inputs['package_sha256'] or any(
            sha(inputs['root']/name) != digest for name,digest in inputs['package']['files'].items()):
            raise ValueError('Immutable input package changed during continuation')
        write_json(root/'completion.json', dict(state='HORIZON_COMPLETE' if step == a.steps else 'BUDGET_STOP',
            step=step, model_step=START_STEP+step, wall_seconds=time.monotonic()-start,
            budget_stop_is_failure=False, signal_received=stop[0], zeroth_order_only=True,
            trajectory_summary=trajectory_summary(rows), original_inputs_unchanged=True))
    except Exception as error:
        write_json(root/'failure.json', dict(state='CONTINUATION_FAILURE', step=step, reason=repr(error)))
        raise
    finally:
        for sig, handler in old_handlers.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--verify-inputs', action='store_true')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seconds', type=int, default=3540)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--deadline-unix', type=float)
    args = parser.parse_args()
    if args.verify_inputs:
        torch.set_num_threads(1)
        verified = load_inputs(args.inputs)
        print(json.dumps(dict(state='INPUTS_VERIFIED', package_sha256=verified['package_sha256'], source_step=START_STEP)))
    else:
        if args.arm is None or args.output is None: parser.error('--arm and --output are required for a continuation')
        run(args)
