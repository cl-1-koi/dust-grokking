import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.followup_data import canonical_hash
from src.seed_continue import adam_sha, equal_nested, tensor_sha
from src.seed_train import initialize, masks
from src.stability_train import (INPUT_FILES, build_data, create_output, evaluate_arm,
    group_keys, initialize_parent, load_inputs, sha, trajectory_summary,
    validate_manifest, validate_schedule)
from src.task import dataset, hashes
from src.train import checkpoint, update


class StabilityTrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data = dataset()
        cls.model, cls.opt, cls.dust = initialize('dust', 9278701, 9278902)
        # Populate complete Adam state without using backpropagation.
        for p in cls.model.parameters(): p.grad = torch.zeros_like(p)
        cls.opt.step()
        for state in cls.opt.state.values(): state['step'].fill_(3177)
        cls.saved = dict(model=copy.deepcopy(cls.model.state_dict()), optimizer=copy.deepcopy(cls.opt.state_dict()),
                         step=3177, dust_rng=cls.dust.rng.get_state().clone(), dust_counts=dict(cls.dust.counts),
                         counters=dict(bp_forward_examples=0, eval_forward_examples=0,
                                       supervised_token_presentations=3177*2823*2))
        cls.config = dict(method='dust', inherited_backprop_steps=0, zeroth_order_only=True,
            model_seed=9278701, noise_seed=9278902, dust_settings=asdict(cls.dust.s),
            split_sha256=hashlib.sha256(cls.data['order'].tobytes()).hexdigest(),
            vocabulary=cls.data['vocab'], reference_hashes=hashes())
        unseen = np.flatnonzero(masks(cls.data)[1])
        ordered_groups = sorted(group_keys(unseen, cls.data['pairs']))
        pool_groups = set(ordered_groups[:len(ordered_groups)//2])
        pool = [int(i) for i in unseen if tuple(sorted(map(int, cls.data['pairs'][i]))) in pool_groups]
        clean = [int(i) for i in unseen if i not in set(pool)]
        cls.plan = dict(vocabulary=cls.data['vocab'], partitions=dict(S=cls.data['train'].tolist(),
            U=unseen.tolist(), pseudo_pool=pool, clean_eval=clean), provenance={'fixture': 'input-only partition'})
        cls.plan['provenance_sha256'] = canonical_hash(cls.plan['provenance'])
        cls.plan['plan_sha256'] = canonical_hash(cls.plan)
        unique = []; seen = set()
        for i in pool:
            key = tuple(sorted(map(int, cls.data['pairs'][i])))
            if key not in seen: unique.append(i); seen.add(key)
        cls.rows = []
        for rule, ids, value in [('confidence', unique[:256], '3'), ('replica', unique[-256:], '8')]:
            cls.rows.append(dict(teacher='i1-n1', rule=rule, requested_count=256, status='SELECTED',
                                 selected_indices=ids, pseudo_tokens=[cls.data['vocab'].index(value)]*256))

    def fixture(self, root):
        root.mkdir()
        torch.save(self.saved, root/'checkpoint.pt')
        (root/'parent_config.json').write_text(json.dumps(self.config))
        (root/'followup_plan.json').write_text(json.dumps(self.plan))
        meta = dict(run='i1-n1', step=3177, method='dust', model_seed=9278701, noise_seed=9278902,
                    inherited_backprop_steps=0, current_backprop_steps=0,
                    checkpoint_sha256=sha(root/'checkpoint.pt'), split_sha256=self.config['split_sha256'],
                    vocabulary=self.data['vocab'])
        manifest = dict(selection_step=3177, future_inputs_used=False, target_margin_ce_correctness_used=False,
                        replica_provenance={'i1-n1': meta}, selections=copy.deepcopy(self.rows),
                        input_sha256={'original_plan':sha(root/'followup_plan.json')})
        (root/'selection_manifest.json').write_text(json.dumps(manifest))
        commit = dict(selection_manifest_sha256=sha(root/'selection_manifest.json'),
                      canonical_manifest_sha256=canonical_hash(manifest),
                      committed_utc=datetime.now(timezone.utc).isoformat(),
                      future_predictions_opened=False, labels_opened=False)
        (root/'selection_commit.json').write_text(json.dumps(commit))
        package = dict(format_version=1, created_unix=time.time(), files={n:sha(root/n) for n in INPUT_FILES})
        (root/'inputs.json').write_text(json.dumps(package))
        return load_inputs(root)

    def test_independent_parent_restores_and_actual_zeroth_order_update(self):
        parents = [initialize_parent(self.saved, self.config) for _ in range(3)]
        expected_adam = adam_sha(parents[0][1]); expected_rng = tensor_sha(self.saved['dust_rng'])
        for model, opt, dust in parents:
            self.assertTrue(equal_nested(model.state_dict(), self.saved['model']))
            self.assertTrue(equal_nested(opt.state_dict(), self.saved['optimizer']))
            self.assertEqual(adam_sha(opt), expected_adam)
            self.assertEqual(tensor_sha(dust.rng.get_state()), expected_rng)
            self.assertEqual(type(dust).__name__, 'CpuDust')
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
        model, opt, dust = parents[0]
        tiny = dict(self.data, train=self.data['train'][:4])
        with patch.object(torch.Tensor, 'backward', side_effect=AssertionError('Unexpected backward')):
            with patch('torch.autograd.grad', side_effect=AssertionError('Unexpected analytic gradient')):
                loss = update(model, opt, dust, tiny, 3177, 4)
        self.assertTrue(np.isfinite(loss))
        self.assertNotEqual(adam_sha(opt), expected_adam)
        self.assertNotEqual(tensor_sha(dust.rng.get_state()), expected_rng)
        self.assertTrue(any(not torch.equal(p, self.saved['model'][n]) for n,p in model.state_dict().items()))
        for untouched, untouched_opt, untouched_dust in parents[1:]:
            self.assertTrue(equal_nested(untouched.state_dict(), self.saved['model']))
            self.assertEqual(adam_sha(untouched_opt), expected_adam)
            self.assertEqual(tensor_sha(untouched_dust.rng.get_state()), expected_rng)
        self.assertEqual(int(next(iter(self.saved['optimizer']['state'].values()))['step']), 3177)

    def test_original_training_and_pseudo_teacher_forcing_are_preserved(self):
        manifest = dict(selections=self.rows)
        x, y = self.data['x'].clone(), self.data['y'].clone()
        for arm in ('baseline', 'confidence', 'stability'):
            train, pseudo = build_data(self.data, manifest, self.plan, arm)
            count = 0 if arm == 'baseline' else 256
            self.assertEqual(len(train['train']), 2823+count)
            np.testing.assert_array_equal(train['train'][:2823], self.data['train'])
            torch.testing.assert_close(train['x'][self.data['train']], x[self.data['train']], atol=0, rtol=0)
            torch.testing.assert_close(train['y'][self.data['train']], y[self.data['train']], atol=0, rtol=0)
            self.assertFalse(group_keys(train['train'], self.data['pairs']) & group_keys(self.plan['partitions']['clean_eval'], self.data['pairs']))
            if count:
                expected = torch.tensor(self.rows[0 if arm=='confidence' else 1]['pseudo_tokens'])
                self.assertTrue(torch.equal(train['x'][pseudo,5], expected))
                self.assertTrue(torch.equal(train['y'][pseudo,4], expected))
                self.assertTrue(torch.equal(train['y'][pseudo,5], y[pseudo,5]))
                self.assertEqual(int((train['y'][train['train']]!=-1).sum()), 2*(2823+256))
        torch.testing.assert_close(self.data['x'], x, atol=0, rtol=0)
        torch.testing.assert_close(self.data['y'], y, atol=0, rtol=0)

    def test_clean_evaluation_rejects_both_orientations(self):
        clean_id = next(i for i in self.plan['partitions']['clean_eval'] if self.data['pairs'][i,0] != self.data['pairs'][i,1])
        a,b = map(int,self.data['pairs'][clean_id]); reversed_id=b*97+a
        for injected in (clean_id,reversed_id):
            corrupt = copy.deepcopy(self.rows)
            corrupt[0]['selected_indices'][0] = injected
            with self.assertRaisesRegex(ValueError,'clean evaluation'):
                build_data(self.data, {'selections':corrupt}, self.plan, 'confidence')

    def test_manifest_hash_teacher_and_temporal_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'inputs'; inputs = self.fixture(root)
            self.assertEqual(inputs['saved']['step'],3177)
            self.assertEqual(inputs['package_sha256'],sha(root/'inputs.json'))
            # A validly rehashed selection is still rejected if its freeze receipt
            # says future outcomes or labels had already been opened.
            for key in ('labels_opened','future_predictions_opened'):
                commit = dict(inputs['commit']); commit[key] = True
                with self.assertRaisesRegex(ValueError,'receipt'):
                    validate_manifest(inputs['manifest'],commit,self.plan,self.config,
                        inputs['package']['files']['checkpoint.pt'], inputs['package']['files']['selection_manifest.json'],self.data)
            changed = copy.deepcopy(inputs['manifest'])
            changed['replica_provenance']['i1-n1']['checkpoint_sha256'] = '0'*64
            commit = dict(inputs['commit'],canonical_manifest_sha256=canonical_hash(changed))
            with self.assertRaisesRegex(ValueError,'teacher'):
                validate_manifest(changed,commit,self.plan,self.config,inputs['package']['files']['checkpoint.pt'],
                                  inputs['package']['files']['selection_manifest.json'],self.data)
            package = inputs['package']; package['created_unix'] = 0
            (root/'inputs.json').write_text(json.dumps(package))
            with self.assertRaisesRegex(ValueError,'created after'):
                load_inputs(root)
            package['created_unix']=time.time(); (root/'inputs.json').write_text(json.dumps(package))
            with (root/'parent_config.json').open('a') as stream: stream.write(' ')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                load_inputs(root)

    def test_checkpoint_keeps_model_adam_rng_and_refuses_low_disk(self):
        model,opt,dust = initialize_parent(self.saved,self.config)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoint(root,model,opt,dust,3177,self.saved['counters'])
            persisted=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
            self.assertTrue(equal_nested(persisted['model'],self.saved['model']))
            self.assertTrue(equal_nested(persisted['optimizer'],self.saved['optimizer']))
            self.assertTrue(torch.equal(persisted['dust_rng'],self.saved['dust_rng']))
            before=sha(root/'latest.pt')
            with patch('src.train.shutil.disk_usage') as usage:
                usage.return_value.free=5*1024**3-1
                with self.assertRaises(OSError):checkpoint(root,model,opt,dust,3178,self.saved['counters'])
            self.assertEqual(sha(root/'latest.pt'),before)
            self.assertFalse((root/'latest.tmp.pt').exists())

    def test_output_and_deadline_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); inputs=base/'inputs'; inputs.mkdir(); output=base/'out'
            with self.assertRaises(FileExistsError):create_output(inputs/'child',inputs)
            create_output(output,inputs)
            with self.assertRaises(FileExistsError):create_output(output,inputs)
            with patch('src.train.shutil.disk_usage') as usage:
                usage.return_value.free=0
                with self.assertRaises(OSError):create_output(base/'no-space',inputs)
            self.assertFalse((base/'no-space').exists())
        with self.assertRaises(ValueError):validate_schedule(201,3540,8)
        with self.assertRaises(ValueError):validate_schedule(200,3541,8)
        with self.assertRaises(ValueError):validate_schedule(200,3540,8,time.time()-1)
        with self.assertRaises(ValueError):validate_schedule(200,3540,8,time.time()+5000)

    def test_clean_pair_retention_and_mandatory_trajectory_summary(self):
        clean=np.array(self.plan['partitions']['clean_eval'])
        predicted=np.zeros(9409,dtype=np.int64); correct=np.zeros(9409,dtype=bool)
        correct[clean[:2]]=True
        payload=dict(predicted_token=predicted,correct=correct,answer_ce=np.ones(9409),margin=np.zeros(9409))
        measured=dict(train_accuracy=.7,train_loss=1.,train_product_loss=1.2,heldout_accuracy=.4,
                      heldout_product_loss=2.,weight_norm=3.,evaluation_forward_examples=9409)
        with patch('src.stability_train.score',return_value=(measured,payload)):
            first,start=evaluate_arm(None,self.data,self.data,self.plan,np.array([],dtype=np.int64))
        changed=copy.deepcopy(payload);changed['correct'][clean[0]]=False;changed['correct'][clean[2]]=True
        changed['predicted_token'][clean[0]]=1
        with patch('src.stability_train.score',return_value=(measured,changed)):
            after,_=evaluate_arm(None,self.data,self.data,self.plan,np.array([],dtype=np.int64),start,start)
        self.assertEqual(after['clean_eval_retained_correct'],1)
        self.assertEqual(after['clean_eval_lost_initial_correct'],1)
        self.assertEqual(after['clean_eval_recovered_initial_wrong'],1)
        self.assertEqual(after['clean_eval_recovered_since_previous'],1)
        rows=[dict(step=s,clean_eval_accuracy=.5+s/1000,clean_eval_answer_ce=1.) for s in [0,10,20,23]+list(range(100,201,10))]
        summary=trajectory_summary(rows)
        self.assertEqual(summary['minimum_clean_eval_accuracy'],.5)
        self.assertAlmostEqual(summary['mean_clean_eval_accuracy_updates100_to200'],.65)
        self.assertEqual(summary['final_step'],200)
        self.assertTrue(summary['late_window_complete'])
        self.assertIsNone(trajectory_summary(rows[:4])['mean_clean_eval_accuracy_updates100_to200'])

    def test_queue_refuses_failed_predecessor_without_starting_an_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); self.fixture(root/'inputs'); predecessor=root/'predecessor';predecessor.mkdir()
            (root/'predecessor.exit').write_text('1\n')
            (predecessor/'completion.json').write_text(json.dumps(dict(state='HORIZON_COMPLETE')))
            env=dict(os.environ,STABILITY_PYTHON=sys.executable)
            run=subprocess.run(['bash','scripts/queue_stability_followup.sh',str(predecessor),str(root/'inputs'),str(root/'outputs')],
                               text=True,capture_output=True,env=env,timeout=45)
            self.assertEqual(run.returncode,76,run.stdout+run.stderr)
            self.assertIn('did not exit cleanly',run.stderr)
            self.assertFalse((root/'outputs'/'baseline').exists())


if __name__=='__main__':unittest.main()
