"""Continuation guards, using real serialized parent state and frozen equations.

The parallel estimator has its own parity tests. Runner tests substitute a
cheap optimizer update, while retaining actual restoration/checkpoint/history
I/O, to audit continuation boundaries without a full scientific training run.
"""
import copy
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src import followup_scale as scale
from src.followup_data import build_training_data, canonical_hash, prepare_plan
from src.seed_continue import adam_sha, equal_nested, restore, tensor_sha
from src.seed_train import initialize
from src.task import dataset
from src.train import source_hashes


class FollowupScaleChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.now = time.time()
        cls.truth = dataset()
        logits = np.zeros((9409, len(cls.truth['vocab'])), dtype=np.float32)
        logits[:, cls.truth['vocab'].index('0')] = 2
        cls.plan = prepare_plan(cls.truth, logits, {'fixture': 'constant teacher, no training'})
        cls.model, cls.opt, cls.dust = initialize('dust', 9278700, 9278902)
        # Populate complete, nonempty Adam state without any backward call.
        for p in cls.model.parameters():
            p.grad = torch.full_like(p, .001)
        cls.opt.step()
        for value in cls.opt.state.values():
            value['step'].fill_(3199)
        torch.randn(17, generator=cls.dust.rng)
        cls.dust.counts = {key: 17+i for i, key in enumerate(cls.dust.counts)}
        cls.saved = dict(model=copy.deepcopy(cls.model.state_dict()),
            optimizer=copy.deepcopy(cls.opt.state_dict()), step=3199,
            dust_rng=cls.dust.rng.get_state().clone(), dust_counts=dict(cls.dust.counts),
            counters=dict(bp_forward_examples=0, eval_forward_examples=18818,
                          supervised_token_presentations=3199*2823*2))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base/'parent'
        self.root.mkdir()
        self.args = SimpleNamespace(inputs=self.root, output=self.base/'continued')
        self.make_package()

    def write(self, name, value):
        (self.root/name).write_text(json.dumps(value))

    def read(self, name):
        return json.loads((self.root/name).read_text())

    def seal(self, **extra):
        files = {str(p.relative_to(self.root)): scale.sha(p)
                 for p in sorted(self.root.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
        self.write('manifest.json', dict(format_version=1, files=files, **extra))

    def make_package(self, variant='teacher_mix'):
        self.training = build_training_data(self.truth, self.plan['variants'][variant])
        self.write('plan.json', self.plan)
        config = dict(method='dust', variant=variant, model_seed=9278700, noise_seed=9278902,
            inherited_backprop_steps=0, inherited_teacher_weights=False, zeroth_order_only=True,
            dust_settings=asdict(self.dust.s), source_hashes=source_hashes(),
            plan_file_sha256=scale.sha(self.root/'plan.json'), canonical_plan_sha256=self.plan['plan_sha256'],
            label_source=self.plan['variants'][variant]['label_source'], training_examples=len(self.training['train']),
            training_tensor_sha256=hashlib.sha256(self.training['x'].numpy().tobytes()+self.training['y'].numpy().tobytes()).hexdigest(),
            steps=3200, seconds=71400, start_unix=self.now-3600, threads=8, chunk=256,
            optimizer={k:self.opt.param_groups[0][k] for k in ('lr','betas','eps','weight_decay')})
        self.write('config.json', config)
        torch.save(self.saved, self.root/'latest.pt')
        self.write('status.json', dict(state='TRAINING', step=3199, update_seconds=12345.,
            counters=self.saved['counters'], dust_counts=self.saved['dust_counts']))
        self.write('completion.json', dict(state='BUDGET_STOP', step=3199, signal_received=True,
            zeroth_order_only=True, budget_stop_is_failure=False))
        (self.root/'exit.txt').write_text('0\n')
        (self.root/'metrics.jsonl').write_text(json.dumps(dict(step=3199, fixture_history=True))+'\n')
        (self.root/'predictions').mkdir(exist_ok=True)
        np.savez_compressed(self.root/'predictions/step-03199.npz', step=np.array(3199),
                            predicted_token=np.zeros(9409, dtype=np.int64))
        self.seal()

    def run_context(self, **patches):
        stack = ExitStack()
        # The production runner requests eight threads; these guard tests keep
        # coordinator compute at the existing two-thread setting.
        stack.enter_context(patch.object(torch, 'set_num_threads'))
        stack.enter_context(patch.object(torch, 'set_num_interop_threads'))
        stack.enter_context(patch.dict(os.environ, {'WORLD_SIZE':'1'}))
        stack.enter_context(patch.object(scale.time, 'time', return_value=self.now))
        stack.enter_context(patch.object(scale.signal, 'signal', return_value=None))
        stack.enter_context(patch.object(torch.Tensor, 'backward', side_effect=AssertionError('backward forbidden')))
        stack.enter_context(patch('torch.autograd.grad', side_effect=AssertionError('autograd forbidden')))
        stack.enter_context(redirect_stdout(io.StringIO()))
        for name, value in patches.items():
            stack.enter_context(patch.object(scale, name, value))
        return stack

    def test_real_package_restores_every_state_without_aliasing(self):
        config, status, saved, plan, truth, training, receipt = scale.load_parent(self.root)
        self.assertEqual(status['step'],3199)
        self.assertEqual(receipt['files']['latest.pt'],scale.sha(self.root/'latest.pt'))
        self.assertEqual(plan['plan_sha256'],config['canonical_plan_sha256'])
        first = restore(saved, config); second = restore(saved, config)
        for model, opt, dust in (first,second):
            self.assertTrue(equal_nested(model.state_dict(),self.saved['model']))
            self.assertTrue(equal_nested(opt.state_dict(),self.saved['optimizer']))
            self.assertTrue(torch.equal(dust.rng.get_state(),self.saved['dust_rng']))
            self.assertEqual(dust.counts,self.saved['dust_counts'])
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            self.assertEqual(type(dust).__name__,'Dust')
            self.assertIsNone(dust.target_average)
        next(iter(first[1].state.values()))['exp_avg'].add_(1)
        self.assertTrue(equal_nested(second[1].state_dict(),self.saved['optimizer']))
        self.assertTrue(equal_nested(saved['optimizer'],self.saved['optimizer']))

    def test_labels_order_and_clean_partition_preserved_for_both_variants(self):
        original_x,original_y=self.truth['x'].clone(),self.truth['y'].clone()
        for variant in ('teacher_mix','oracle_mix'):
            with self.subTest(variant=variant):
                self.make_package(variant)
                config,_,_,plan,truth,training,_=scale.load_parent(self.root)
                spec=plan['variants'][variant]; indices=np.asarray(spec['train_indices'])
                targets=torch.tensor(spec['target_tokens'])
                self.assertEqual(config['training_examples'],2823)
                np.testing.assert_array_equal(training['pairs'],truth['pairs'][indices])
                np.testing.assert_array_equal(training['train'],np.arange(2823))
                self.assertTrue(torch.equal(training['x'][:,:5],truth['x'][indices,:5]))
                self.assertTrue(torch.equal(training['x'][:,5],targets))
                self.assertTrue(torch.equal(training['y'][:,4],targets))
                self.assertTrue((training['y'][:,:4]==-1).all())
                self.assertTrue((training['y'][:,5]==truth['vocab'].index('<|eos|>')).all())
                self.assertEqual(int((training['y']!=-1).sum()),5646)
                trained={tuple(sorted(map(int,p))) for p in training['pairs']}
                clean={tuple(sorted(map(int,p))) for p in truth['pairs'][plan['partitions']['clean_eval']]}
                self.assertFalse(trained & clean)
                if variant=='teacher_mix':
                    self.assertTrue((targets!=truth['y'][indices,4]).any(), 'Fixture must exercise non-oracle pseudo labels')
                else:self.assertTrue(torch.equal(targets,truth['y'][indices,4]))
        self.assertTrue(torch.equal(self.truth['x'],original_x))
        self.assertTrue(torch.equal(self.truth['y'],original_y))

    def test_receipt_requires_checkpoint_and_prediction_coverage_and_rejects_tampering(self):
        original=self.read('manifest.json')
        for name in ('latest.pt','config.json','metrics.jsonl','predictions/step-03199.npz'):
            manifest=copy.deepcopy(original);del manifest['files'][name]
            self.write('manifest.json',manifest)
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'omits'):
                scale.load_parent(self.root)
        self.write('manifest.json',original)
        with (self.root/'metrics.jsonl').open('a') as stream:stream.write('{}\n')
        with self.assertRaisesRegex(ValueError,'changed'):scale.load_parent(self.root)
        self.seal();manifest=self.read('manifest.json');manifest['files']['../outside']='0'*64
        self.write('manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'inside'):scale.load_parent(self.root)

    def test_parent_must_be_graceful_with_matching_step_horizon_and_recipe(self):
        cases=[('completion.json','signal_received',False,'gracefully'),
               ('completion.json','step',3198,'horizon'),
               ('status.json','step',3198,'horizon'),
               ('config.json','steps',3201,'horizon'),
               ('config.json','chunk',128,'recipe'),
               ('config.json','threads',16,'recipe'),
               ('config.json','inherited_teacher_weights',True,'pure-Dust')]
        for filename,key,value,message in cases:
            original=self.read(filename);changed=dict(original);changed[key]=value
            self.write(filename,changed);self.seal()
            with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,message):
                scale.load_parent(self.root)
            self.write(filename,original);self.seal()
        (self.root/'exit.txt').write_text('143\n');self.seal()
        with self.assertRaisesRegex(ValueError,'gracefully'):scale.load_parent(self.root)

    def test_frozen_provenance_equations_and_sample_count_are_bound(self):
        original=self.read('config.json')
        for key,value,message in [('canonical_plan_sha256','0'*64,'plan'),
                ('training_tensor_sha256','0'*64,'equations'),('training_examples',2822,'count')]:
            changed=copy.deepcopy(original);changed[key]=value
            self.write('config.json',changed);self.seal()
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,message):scale.load_parent(self.root)
        changed=copy.deepcopy(original);changed['source_hashes']['src/dust.py']='0'*64
        self.write('config.json',changed);self.seal()
        with self.assertRaisesRegex(ValueError,'source changed'):scale.load_parent(self.root)

    def test_restore_refuses_backprop_or_incomplete_adam(self):
        config,*_=scale.load_parent(self.root)
        for kind in ('backprop','adam_step','missing_adam'):
            bad=copy.deepcopy(self.saved)
            if kind=='backprop':bad['counters']['bp_forward_examples']=1
            elif kind=='adam_step':next(iter(bad['optimizer']['state'].values()))['step'].fill_(3198)
            else:bad['optimizer']['state'].pop(next(iter(bad['optimizer']['state'])))
            with self.subTest(kind=kind),self.assertRaises(ValueError):restore(bad,config)

    def test_expired_and_extended_deadlines_refuse_before_output_or_updates(self):
        config=self.read('config.json')
        deadline=config['start_unix']+config['seconds']
        for metadata in [dict(original_internal_deadline=self.now+179),
                         dict(original_external_deadline=self.now+179),
                         dict(original_internal_deadline=deadline+1),
                         dict(original_external_deadline=config['start_unix']+72001),
                         dict(original_internal_deadline=float('nan')),
                         dict(original_external_deadline=float('inf'))]:
            self.seal(**metadata)
            with self.subTest(metadata=metadata),self.run_context(),patch.object(scale,'update') as update:
                with self.assertRaisesRegex(ValueError,'budget exhausted|extend|finite'):scale.run(self.args)
                update.assert_not_called()
            self.assertFalse(self.args.output.exists())

    def test_output_collision_and_low_disk_refuse_without_updating(self):
        output=self.args.output/'rank0';output.mkdir(parents=True)
        marker=output/'keep.txt';marker.write_text('existing output must survive')
        with self.run_context(),patch.object(scale,'update') as update:
            with self.assertRaises(FileExistsError):scale.run(self.args)
            update.assert_not_called()
        self.assertEqual(marker.read_text(),'existing output must survive')
        self.args.output=self.base/'low-disk'
        with self.run_context(),patch('src.train.shutil.disk_usage') as usage,patch.object(scale,'update') as update:
            usage.return_value.free=5*1024**3-1
            with self.assertRaisesRegex(OSError,'Checkpoint refused'):scale.run(self.args)
            update.assert_not_called()
        self.assertFalse((self.args.output/'rank0').exists())

    def test_final_original_update_keeps_state_history_counters_and_absolute_deadlines(self):
        config=self.read('config.json');deadline=config['start_unix']+config['seconds']-10
        self.seal(original_internal_deadline=deadline,original_external_deadline=config['start_unix']+71990)
        before_hashes=dict(self.read('manifest.json')['files']); calls=[]
        def cheap_update(model,opt,dust,training,step,chunk):
            calls.append((step,chunk,len(training['train'])))
            self.assertTrue(equal_nested(model.state_dict(),self.saved['model']))
            self.assertTrue(equal_nested(opt.state_dict(),self.saved['optimizer']))
            self.assertEqual(tensor_sha(dust.rng.get_state()),tensor_sha(self.saved['dust_rng']))
            self.assertTrue(torch.equal(training['x'],self.training['x']))
            self.assertTrue(torch.equal(training['y'],self.training['y']))
            for p in model.parameters():p.grad=torch.zeros_like(p)
            opt.step();torch.randn(5,generator=dust.rng)
            dust.counts={k:v+1 for k,v in dust.counts.items()}
            return .5
        def cheap_evaluation(*_):
            return dict(evaluation_forward_examples=17,clean_eval_accuracy=.25),dict(predicted_token=np.zeros(9409,dtype=np.int64))
        with self.run_context(update=cheap_update,evaluate_student=cheap_evaluation):scale.run(self.args)
        output=self.args.output/'rank0'
        self.assertEqual(calls,[(3199,256,2823)])
        final=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        self.assertEqual(final['step'],3200)
        self.assertTrue(all(int(value['step'])==3200 for value in final['optimizer']['state'].values()))
        self.assertEqual(final['counters']['bp_forward_examples'],0)
        self.assertEqual(final['counters']['eval_forward_examples'],self.saved['counters']['eval_forward_examples']+17)
        self.assertEqual(final['counters']['supervised_token_presentations'],3200*2823*2)
        self.assertNotEqual(tensor_sha(final['dust_rng']),tensor_sha(self.saved['dust_rng']))
        self.assertEqual(scale.sha(output/'resume.pt'),before_hashes['latest.pt'])
        self.assertEqual(scale.sha(output/'predictions/step-03199.npz'),before_hashes['predictions/step-03199.npz'])
        self.assertTrue((output/'step-03200.pt').is_file())
        history=(output/'metrics.jsonl').read_text().splitlines()
        self.assertEqual(history[0],(self.root/'metrics.jsonl').read_text().strip())
        self.assertEqual([json.loads(row)['step'] for row in history],[3199,3200])
        completion=json.loads((output/'completion.json').read_text())
        self.assertEqual((completion['state'],completion['step']),('HORIZON_COMPLETE',3200))
        self.assertTrue(completion['original_deadlines_preserved'])
        migrated=json.loads((output/'config.json').read_text())
        self.assertEqual(migrated['start_unix'],config['start_unix'])
        self.assertEqual(migrated['seconds'],71400)
        self.assertEqual(migrated['execution_migration']['original_internal_deadline'],deadline)
        self.assertEqual(migrated['execution_migration']['restored_state']['adam_sha256'],adam_sha(self.opt))
        for name,value in before_hashes.items():self.assertEqual(scale.sha(self.root/name),value)

    def test_stop_before_first_update_keeps_usable_latest_checkpoint(self):
        def immediate_stop(sig,handler):
            if sig==signal.SIGTERM:handler(sig,None)
        with self.run_context(),patch.object(scale.signal,'signal',side_effect=immediate_stop),patch.object(scale,'update') as update:
            scale.run(self.args);update.assert_not_called()
        output=self.args.output/'rank0'
        self.assertEqual(scale.sha(output/'latest.pt'),scale.sha(self.root/'latest.pt'))
        self.assertEqual((output/'metrics.jsonl').read_bytes(),(self.root/'metrics.jsonl').read_bytes())
        completion=json.loads((output/'completion.json').read_text())
        self.assertEqual((completion['state'],completion['step']),('BUDGET_STOP',3199))
        self.assertTrue(completion['signal_received'])

    def test_read_only_parent_history_allows_new_predictions_and_metrics(self):
        metrics=self.root/'metrics.jsonl'
        predictions=self.root/'predictions'
        historical=predictions/'step-03199.npz'
        metrics.chmod(0o444);historical.chmod(0o444);predictions.chmod(0o555)
        modes={path:path.stat().st_mode & 0o777 for path in (metrics,predictions,historical)}
        hashes=dict(self.read('manifest.json')['files'])
        def cheap_update(model,opt,dust,training,step,chunk):
            self.assertEqual((step,chunk),(3199,256))
            for parameter in model.parameters():parameter.grad=torch.zeros_like(parameter)
            opt.step();torch.randn(5,generator=dust.rng)
            return .5
        def cheap_evaluation(*_):
            return dict(evaluation_forward_examples=17,clean_eval_accuracy=.25),dict(predicted_token=np.zeros(9409,dtype=np.int64))
        with self.run_context(update=cheap_update,evaluate_student=cheap_evaluation):scale.run(self.args)
        output=self.args.output/'rank0'
        self.assertTrue((output/'predictions/step-03200.npz').is_file())
        self.assertEqual(scale.sha(output/'predictions/step-03199.npz'),hashes['predictions/step-03199.npz'])
        history=(output/'metrics.jsonl').read_text().splitlines()
        self.assertEqual([json.loads(row)['step'] for row in history],[3199,3200])
        final=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        self.assertEqual(final['step'],3200)
        self.assertEqual(json.loads((output/'completion.json').read_text())['state'],'HORIZON_COMPLETE')
        self.assertTrue((output/'metrics.jsonl').stat().st_mode & 0o200)
        self.assertTrue((output/'predictions').stat().st_mode & 0o200)
        for path,mode in modes.items():self.assertEqual(path.stat().st_mode & 0o777,mode)
        for name,digest in hashes.items():self.assertEqual(scale.sha(self.root/name),digest)


if __name__=='__main__':unittest.main()
