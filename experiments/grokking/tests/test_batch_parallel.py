import copy
import argparse
from datetime import timedelta
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist

from src.batch_parallel import noise_shapes, parallel_update
from src.dust import Dust, Settings
from src.model import GrokTransformer
from src.task import dataset
from src.train import optimizer, update


def forbidden(*args, **kwargs):
    raise AssertionError('Backward is forbidden in original-Dust subbatch execution')


def forbid_backward():
    return (patch.object(torch.Tensor, 'backward', forbidden),
            patch('torch.autograd.backward', forbidden), patch('torch.autograd.grad', forbidden))


def restore(saved):
    model = GrokTransformer(seed=9278700)
    model.requires_grad_(False)
    model.load_state_dict(saved['model'])
    opt = optimizer(model)
    opt.load_state_dict(copy.deepcopy(saved['optimizer']))
    dust = Dust(model, Settings(**saved['settings']), seed=7)
    dust.rng.set_state(saved['rng'])
    dust.counts = dict(saved['counts'])
    return model, opt, dust


def state(model, opt, dust):
    return dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
                rng=dust.rng.get_state(), counts=dict(dust.counts),
                gradients={name:p.grad.clone() for name,p in model.parameters_named().items()})


def distributed_worker(rank, directory, init_file):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(directory)
    saved = torch.load(root/'input.pt', weights_only=False)
    model, opt, dust = restore(saved)
    dist.init_process_group('gloo', init_method='file://'+init_file,
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    results = []
    try:
        a,b,c = forbid_backward()
        with a,b,c:
            for step in (12, 13):
                loss = parallel_update(model,opt,dust,saved['data'],step,chunk=2)
                results.append(dict(loss=loss, **state(model,opt,dust)))
        torch.save(dict(results=results, physical=dict(dust.batch_parallel_counts)), root/f'rank{rank}.pt')
    finally:
        dist.destroy_process_group()


class BatchParallelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self):
        data = dataset(); ids = data['train'][:5]
        x=data['x'][ids].clone(); y=data['y'][ids].clone()
        # Uneven supervised-token weights, including the final one-example batch
        # where source head batch centering is disabled.
        y[2,5] = -1
        return dict(x=x,y=y,train=np.arange(5),pairs=data['pairs'][ids].copy())

    def new(self, population=16, nrep=4):
        model=GrokTransformer();model.requires_grad_(False)
        settings=Settings(population=population,nrep=nrep,direct_sigma=.002,
                          hub_last_sigma=.004,local_sigma=.05,head_sigma=.05)
        return model,optimizer(model),Dust(model,settings,seed=12345)

    def assert_recursive_equal(self, left, right, location='root'):
        if isinstance(left,torch.Tensor):
            self.assertTrue(torch.equal(left,right),location)
        elif isinstance(left,dict):
            self.assertEqual(set(left),set(right),location)
            for key in left:self.assert_recursive_equal(left[key],right[key],location+'.'+str(key))
        elif isinstance(left,(list,tuple)):
            self.assertEqual(len(left),len(right),location)
            for i,(a,b) in enumerate(zip(left,right)):self.assert_recursive_equal(a,b,location+'.'+str(i))
        else:self.assertEqual(left,right,location)

    def test_static_replay_matches_actual_all_site_draws_and_rng_uneven_shapes(self):
        data=self.data()
        for population,nrep in [(16,4),(5,3)]:
            model,opt,dust=self.new(population,nrep)
            for batch in (2,1):
                before=dust.rng.get_state(); observed=[]
                def traced(*shape):
                    observed.append(shape)
                    return Dust.noise(dust,*shape)
                dust.noise=traced
                a,b,c=forbid_backward()
                with a,b,c:dust.gradients(data['x'][:batch],data['y'][:batch])
                after=dust.rng.get_state()
                expected=list(noise_shapes(dust,batch,6))
                self.assertEqual(observed,expected)
                replay=Dust(model,dust.s,seed=0);replay.rng.set_state(before)
                for shape in expected:replay.noise(*shape)
                self.assertTrue(torch.equal(after,replay.rng.get_state()))
                self.assertIn((min(nrep,population),batch,6,512),expected)
                self.assertIn((min(nrep,population*4),batch,6,239),expected)

    def test_world_one_is_bitwise_serial_with_original_sparse_chunk_weights(self):
        model,opt,dust=self.new(); data=self.data()
        saved=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
                   rng=dust.rng.get_state(),counts=dict(dust.counts),settings=vars(dust.s))
        other,other_opt,other_dust=restore(saved)
        a,b,c=forbid_backward()
        with a,b,c:
            serial_loss=update(model,opt,dust,data,12,2)
            parallel_loss=parallel_update(other,other_opt,other_dust,data,12,2)
        self.assertEqual(serial_loss,parallel_loss)
        self.assert_recursive_equal(state(model,opt,dust),state(other,other_opt,other_dust))
        self.assertEqual(other_dust.batch_parallel_counts['replayed_noise_calls'],0)

    def test_two_ranks_exact_model_adam_gradients_rng_loss_and_logical_counters(self):
        model,opt,dust=self.new();data=self.data()
        # Warm Adam and logical counters before branching, as in live migration.
        a,b,c=forbid_backward()
        with a,b,c:update(model,opt,dust,data,11,2)
        saved=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
                   rng=dust.rng.get_state(),counts=dict(dust.counts),settings=vars(dust.s),data=data)
        expected=[]
        a,b,c=forbid_backward()
        with a,b,c:
            for step in (12,13):
                loss=update(model,opt,dust,data,step,2)
                expected.append(dict(loss=loss,**state(model,opt,dust)))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);torch.save(saved,root/'input.pt')
            context=mp.get_context('spawn')
            processes=[context.Process(target=distributed_worker,args=(rank,temporary,str(root/'gloo-init')))
                       for rank in range(2)]
            for process in processes:process.start()
            deadline=time.monotonic()+90
            try:
                for process in processes:process.join(max(0,deadline-time.monotonic()))
                self.assertTrue(all(not p.is_alive() for p in processes),'Finite collective test timeout')
                self.assertEqual([p.exitcode for p in processes],[0,0])
                result=[torch.load(root/f'rank{rank}.pt',weights_only=False) for rank in range(2)]
                for rank in range(2):self.assert_recursive_equal(expected,result[rank]['results'],f'rank{rank}')
                self.assertEqual(result[0]['physical']['owned_subbatches'],4)
                self.assertEqual(result[0]['physical']['replayed_subbatches'],2)
                self.assertEqual(result[1]['physical']['owned_subbatches'],2)
                self.assertEqual(result[1]['physical']['replayed_subbatches'],4)
                self.assertEqual(result[0]['physical']['owned_noise_calls'],result[1]['physical']['replayed_noise_calls'])
                self.assertEqual(result[0]['physical']['replayed_noise_values'],result[1]['physical']['owned_noise_values'])
                self.assertEqual(expected[-1]['counts']['clean_forwards'],9)
            finally:
                for process in processes:
                    if process.is_alive():process.terminate();process.join(5)

    def test_rejects_population_collectives_cpudust_empty_and_invalid_chunk(self):
        from src.cpu_dust import CpuDust
        model,opt,dust=self.new();data=self.data()
        dust.target_average=lambda value:value
        with self.assertRaises(ValueError):parallel_update(model,opt,dust,data,11,2)
        dust.target_average=None
        with self.assertRaises(ValueError):parallel_update(model,opt,dust,data,11,0)
        empty=dict(data,train=np.array([],dtype=np.int64))
        with self.assertRaises(ValueError):parallel_update(model,opt,dust,empty,11,2)
        cpu=CpuDust(model,dust.s,seed=1)
        with self.assertRaises(TypeError):parallel_update(model,opt,cpu,data,11,2)

    def test_audit_state_hash_tracks_content_shape_dtype_and_adam(self):
        from src.batch_parallel_audit import state_hash
        self.assertEqual(state_hash({'a':torch.tensor([1.]),'b':[2,3]}),
                         state_hash({'b':[2,3],'a':torch.tensor([1.])}))
        values=[torch.tensor([1.]),torch.tensor([2.]),torch.tensor([[1.]]),torch.tensor([1.],dtype=torch.float64)]
        self.assertEqual(len({state_hash(value) for value in values}),len(values))
        original=dict(state={0:dict(step=torch.tensor(7.),exp_avg=torch.tensor([.25]),exp_avg_sq=torch.tensor([.125]))})
        changed=copy.deepcopy(original);changed['state'][0]['exp_avg_sq'][0]=.126
        self.assertNotEqual(state_hash(original),state_hash(changed))

    def test_discarded_audit_driver_uses_saved_next_step_and_never_changes_inputs(self):
        from src.batch_parallel_audit import hashes,run
        model,opt,dust=self.new();data=self.data()
        saved=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
                   rng=dust.rng.get_state(),counts=dict(dust.counts),settings=vars(dust.s),
                   step=12,counters=dict(bp_forward_examples=0,eval_forward_examples=3,supervised_token_presentations=9))
        expected,expected_opt,expected_dust=restore(saved)
        a,b,c=forbid_backward()
        with a,b,c:loss=update(expected,expected_opt,expected_dust,data,12,256)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);inputs=[]
            for name in ('latest.pt','config.json','plan.json'):
                path=root/name;path.write_bytes(('immutable:'+name).encode());inputs.append(path)
            args=argparse.Namespace(checkpoint=inputs[0],config=inputs[1],plan=inputs[2],
                                    output=root/'audit',mode='serial',steps=1,threads=1)
            a,b,c=forbid_backward()
            with a,b,c,patch('src.batch_parallel_audit.restore_input',return_value=(model,opt,dust,data,saved,{})), \
                 patch('src.batch_parallel_audit.torch.set_num_interop_threads'),patch('builtins.print'):
                run(args)
            final=torch.load(args.output/'final-state.pt',weights_only=False)
            import json
            report=json.loads((args.output/'results.json').read_text())
            self.assertEqual(report['rows'][0]['loss'],loss)
            self.assertEqual(final['step'],13)
            self.assertEqual(report['final_hashes'],hashes(expected,expected_opt,expected_dust))
            self.assertEqual(final['counters']['supervised_token_presentations'],18)
            self.assertEqual(final['counters']['bp_forward_examples'],0)
            self.assertFalse(report['candidate_retained_for_training'])
            for path in inputs:self.assertEqual(path.read_bytes(),('immutable:'+path.name).encode())


if __name__=='__main__':unittest.main()
