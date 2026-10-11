import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.prediction_stability import (ENDPOINT, HISTORY, audit_transition,
    commit_selection, evaluate_after_commit, read_distribution_inputs,
    read_partitions, read_top1, select_pseudo_labels, top1_gap)


class PredictionStabilityChecks(unittest.TestCase):
    def fixture(self):
        pairs = np.array([[0,1], [1,0], [0,0], [1,1], [0,2], [2,0], [2,2]])
        pred = np.array([1,1,2,2,1,1,2])
        return dict(pairs=pairs, candidate_indices=list(range(6)), clean_eval_indices=[6],
                    valid_token_ids=[1,2], predicted_token=pred,
                    confidence_gap=np.arange(7, dtype=float)+1, current_step=ENDPOINT,
                    historical_predictions=np.tile(pred, (6,1)), historical_steps=HISTORY,
                    peer_predictions=np.tile(pred, (6,1)), peer_step=ENDPOINT,
                    rule='confidence', count=3)

    def test_confidence_unique_groups_ties_and_insufficient_result(self):
        f = self.fixture(); selected = select_pseudo_labels(**f)
        self.assertEqual(selected['selected_indices'], [5,3,2])
        f['confidence_gap'][:] = 1.
        self.assertEqual(select_pseudo_labels(**f)['selected_indices'], [0,2,3])
        f['count'] = 5; insufficient = select_pseudo_labels(**f)
        self.assertEqual(insufficient['status'], 'INSUFFICIENT_POOL')
        self.assertEqual(insufficient['eligible_unordered'], 4)
        self.assertEqual(insufficient['selected_indices'], [])

    def test_reversed_clean_pair_is_excluded_before_selection(self):
        f = self.fixture(); f['clean_eval_indices'] = [1,6]; f['candidate_indices'] = [0,2,3]
        with self.assertRaises(ValueError): select_pseudo_labels(**f)

    def test_temporal_and_replica_priority_differ_from_confidence(self):
        f = self.fixture(); f['candidate_indices'] = [0,2]; f['count'] = 1
        f['confidence_gap'][0] = .01; f['confidence_gap'][2] = 20.
        f['historical_predictions'][:5,2] = 1
        f['peer_predictions'][1:,0] = 2
        self.assertEqual(select_pseudo_labels(**f)['selected_indices'], [2])
        f['rule'] = 'temporal'
        self.assertEqual(select_pseudo_labels(**f)['selected_indices'], [0])
        f['rule'] = 'replica'
        self.assertEqual(select_pseudo_labels(**f)['selected_indices'], [2])

    def test_future_history_peer_or_current_step_rejected(self):
        for field, value in [('current_step',3200), ('peer_step',3200),
                             ('historical_steps',(*HISTORY[:-1],3200))]:
            f = self.fixture(); f[field] = value
            with self.assertRaises(ValueError): select_pseudo_labels(**f)
        f = self.fixture(); f['historical_predictions'][-1,0] = 2
        with self.assertRaises(ValueError): select_pseudo_labels(**f)

    def test_selector_has_no_label_fields_and_confidence_ignores_other_agreement(self):
        parameters = set(inspect.signature(select_pseudo_labels).parameters)
        self.assertFalse(parameters & {'truth','labels','correct','margin','ce','answer_ce','future_predictions'})
        f = self.fixture(); before = select_pseudo_labels(**f)['selected_indices']
        f['historical_predictions'][:-1] = 2; f['peer_predictions'][:] = 2
        self.assertEqual(select_pseudo_labels(**f)['selected_indices'], before)
        with self.assertRaises(TypeError): select_pseudo_labels(**f, truth=np.arange(7))

    def test_poisoned_true_metadata_never_deserialized_by_input_readers(self):
        f = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)/'history.npz'
            data = dict(step=np.array(ENDPOINT), predicted_token=f['predicted_token'], pairs=f['pairs'],
                        train=np.zeros(7,bool), unseen_pair=np.ones(7,bool), answer_logits=np.zeros((7,239)))
            np.savez(p, **data, correct=np.array([dict(poison=True)],dtype=object),
                     margin=np.array([None],dtype=object), answer_ce=np.array([None],dtype=object),
                     target_token=np.array([dict(poison=True)],dtype=object))
            history = read_top1(p, ENDPOINT); current = read_distribution_inputs(p, ENDPOINT)
            self.assertEqual(set(history), {'predicted_token','pairs','train','unseen_pair'})
            self.assertEqual(set(current), {'answer_logits','pairs','train','unseen_pair'})
            np.testing.assert_array_equal(history['predicted_token'], f['predicted_token'])
            plan = Path(temporary)/'plan.json'
            plan.write_text(json.dumps(dict(partitions=dict(U=[1,2],pseudo_pool=[1],clean_eval=[2]),
                                            selected_teacher_audit=dict(correct=999,selected_true_tokens=['POISON'])),indent=2))
            self.assertEqual(read_partitions(plan), dict(U=[1,2],pseudo_pool=[1],clean_eval=[2]))

    def test_full239_gap_invalid_top1_and_ensemble_without_any_matching_vote(self):
        z = np.zeros((7,239)); z[:,1] = 3.; z[5,0] = 100.
        pred, gap = top1_gap(z)
        f = self.fixture(); f['predicted_token']=pred; f['confidence_gap']=gap
        f['historical_predictions']=np.tile(pred,(6,1))
        self.assertNotIn(5, select_pseudo_labels(**f)['selected_indices'])
        f['rule']='ensemble'; f['predicted_token'][:]=2
        f['peer_predictions'][:]=1; f['historical_predictions']=None; f['historical_steps']=()
        chosen=select_pseudo_labels(**f)
        self.assertEqual(chosen['pseudo_tokens'], [2,2,2])
        self.assertTrue(all(v['replica_agreement']==0 for v in chosen['selected_features']))

    def test_commit_is_required_and_mutation_detected_before_labels_or_future_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); manifest=dict(selections=[dict(selected_indices=[1],pseudo_tokens=[2])])
            arguments=dict(output=root,distributions=root,pairs=None,partitions=None,current=None,
                           histories=None,metas=None,manifest=manifest,commit={},predicted=None,ensemble_pred=None)
            with patch('src.prediction_stability.np.load') as opened:
                with self.assertRaises(ValueError): evaluate_after_commit(**arguments)
                opened.assert_not_called()
            commit=commit_selection(root,manifest)
            changed=copy.deepcopy(manifest);changed['selections'][0]['selected_indices']=[2]
            arguments.update(commit=commit,manifest=changed)
            with patch('src.prediction_stability.np.load') as opened:
                with self.assertRaises(ValueError): evaluate_after_commit(**arguments)
                opened.assert_not_called()

    def test_stable_wrong_and_future_correctness_partition_independent_toy(self):
        row=audit_transition(np.array([1,2,3,4,5]),np.array([1,9,3,7,8]),np.array([1,2,0,7,0]))
        self.assertEqual(row['pseudo_correct'],2);self.assertEqual(row['future_correct'],2)
        self.assertEqual(row['stable_correct'],1);self.assertEqual(row['stable_wrong'],1)
        self.assertEqual(row['correct_lost'],1);self.assertEqual(row['wrong_corrected'],1)
        self.assertEqual(row['wrong_to_different_wrong'],1)
        self.assertEqual(row['stable_correct']+row['stable_wrong'],row['retained'])


if __name__=='__main__': unittest.main()
