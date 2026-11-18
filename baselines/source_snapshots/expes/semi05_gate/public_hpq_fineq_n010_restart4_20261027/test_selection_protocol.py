"""CPU protocol tests only; no training, GPU access or remote mutations."""
import unittest
from pathlib import Path
from decimal import Decimal
from collections import Counter

import selection_protocol as p


def history(scores):
    return [dict(completed_epochs=i, successful_updates=i * 68, case_count=87,
                 mean_dice_percent=str(score)) for i, score in enumerate(scores, 1)]


class ProtocolTests(unittest.TestCase):
    def test_real_local_splits_and_fineq_sha(self):
        work = Path(__file__).resolve().parent.parent
        result = p.verify_inputs(work / 'public_ncase_20261019/splits',
                                 work / 'public_hpq_fineq_r3_20261020/fine_q.py')
        self.assertEqual(len(result['labeled']), 10)
        self.assertEqual(len(result['unlabeled']), 336)
        self.assertFalse(result['patient_alias_verified'])

    def test_budget_11_and_36(self):
        self.assertEqual(p.budget(11)['successful_updates'], 748)
        self.assertEqual(p.budget(11)['primary_labeled_inputs'], 3740)
        self.assertEqual(p.budget(36)['successful_updates'], 2448)
        self.assertEqual(p.budget(36)['primary_unlabeled_inputs'], 12240)
        self.assertEqual(p.budget(36)['ssl_inputs_per_labeled_case'], 1224)
        self.assertEqual(p.budget(36)['supervised_prefix_updates'], 600)

    def test_reject_wrong_budget(self):
        for value in (True, -1, 37, 11.0, '11'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                p.budget(value)

    def test_all_cycles_cover_10_and_336_with_4_replays(self):
        for epoch in range(36):
            trace = p.cycle_indices(epoch, list(reversed(range(10))))
            self.assertEqual(len(trace), 340)
            self.assertEqual(Counter(li for li, _, _ in trace), Counter({i: 34 for i in range(10)}))
            self.assertEqual({ui for _, ui, _ in trace}, set(range(336)))
            self.assertEqual(sum(replay for _, _, replay in trace), 4)
            self.assertEqual(len({ui for _, ui, replay in trace if replay}), 4)

    def test_reject_five_case_permutation(self):
        with self.assertRaises(ValueError):
            p.cycle_indices(0, list(range(5)))

    def test_flat_curve_earliest_stop_is_16(self):
        for epoch in range(1, 16):
            self.assertFalse(p.decision(history([70] * epoch))['stop'])
        result = p.decision(history([70] * 16))
        self.assertTrue(result['stop'])
        self.assertEqual(result['stop_reason'], 'PLATEAU_5_CYCLES')
        self.assertEqual(result['best_completed_epoch'], 1)

    def test_early_peak_sets_anchor_at_11(self):
        scores = [71] + [70] * 10 + [Decimal('70.9')] * 5
        self.assertTrue(p.decision(history(scores))['stop'])
        self.assertEqual(p.decision(history(scores))['best_completed_epoch'], 1)

    def test_small_gain_updates_best_without_resetting_patience(self):
        result = p.decision(history([70] * 11 + [Decimal('70.01')] * 5))
        self.assertTrue(result['stop'])
        self.assertEqual(result['best_completed_epoch'], 12)

    def test_cumulative_threshold_resets_inclusive(self):
        scores = [70] * 11 + [Decimal('70.01'), Decimal('70.02'), Decimal('70.03'),
                             Decimal('70.04'), Decimal('70.05')]
        result = p.decision(history(scores))
        self.assertFalse(result['stop'])
        self.assertEqual(result['stale_epochs_after_warmup'], 0)

    def test_budget_cap_on_improving_curve(self):
        scores = [Decimal('70') + Decimal('0.1') * i for i in range(36)]
        result = p.decision(history(scores))
        self.assertEqual(result['stop_reason'], 'MAX_36_CYCLES')
        self.assertEqual(result['best_completed_epoch'], 36)

    def test_reject_post_stop_training(self):
        with self.assertRaises(ValueError):
            p.decision(history([70] * 17))

    def test_reject_partial_gapped_nonfinite_history(self):
        for key, value in [('case_count', 86), ('completed_epochs', 2),
                           ('successful_updates', 69), ('mean_dice_percent', 'NaN'),
                           ('mean_dice_percent', 'Infinity')]:
            data = history([70])
            data[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                p.decision(data)

    def test_complete_ordered_87_case_aggregation(self):
        ids = ['case_%03d' % i for i in range(87)]
        rows = [dict(case_id=case, dice=.7) for case in ids]
        result = p.evaluation_record(1, 68, rows, ids)
        self.assertEqual(Decimal(result['mean_dice_percent']), Decimal(70))
        self.assertEqual(p.decision([result])['best_completed_epoch'], 1)

    def test_reject_incomplete_wrong_order_invalid_units(self):
        ids = ['case_%03d' % i for i in range(87)]
        rows = [dict(case_id=case, dice=.7) for case in ids]
        for wrong in (rows[:-1], list(reversed(rows)),
                      [dict(case_id=case, dice=70) for case in ids],
                      [dict(case_id=case, dice=float('nan')) for case in ids]):
            with self.assertRaises(ValueError):
                p.evaluation_record(1, 68, wrong, ids)

    def test_empty_is_not_training_or_checkpoint_proof(self):
        result = p.decision([])
        self.assertFalse(result['stop'])
        self.assertIsNone(result['best_completed_epoch'])
        self.assertFalse(result['trainer_integrated'])
        self.assertFalse(result['checkpoint_receipts_verified'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
