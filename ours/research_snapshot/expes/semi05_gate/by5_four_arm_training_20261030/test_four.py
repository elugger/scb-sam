import copy, json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from four_contract import GROUPS, group_config, check_step

def valid_row(group, overflow=False):
    c = dict(reduction=GROUPS[group][1], coefficient=1.0, applied=group in ('HQ', 'HPQ'), no_gt_argument=True, declared_state_restored=True, relation_computed=False, foreground_voxels=32, background_voxels=32, reliable_blocks=1)
    return dict(skipped=overflow, teacher_no_grad=True, clip_calls=0 if overflow else 1, ema_calls=0 if overflow else 1, context_computed=True, gradient_groups={str(i): 1.0 for i in range(6)}, microbatches=[dict(context=copy.deepcopy(c), supervised=0.4, pseudo=0.1, wsv=0.02, baseline_total=0.52) for _ in range(5)])

class ContractTests(unittest.TestCase):

    def test_common_settings_and_original_unmodified(self):
        ancestor = {'experiment': {'seed': 999, 'gpu': '1'}, 'semi': {'ucp_epochs': 36, 'ucp_accumulation': 5, 'xsa_cross_gain': 0.5}}
        old = copy.deepcopy(ancestor)
        for phase in ('bridge', 'formal'):
            rows = []
            for g in GROUPS:
                r = group_config(ancestor, g, phase)
                self.assertEqual(r['experiment']['seed'], 2027)
                self.assertEqual((r['semi']['matrix_group'], r['semi']['matrix_c']), GROUPS[g])
                r['experiment'].pop('output_dir')
                for k in ('paired_checkpoint_dir', 'matrix_group', 'matrix_c'):
                    r['semi'].pop(k)
                rows.append(r)
            self.assertTrue(all((r == rows[0] for r in rows)))
        self.assertEqual(ancestor, old)

    def test_changed_budget_rejected(self):
        for epochs, accum in ((35, 5), (36, 4)):
            with self.assertRaises(ValueError):
                group_config({'experiment': {}, 'semi': {'ucp_epochs': epochs, 'ucp_accumulation': accum}}, 'H', 'formal')

    def test_each_arm_and_artificial_overflow(self):
        for g in GROUPS:
            check_step(valid_row(g), g)
            check_step(valid_row(g, True), g, True)

    def test_accidental_overflow_is_not_silently_counted(self):
        for g in GROUPS:
            with self.assertRaises(ValueError):
                check_step(valid_row(g, True), g)

    def test_invalid_execution_rejected(self):
        bad = [('clip_calls', 0), ('ema_calls', 2), ('teacher_no_grad', False), ('context_computed', False), ('microbatches', []), ('gradient_groups', {'a': float('nan')}), ('gradient_groups', {'a': 0.0})]
        for k, v in bad:
            r = valid_row('HQ')
            r[k] = v
            with self.subTest(k=k, v=v), self.assertRaises(ValueError):
                check_step(r, 'HQ')

    def test_factor_boundary_rejected(self):
        bad = [('reduction', '0'), ('coefficient', 0.5), ('applied', False), ('no_gt_argument', False), ('declared_state_restored', False), ('relation_computed', True), ('foreground_voxels', 33)]
        for k, v in bad:
            r = valid_row('HQ')
            r['microbatches'][0]['context'][k] = v
            with self.subTest(k=k), self.assertRaises(ValueError):
                check_step(r, 'HQ')

    def test_nonfinite_loss_rejected(self):
        r = valid_row('H')
        r['microbatches'][0]['supervised'] = float('inf')
        with self.assertRaises(ValueError):
            check_step(r, 'H')
if __name__ == '__main__':
    unittest.main(verbosity=2)
