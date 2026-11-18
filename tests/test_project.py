import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class ProjectEntryTests(unittest.TestCase):
    def command(self, *arguments):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'reproduce.py'), *arguments], capture_output=True, text=True, encoding='utf-8')
        return result, json.loads(result.stdout) if result.stdout.strip() else None

    def test_exact_scope_and_unique_routes(self):
        result, rows = self.command('list')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({r['id'] for r in rows}), 32)
        self.assertEqual(sum(r['table'] == 'main' for r in rows), 28)
        self.assertEqual(sum(r['table'] == 'ablation' for r in rows), 4)

    def test_hq_result_and_no_historical_weight_claim(self):
        result, row = self.command('report', '--experiment', 'ablation_HQ')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertAlmostEqual(row['metrics']['dice'] * 100, 66.90289753904912, places=10)
        self.assertFalse(row['checkpoint_available'])

    def test_valid_structure_does_not_claim_training_readiness(self):
        result, row = self.command('verify')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertTrue(row['structural_checks_passed'], row['errors'])
        self.assertEqual(row['status'], 'NOT_READY_TO_RELEASE')
        self.assertFalse(row['readiness']['clean_environment_training_verified'])
        self.assertFalse(row['training_started'])
        self.assertEqual(row['weight_binaries_in_repository'], 0)

    def test_plan_keeps_real_group_and_no_launch(self):
        result, row = self.command('plan', '--experiment', 'ablation_HPQ')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(row['recorded_route']['group'], 'HPQ')
        self.assertFalse(row['training_started'])
        self.assertFalse(row['recorded_route']['execution']['enabled'])

    def test_verify_from_isolated_interpreter_outside_project(self):
        result = subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'reproduce.py'), 'verify'], cwd=ROOT.parent, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 2, result.stderr)
        row = json.loads(result.stdout)
        self.assertTrue(row['structural_checks_passed'], row['errors'])
        self.assertFalse(row['training_started'])

    def test_unqualified_train_is_rejected(self):
        result, row = self.command('train', '--experiment', 'BY_scb_seg_n005')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(row['status'], 'TRAINING_NOT_DISPATCHED')
        self.assertFalse(row['training_started'])

    def test_unknown_experiment_is_rejected(self):
        result, row = self.command('report', '--experiment', '../../private')
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(row)
        self.assertIn('Unknown experiment', result.stderr)

if __name__ == '__main__':
    unittest.main()
