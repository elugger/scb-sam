import unittest
from budget_protocol import BUDGETS, Budget


class BudgetTests(unittest.TestCase):
    def test_reference_totals(self):
        self.assertEqual([b.primary_pairs for b in BUDGETS],
                         [3420, 3240, 11880, 11340])
        self.assertEqual([b.hpq_reference_updates for b in BUDGETS],
                         [684, 648, 2376, 2268])

    def test_replay(self):
        self.assertEqual([b.replay_per_cycle for b in BUDGETS], [0, 0, 1, 4])

    def test_native_pair_iteration(self):
        for b in BUDGETS:
            p = b.iteration_plan(1, 1)
            self.assertEqual(p["full_iterations"], b.primary_pairs)
            self.assertIsNone(p["optimizer_updates"])

    def test_four_inputs_per_stream(self):
        for b in BUDGETS:
            p = b.iteration_plan(4, 4)
            self.assertEqual(p["full_iterations"] * 4, b.primary_pairs)
            self.assertEqual(p["remaining_unlabeled_inputs"], 0)

    def test_tail_explicit_not_dropped(self):
        p = BUDGETS[0].iteration_plan(7, 7)
        self.assertEqual(p["full_iterations"] * 7 + p["remaining_unlabeled_inputs"], 3420)
        self.assertGreater(p["remaining_unlabeled_inputs"], 0)

    def test_unequal_stream_requires_audit(self):
        with self.assertRaises(ValueError):
            BUDGETS[0].iteration_plan(2, 1)

    def test_invalid_counts(self):
        for v in (0, -1, 1.5, True):
            with self.subTest(v=v), self.assertRaises(ValueError):
                BUDGETS[0].iteration_plan(v, v)
            with self.subTest(v=v), self.assertRaises(ValueError):
                Budget("BY", 5, 5, v)

    def test_no_dispatch_or_runtime_claim(self):
        for b in BUDGETS:
            self.assertFalse(b.record()["new_baseline_dispatched"])
            self.assertFalse(b.iteration_plan(1, 1)["runtime_verified"])


if __name__ == "__main__":
    unittest.main()
