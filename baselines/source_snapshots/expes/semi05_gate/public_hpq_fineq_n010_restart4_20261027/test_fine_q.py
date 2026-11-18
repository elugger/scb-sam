import copy
import unittest
import torch
import torch.nn.functional as F
from fine_q import context_targets, context_objectives, validate_context_row, SCHEMA


class FineQTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.signal = torch.ones((1, 1, 32, 32, 32), dtype=torch.bool)
        self.p = torch.zeros((1, 1, 128, 128, 128))

    def test_uniform_matches_original_three_axes(self):
        for value in (0., 1.):
            self.p.fill_(value)
            for axis in range(3):
                m, t, s = context_targets(self.p, self.p, self.signal, axis)
                self.assertTrue(torch.equal(m, F.interpolate(s.float(), scale_factor=4,
                                                             mode="nearest").bool()[:, 0]))
                self.assertTrue(bool((t[m] == value).all()))
                self.assertEqual(int(m.sum()), 128 ** 3 // 2)

    def test_single_voxel_not_erased(self):
        self.p[..., 64, 64, 64] = 1.
        for axis in range(3):
            m, t, _ = context_targets(self.p, self.p, self.signal, axis)
            self.assertTrue(m[0, 64, 64, 64])
            self.assertEqual(t[0, 64, 64, 64], 1)
            self.assertLess(F.avg_pool3d(self.p, 4, 4)[0, 0, 16, 16, 16], .3)

    def test_uncertainty_disagreement_and_signal(self):
        for pa, pb in ((.5, .5), (1., 0.), (0., 1.)):
            m, _, _ = context_targets(self.p + pa, self.p + pb, self.signal, 0)
            self.assertFalse(bool(m.any()))
        m, _, _ = context_targets(self.p, self.p, self.signal & False, 0)
        self.assertFalse(bool(m.any()))

    def test_thresholds_and_geometry(self):
        for p in (.3, .7):
            m, t, _ = context_targets(self.p + p, self.p + p, self.signal, 2)
            self.assertEqual(int(m.sum()), 128 ** 3 // 2)
            self.assertTrue(bool((t[m] == int(p >= .5)).all()))
        with self.assertRaises(ValueError):
            context_targets(self.p.requires_grad_(), self.p, self.signal, 0)
        with self.assertRaises(ValueError):
            context_targets(self.p.detach() + float("nan"), self.p.detach(), self.signal, 0)

    def test_balanced_reference_and_gradients(self):
        logits = torch.zeros((1, 2, 128, 128, 128), requires_grad=True)
        self.p[..., 64, 64, 64] = 1.
        m, t, _ = context_targets(self.p, self.p, self.signal, 0)
        balanced, voxel, counts, _ = context_objectives(logits, m, t)
        logp = logits.log_softmax(1)
        ref = (-logp[:, 0][m & (t == 0)].mean() - logp[:, 1][m & (t == 1)].mean()) / 2
        ga, = torch.autograd.grad(balanced, logits, retain_graph=True)
        gb, = torch.autograd.grad(ref, logits)
        self.assertTrue(torch.equal(ga, gb))
        self.assertEqual(counts[1], 1)
        self.assertTrue(torch.allclose(balanced, voxel))
        self.assertTrue(bool(torch.isfinite(ga).all()))

    def test_empty_mask_zero_gradient(self):
        logits = torch.zeros((1, 2, 128, 128, 128), requires_grad=True)
        m, t, _ = context_targets(self.p + .5, self.p + .5, self.signal, 0)
        loss, _, counts, _ = context_objectives(logits, m, t)
        grad, = torch.autograd.grad(loss, logits)
        self.assertEqual(counts, [0, 0])
        self.assertEqual(float(grad.abs().sum()), 0.)

    def test_voxel_ledger_rejects_wrong_count_or_boundary(self):
        row = dict(schema=SCHEMA, reduction="B", coefficient=1., applied=True,
                   target_resolution=[128] * 3, support_resolution=[32] * 3,
                   declared_state_restored=True, no_gt_argument=True, relation_computed=False,
                   foreground_voxels=1, background_voxels=2, reliable_voxels=3,
                   support_blocks=1, raw_gradient_norms={str(i): .1 for i in range(6)})
        validate_context_row(row, "B")
        for key, value in (("reliable_voxels", 64), ("support_blocks", 0),
                           ("no_gt_argument", False), ("target_resolution", [32] * 3)):
            changed = copy.deepcopy(row)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_context_row(changed, "B")


if __name__ == "__main__":
    unittest.main()
