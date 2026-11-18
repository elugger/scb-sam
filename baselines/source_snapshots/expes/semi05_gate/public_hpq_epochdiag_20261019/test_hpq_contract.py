import copy
import tempfile
from pathlib import Path, PurePosixPath
import unittest
from unittest.mock import patch
import hpq_contract as c
from strict_hpq_data import sample_boundary, make_dataset


def splits():
    lab = ['L%03d' % i for i in range(17)]
    unlab = ['U%03d' % i for i in range(329)]
    return dict(train_lab=lab, train_unlab=unlab, train_whole=lab + unlab,
                test=['T%03d' % i for i in range(87)])


def trace(epoch):
    s = splits()
    return [dict(epoch=epoch, step=step, labeled_index=step % 17, unlabeled_index=t[2],
                 labeled_case_id=s['train_lab'][step % 17], unlabeled_case_id=s['train_unlab'][t[2]],
                 unlabeled_replay=t[1], padding=bool(t[1]),
                 padding_case_id=s['train_unlab'][t[2]] if t[1] else None,
                 unlabeled_label_returned=False, gt_roi_used=True)
            for step, t in enumerate(c.epoch_tokens(epoch))]


class ContractTest(unittest.TestCase):
    def test_scope(self):
        for phase in ('bridge', 'formal'):
            c.validate_scope(2, 2027, phase, 0)

    def test_scope_rejects_other_fold(self):
        for fold in (0, 1, 3, 4, True, 2.0):
            with self.assertRaises(ValueError): c.validate_scope(fold, 2027, 'formal', 0)

    def test_scope_rejects_other_seed_phase_gpu(self):
        for args in ((2, 2028, 'formal', 0), (2, 2027, 'test', 0), (2, 2027, 'formal', True)):
            with self.assertRaises(ValueError): c.validate_scope(*args)

    def test_initialization(self):
        a = dict(status='VERIFIED_COMPLETE', final_sha256=c.INIT_SHA, final_metrics_sha256=c.METRICS_SHA)
        c.validate_initialization(c.INIT, c.INIT_SHA, a)
        for path in (str(c.INIT).replace('sup05', 'sup100'), str(c.INIT).replace('final', 'best')):
            with self.assertRaises(ValueError): c.validate_initialization(path, c.INIT_SHA, a)

    def test_initialization_acceptance_binding(self):
        for a in ({}, dict(status='VERIFIED_COMPLETE', final_sha256=c.INIT_SHA, final_metrics_sha256='0'*64)):
            with self.assertRaises(ValueError): c.validate_initialization(c.INIT, c.INIT_SHA, a)

    def test_checkpoint_metadata(self):
        p = dict(format='public_basic_sup_fixed_final_student_v1', completed_epochs=60, epoch=59,
                 successful_updates=1020, test_evaluations_before_save=0,
                 selection_rule='fixed_epoch_60_student', student={'example': object()})
        c.validate_checkpoint_metadata(p)
        for key, value in (('successful_updates', 2100), ('epoch', 58), ('test_evaluations_before_save', 1),
                           ('selection_rule', 'test_best'), ('student', {})):
            with self.assertRaises(ValueError): c.validate_checkpoint_metadata(dict(p, **{key: value}))

    def test_split_boundary(self):
        self.assertFalse(c.validate_splits(splits())['patient_alias_boundary_verified'])

    def test_duplicate_rejected(self):
        s = splits(); s['train_lab'][1] = s['train_lab'][0]
        with self.assertRaises(ValueError): c.validate_splits(s)

    def test_test_overlap_rejected(self):
        s = splits(); s['test'][0] = s['train_lab'][0]
        with self.assertRaises(ValueError): c.validate_splits(s)

    def test_train_union_rejected(self):
        s = splits(); s['train_whole'][0] = 'missing'
        with self.assertRaises(ValueError): c.validate_splits(s)

    def test_traversal_rejected(self):
        s = splits(); s['test'][0] = '../case'
        with self.assertRaises(ValueError): c.validate_splits(s)

    def test_all_epochs_exact_coverage(self):
        for epoch in range(36):
            r = c.validate_epoch_trace(trace(epoch), splits(), epoch, list(range(17)))
            self.assertEqual(r['padding_case_id'], splits()['train_unlab'][epoch])

    def test_token_boundaries(self):
        for epoch in (-1, 36, True, 0.0):
            with self.assertRaises(ValueError): c.epoch_tokens(epoch)

    def test_actual_case_substitution_rejected(self):
        r = trace(0); r[0]['unlabeled_case_id'] = 'U001'
        with self.assertRaises(ValueError): c.validate_epoch_trace(r, splits(), 0, list(range(17)))

    def test_partial_epoch_rejected(self):
        with self.assertRaises(ValueError): c.validate_epoch_trace(trace(0)[:-1], splits(), 0, list(range(17)))

    def test_unlabeled_hidden_label_metadata_rejected(self):
        sample = sample_boundary(object(), object(), 'U000', 0, 'train_unlab')
        sample['label'] = object()
        with self.assertRaises(ValueError): c.validate_loaded_case(sample, 'U000', 0, False)

    def test_unlabeled_boundary_never_returns_label_object(self):
        label = object(); sample = sample_boundary(object(), label, 'U000', 0, 'train_unlab')
        self.assertNotIn('label', sample)
        self.assertFalse(any(x is label for x in sample.values()))
        self.assertTrue(sample['gt_roi_used'])

    def test_labeled_boundary_preserves_label(self):
        label = object(); sample = sample_boundary(object(), label, 'L000', 0, 'train_lab')
        self.assertIs(sample['label'], label)

    def test_actual_loaded_id_checked_before_wrapper(self):
        sample = sample_boundary(object(), object(), 'U001', 1, 'train_unlab')
        with self.assertRaises(ValueError): c.validate_loaded_case(sample, 'U000', 0, False)

    def test_unlab_label_trace_rejected(self):
        r = trace(0); r[0]['unlabeled_label_returned'] = True
        with self.assertRaises(ValueError): c.validate_epoch_trace(r, splits(), 0, list(range(17)))

    def test_padding_trace_rejected(self):
        r = trace(0); r[-1]['padding_case_id'] = 'U001'
        with self.assertRaises(ValueError): c.validate_epoch_trace(r, splits(), 0, list(range(17)))

    def test_final_test_only(self):
        state = dict(phase='formal', completed_epochs=36, updates=2376, microbatches=11880,
                     saved_epoch=35, test_evaluations_started=35, strict_reload_verified=True,
                     final_path=str(c.STORE/'final.pth'), final_sha256='a'*64, reloaded_sha256='a'*64)
        c.validate_test_boundary(state)
        for key, value in (('phase', 'bridge'), ('completed_epochs', 35), ('updates', 2375),
                           ('saved_epoch', 34), ('test_evaluations_started', 1),
                           ('strict_reload_verified', False), ('reloaded_sha256', 'b'*64),
                           ('final_path', str(c.STORE/'best.pth'))):
            with self.assertRaises(ValueError): c.validate_test_boundary(dict(state, **{key: value}))

    def test_test_dataset_no_early_permit(self):
        with self.assertRaisesRegex(ValueError, 'No unique final test intent'):
            make_dataset(str(c.ROOT/'data/Public_433_Folds/fold_2'), ['T000'], 'test')

    def test_dataset_wrong_fold_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Only fixed fold2'):
            make_dataset(str(c.ROOT/'data/Public_433_Folds/fold_0'), ['U000'], 'train_unlab')

    def test_exclusive_test_claim_and_final_digest(self):
        # Synthetic bytes only: this CPU test runs no dataset/model/test inference.
        fixture_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=str(fixture_root), prefix='cpu_fixture_') as tmp:
            self.assertEqual(Path(tmp).resolve().parent, fixture_root)
            base = Path(tmp)/'output'; store = Path(tmp)/'weights'; output = base/'formal_r1'
            output.mkdir(parents=True); store.mkdir()
            (store/'final.pth').write_bytes(b'synthetic test fixture, not weights')
            digest = c.sha_file(store/'final.pth')
            state = dict(phase='formal', completed_epochs=36, updates=2376, microbatches=11880,
                         saved_epoch=35, test_evaluations_started=35, strict_reload_verified=True,
                         final_path=(store/'final.pth').as_posix(), final_sha256=digest, reloaded_sha256=digest)
            with patch.object(c, 'BASE', PurePosixPath(base.as_posix())), patch.object(c, 'STORE', PurePosixPath(store.as_posix())):
                marker = c.claim_final_test(output, state)
                self.assertTrue(marker.is_file())
                with self.assertRaises(FileExistsError): c.claim_final_test(output, state)
                (store/'final.pth').write_bytes(b'changed fixture')
                with self.assertRaisesRegex(ValueError, 'Final changed'): c.claim_final_test(output, state)

    def test_config_only_transport_diff_allowed(self):
        cfg = {'dataset': {'fold': 2, 'split_root': 'expes/splits/public/public_semi05_cv5',
                           'fold_cv_dir_template': 'data/Public_433_Folds/fold_{fold}'},
               'experiment': {'seed': 2027, 'gpu': 0, 'output_dir': 'new'},
               'semi': {'matrix_group': 'A1B', 'matrix_c': 'B', 'ucp_epochs': 36, 'ucp_accumulation': 5,
                        'paired_refresh_labels': False, 'paired_workers': 0, 'paired_checkpoint_dir': 'new',
                        'ucp_encoder_lr': 5e-8, 'ucp_decoder_lr': 2e-7, 'ucp_coarse_lr': 2e-7,
                        'fclr_head_lr': 1e-4, 'zea_lr': 1e-4, 'xsa_cross_gain': .5,
                        'cross_style_complement': 'two-donor-full-decoder'}}
        ancestor = copy.deepcopy(cfg); ancestor['dataset']['fold'] = 0
        ancestor['experiment']['output_dir'] = 'old'; ancestor['semi']['paired_checkpoint_dir'] = 'old'
        c.validate_scientific_config(cfg, ancestor)
        for key, value in (('ucp_encoder_lr', 1e-4), ('ucp_epochs', 20), ('matrix_c', 'V'),
                           ('paired_refresh_labels', True), ('new_unreviewed_weight', 5)):
            bad = copy.deepcopy(cfg); bad['semi'][key] = value
            with self.assertRaises(ValueError): c.validate_scientific_config(bad, ancestor)


if __name__ == '__main__':
    unittest.main(verbosity=2)
