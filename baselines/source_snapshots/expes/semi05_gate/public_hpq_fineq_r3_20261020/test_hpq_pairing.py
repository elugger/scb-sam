import unittest
from types import SimpleNamespace
from hpq_pairing import _iter_groups, validate_collated
from test_hpq_contract import splits
from strict_hpq_data import sample_boundary


class FakeTensor:
    def to(self, device):
        return self


def seed(*args):
    return repr(args)


def labs(count, steps, epoch, base, refresh):
    return [(epoch, 0, i) for i in reversed(range(count))]


def loader(dataset, tokens, base, stream, workers):
    for token in tokens:
        i = token[2]
        sample = sample_boundary(FakeTensor(), FakeTensor(), dataset.actual_case_ids[i], i,
                                 'train_lab' if stream == 'lab' else 'train_unlab')
        yield dict(sample, case_id=[sample['case_id']], requested_index=i, augmentation_seed=seed(base, stream, *token))


class PairingTest(unittest.TestCase):
    def engine(self):
        s = splits(); records = []
        t = SimpleNamespace(pair_seed=2027, refresh=False, workers=0, device='fake', norm_transform=None,
                            lab_set=SimpleNamespace(actual_case_ids=s['train_lab']),
                            unlab_set=SimpleNamespace(actual_case_ids=s['train_unlab']),
                            _prepare_batch=lambda b: (b['image'], b['label']))
        return SimpleNamespace(t=t, input_splits=s, normalize_batch=lambda x, _: x,
                               append=lambda name, row: records.append((name, row)), records=records)

    def test_full_epoch_groups_and_trace(self):
        e = self.engine()
        groups = list(_iter_groups(e, 11, True, (labs, loader, seed, lambda x: 'unused')))
        self.assertEqual(len(groups), 69)
        self.assertTrue(all(len(g) == 5 for g in groups))
        self.assertEqual(len(e.records), 346)
        self.assertEqual(e.records[-1][1]['padding_case_ids'], ['U044','U045','U046','U047'])
        self.assertTrue(all(set(item) == {'epoch', 'step', 'image_l', 'label_l', 'image_u'}
                            for group in groups for item in group))

    def test_epoch0_digest_uses_actual_images(self):
        e = self.engine(); seen = []
        def digest(x):
            seen.append(x); return 'measured'
        list(_iter_groups(e, 0, True, (labs, loader, seed, digest)))
        self.assertEqual(len(seen), 690)
        self.assertEqual(e.records[0][1]['label_image_sha'], 'measured')

    def test_partial_bridge_read_has_no_complete_epoch_receipt(self):
        e = self.engine(); it = _iter_groups(e, 0, False, (labs, loader, seed, lambda x: 'unused'))
        next(it); next(it); it.close()
        self.assertEqual(e.records, [])

    def test_resampled_actual_id_rejected(self):
        e = self.engine()
        def bad_loader(*args):
            for b in loader(*args):
                if args[3] == 'unlab': b['case_id'] = ['U001']
                yield b
        with self.assertRaisesRegex(ValueError, 'Actual returned case'):
            next(_iter_groups(e, 0, False, (labs, bad_loader, seed, lambda x: 'unused')))

    def test_hidden_label_rejected_before_model(self):
        e = self.engine()
        def bad_loader(*args):
            for b in loader(*args):
                if args[3] == 'unlab': b['label'] = FakeTensor()
                yield b
        with self.assertRaisesRegex(ValueError, 'Unexpected collated fields'):
            next(_iter_groups(e, 0, False, (labs, bad_loader, seed, lambda x: 'unused')))

    def test_missing_microbatch_fails_full_epoch(self):
        e = self.engine()
        def short_loader(dataset, tokens, *args):
            yield from loader(dataset, tokens[:-1] if len(tokens) == 345 else tokens, *args)
        with self.assertRaises(ValueError):
            list(_iter_groups(e, 0, False, (labs, short_loader, seed, lambda x: 'unused')))

    def test_extra_microbatch_rejected(self):
        e = self.engine()
        def long_loader(dataset, tokens, *args):
            yield from loader(dataset, tokens + tokens[-1:] if len(tokens) == 345 else tokens, *args)
        with self.assertRaisesRegex(ValueError, 'Unexpected extra'):
            list(_iter_groups(e, 0, False, (labs, long_loader, seed, lambda x: 'unused')))

    def test_wrong_dataset_order_rejected(self):
        e = self.engine(); e.t.unlab_set.actual_case_ids = list(reversed(e.input_splits['train_unlab']))
        with self.assertRaisesRegex(ValueError, 'Dataset order'):
            next(_iter_groups(e, 0, False, (labs, loader, seed, lambda x: 'unused')))

    def test_label_refresh_rule_cannot_drift(self):
        e = self.engine(); e.t.refresh = True
        with self.assertRaises(ValueError):
            next(_iter_groups(e, 0, False, (labs, loader, seed, lambda x: 'unused')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
