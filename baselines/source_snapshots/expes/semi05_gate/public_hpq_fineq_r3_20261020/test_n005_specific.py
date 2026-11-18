"""Regressions specific to the five-case protocol, without test inference."""
import unittest
import ast
from pathlib import Path
from collections import Counter
import hpq_contract as c
from test_hpq_contract import splits, trace
import hpq_resources as resources


class FiveCaseProtocol(unittest.TestCase):
    def test_all_36_epochs_four_distinct_legal_replays(self):
        for epoch in range(36):
            tokens = c.epoch_tokens(epoch)
            self.assertEqual(len(tokens), 345)
            self.assertEqual(tokens[:341], [(epoch, 0, i) for i in range(341)])
            self.assertEqual(tokens[341:], [(epoch, 1, (4*epoch+j)%341) for j in range(4)])
            self.assertEqual(len(set(t[2] for t in tokens[341:])), 4)
            counts = Counter(t[2] for t in tokens)
            self.assertEqual(sorted(counts.values()), [1]*337+[2]*4)
            c.validate_epoch_trace(trace(epoch), splits(), epoch, list(range(5)))

    def test_never_allow_gpu1(self):
        for phase in ('bridge', 'formal'):
            with self.assertRaises(ValueError): c.validate_scope(2,2027,phase,1)

    def test_new_initializer_not_17_case(self):
        self.assertIn('/lab005/', str(c.INIT))
        self.assertEqual(c.INIT_SHA, '12a580d460406f52d526b3d73c17b3053d34bb7dc41d814b1daedcfddbe9153f')
        self.assertEqual((c.UPDATES, c.MICROBATCHES), (2484,12420))

    def test_old_17_case_counts_fail(self):
        bad = splits()
        bad['train_lab'] += bad['train_unlab'][:12]
        bad['train_unlab'] = bad['train_unlab'][12:]
        with self.assertRaises(ValueError): c.validate_splits(bad)

    def test_physical_lists_nested_in_accepted_five(self):
        import hpq_inputs as i
        root = Path(__file__).resolve().parent/'splits/fold2'
        actual = {k:(root/(k+'.txt')).read_text().split() for k in i.SPLIT_SHA}
        c.validate_splits(actual)
        self.assertEqual(actual['train_lab'], ['PUBLIC_NATIVE_CASE','PUBLIC_NATIVE_CASE',
            'PUBLIC_NATIVE_CASE','PUBLIC_NATIVE_CASE','PUBLIC_NATIVE_CASE'])
        for k,v in i.SPLIT_SHA.items(): self.assertEqual(c.sha_file(root/(k+'.txt')),v)

    def test_old_resource_reservations_preserved(self):
        import inspect
        text = inspect.getsource(resources.backend)
        self.assertIn('PUBLIC05_HPQ_EPOCHDIAG_20261019', text)
        self.assertIn('original_targets() +', text)
        self.assertIn('active_targets.json', text)
        self.assertNotIn('unlink(', text)
        self.assertIn('/MAIN_TABLE_FIXED_FINAL_20261019/hpq_ncase/', str(c.STORE))

    def test_no_scientific_override_in_count_adapter(self):
        tree=ast.parse((Path(__file__).resolve().parent/'hpq_trainer_adapter.py').read_text())
        for cls in (n for n in tree.body if isinstance(n,ast.ClassDef)):
            self.assertTrue(all(n.name=='__init__' for n in cls.body if isinstance(n,ast.FunctionDef)))


if __name__ == '__main__': unittest.main()
