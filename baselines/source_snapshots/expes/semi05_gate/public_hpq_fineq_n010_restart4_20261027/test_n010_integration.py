"""Ten-case transport, initializer, science and budget regression tests."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import unittest
from unittest.mock import patch
import hpq_contract as c
import hpq_inputs as inputs
import hpq_preparation as prep
import hpq_resources as resources
import selection_protocol as selection
from test_hpq_contract import splits, trace

class TenIntegrationTests(unittest.TestCase):
    def test_science_bytes_unchanged(self):
        self.assertEqual(c.sha_file(Path(__file__).with_name('fine_q.py')), selection.FINE_Q_SHA256)

    def test_accepted_ten_initializer(self):
        path=Path(__file__).with_name('initializer_acceptance.json')
        self.assertEqual(c.sha_file(path), inputs.ACCEPTANCE_SHA)
        receipt=json.loads(path.read_text(encoding='utf-8'))
        c.validate_initialization(c.INIT,c.INIT_SHA,receipt)
        self.assertEqual((receipt['labeled_cases'],receipt['successful_updates'],receipt['completed_epochs']), (10,600,60))
        self.assertIn('/basic_cascade_ncase/Public433/lab010/',str(c.INIT))

    def test_foreign_initializers_rejected(self):
        receipt=json.loads(Path(__file__).with_name('initializer_acceptance.json').read_text(encoding='utf-8'))
        for count in ('lab005','lab035','pct010'):
            with self.assertRaises(ValueError):c.validate_initialization(str(c.INIT).replace('lab010',count),c.INIT_SHA,receipt)

    def test_ten_split_pins(self):
        self.assertEqual(inputs.SPLIT_SHA['train_lab'],selection.SPLIT_PINS['n010_lab.txt'])
        self.assertEqual(inputs.SPLIT_SHA['train_unlab'],selection.SPLIT_PINS['n010_unlab.txt'])
        self.assertEqual(str(inputs.SPLIT_ROOT),'__RECORDED_PROJECT_ROOT__/expes/semi05_gate/public_ncase_20261019/splits')

    def test_all_cycles_each_label34(self):
        for epoch in range(36):
            rows=trace(epoch)
            c.validate_epoch_trace(rows,splits(),epoch,list(range(10)))
            self.assertEqual(set(Counter(row['labeled_case_id'] for row in rows).values()),{34})
            self.assertEqual(len(rows),340)

    def test_full_budget_matches_selection(self):
        self.assertEqual((c.UPDATES_PER_EPOCH,c.UPDATES,c.MICROBATCHES),(68,2448,12240))
        self.assertEqual(selection.budget(36)['successful_updates'],c.UPDATES)
        self.assertEqual(selection.POLICY['labeled_cases'],10)
        self.assertEqual(selection.POLICY['min_delta_pp'],'0.05')

    def test_old_label_count_rejected(self):
        value=splits();value['train_lab']+=['EXTRA%02d'%i for i in range(25)]
        with self.assertRaises(ValueError):c.validate_splits(value)

    def test_original_storage_bounds_unchanged(self):
        owned=resources.owned_targets()
        self.assertEqual(len(owned),6)
        self.assertEqual(sum(b for _,b in owned),10*1024**3+512*1024**2)
        self.assertTrue(all('/hpq_ncase/Public433/lab010/' in p.as_posix() for p,_ in owned))
        self.assertEqual(resources.RAM_REQUIRED,12*1024**3+16*1024**2)

    def test_no_mixed_ten_percent_paths(self):
        self.assertIn('PUBLIC_N010_FINEQ_BEST',str(c.BASE))
        self.assertIn('public_hpq_fineq_n010',str(prep.SRC))
        self.assertNotIn('hpq_percent',str(c.STORE))

    def test_policy_hash_is_deterministic(self):
        raw=json.dumps(selection.POLICY,sort_keys=True,separators=(',',':')).encode()
        self.assertEqual(selection.policy_sha(),hashlib.sha256(raw).hexdigest())

if __name__=='__main__':unittest.main()
