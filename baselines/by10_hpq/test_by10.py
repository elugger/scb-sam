"""Budget, split selection, immutable ancestor, and scientific update gates."""
import copy,unittest
from by_protocol import base_config,validate_supervised_config,targets,GIB
from four_contract import group_config

class ContractTests(unittest.TestCase):
    def config(self):
        c=base_config();c['source_sha256']={'test':'a'*64};return c
    def test_budget(self):
        c=self.config();self.assertEqual(validate_supervised_config(c)['total_updates'],600)
        self.assertEqual(c['train_count'],10);self.assertIn('by_semi10',c['train_list'])
        self.assertEqual(38*90,3420);self.assertEqual(38*90//5,684)
    def test_reject_five_case_initializer(self):
        c=self.config();c['train_count']=5
        with self.assertRaises(ValueError):validate_supervised_config(c)
    def test_reject_wrong_epoch(self):
        c=self.config();c['epochs']=30
        with self.assertRaises(ValueError):validate_supervised_config(c)
    def test_only_authorized_targets(self):
        rows=targets();self.assertEqual(len(rows),4)
        self.assertEqual(sum(b for p,b in rows),3*GIB+128*1024**2)
        self.assertTrue(all('by10_hpq_20261107' in p for p,b in rows))
    def test_config_changes_are_explicit(self):
        a={'experiment':{'seed':2027},'dataset':{'split_root':'old'},'semi':{'ucp_epochs':36,'ucp_accumulation':5,'ucp_ramp_epochs':12,'matrix_c':'B'}}
        before=copy.deepcopy(a);c=group_config(a,'HPQ','formal')
        self.assertEqual(a,before);self.assertEqual(c['semi']['ucp_epochs'],38)
        self.assertEqual(c['semi']['ucp_ramp_epochs'],12);self.assertEqual(c['semi']['matrix_c'],'B')

if __name__=='__main__':unittest.main()
