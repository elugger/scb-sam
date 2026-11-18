"""Pure-CPU acceptance regression tests; zero model/test inference."""
import ast
import copy
from pathlib import Path
import unittest
import verify_public_sup_recovery_final_20261019 as v

class VerifierTests(unittest.TestCase):
    def test_frozen_common_validators(self):
        here=Path(__file__).parent
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n,ast.FunctionDef)}
        old=functions(here/'inspect_public_basic_sup_20261018.py')
        new=functions(here/'verify_public_sup_recovery_final_20261019.py')
        for name in old.keys()-{'inspect_arm','process_snapshot'}:
            self.assertEqual(old[name],new[name],name)
    def test_metrics_source_order_and_aggregate(self):
        rows=[{'actual_case_id':'a','requested_case_id':'a','case_replaced':False,'final_sha256':'a'*64,
               'metrics':{'dice':.7,'jaccard':.5,'hd95':3.,'asd':1.}}]
        self.assertEqual(v.validate_case_metrics(rows,['a'],'a'*64), rows[0]['metrics'])
        for key,value in [('actual_case_id','b'),('case_replaced',True),('final_sha256','b'*64)]:
            wrong=copy.deepcopy(rows); wrong[0][key]=value
            with self.assertRaises(v.EvidenceError):
                v.validate_case_metrics(wrong,['a'],'a'*64)
    def test_reject_nonfinite(self):
        with self.assertRaises(v.EvidenceError):
            v.parse_json('{"dice":NaN}')
        with self.assertRaises(v.EvidenceError):
            v.close(.7,.8,'aggregate')
    def test_reject_discontinuous_steps(self):
        row={'successful_update':2,'epoch':0,'epoch_number':1,'epoch_position':0}
        with self.assertRaises(v.EvidenceError):
            v.validate_ledgers(['a'],[row],[])
    def test_recovery_gate_present(self):
        text=(Path(__file__).parent/'verify_public_sup_recovery_final_20261019.py').read_text()
        for required in ["steps[:14878] == old_steps[:14878]", "len(latest_saves) == 17",
                         "len(saves) == 18", "final['continuation_replay_verified'] is False",
                         "all(r['amp_scale'] == 1024.0", "sha_file(old_steps_path) == support.LEDGER_SHA"]:
            self.assertIn(required,text)
        self.assertEqual(v.ARMS,{'sup100_recovery_r1':(100,346)})
if __name__=='__main__':
    unittest.main(verbosity=2)
