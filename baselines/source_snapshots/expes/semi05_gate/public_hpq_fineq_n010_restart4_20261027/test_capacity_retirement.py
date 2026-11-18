import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import capacity_retirement as c

class RetirementTests(unittest.TestCase):
    def setUp(self):
        self.expected=c.expected_paths()
        self.proof=dict(status='AUTHORIZED_EMPTY_10G_RESERVATIONS_RETIRED',files_deleted=0,
            other_rows_exactly_preserved=True,retired_reserved_bytes=10*1024**3,
            removed=[dict(path=p,bound=2*1024**3) for p in self.expected],terminal_evidence={},
            before_guard_targets=[dict(path='/history',bound=20),dict(path='/keep',bound=10)])
    def apply(self,registered=(),proof=None,occupied=False):
        with patch.object(c,'bound_small_bytes',return_value=json.dumps(self.proof if proof is None else proof).encode()),patch.object(c.os.path,'lexists',return_value=occupied),patch.object(Path,'is_symlink',return_value=False):
            return {str(p):b for p,b in c.apply_retirement([(Path('/keep'),30),(Path('/new'),40)]+[(Path(p),2*1024**3) for p in self.expected],set(registered))}
    def test_preserve_all_history_and_new_targets(self):self.assertEqual(self.apply(),{'/keep':30,'/new':40,'/history':20})
    def test_wrong_set_rejected(self):
        p=copy.deepcopy(self.proof);p['removed'].pop()
        with self.assertRaises(ValueError):self.apply(proof=p)
    def test_wrong_bound_rejected(self):
        p=copy.deepcopy(self.proof);p['removed'][0]['bound']=1
        with self.assertRaises(ValueError):self.apply(proof=p)
    def test_occupied_rejected(self):
        with self.assertRaises(ValueError):self.apply(occupied=True)
    def test_registered_rejected(self):
        with self.assertRaises(ValueError):self.apply(registered=[next(iter(self.expected))])
if __name__=='__main__':unittest.main()
