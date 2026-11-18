"""CPU-only targeted regression tests; no Torch import, training or evaluation."""
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import recovery_support_20261019 as support
import train_public_sup_recovery_20261019 as runner
HERE=Path(__file__).parent
OLD=(HERE/'train_public_basic_sup_20261018.py').read_text()
NEW=(HERE/'train_public_sup_recovery_20261019.py').read_text()
def funcs(text):
    return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}
class RecoveryTests(unittest.TestCase):
    def test_unchanged_helpers(self):
        old,new=funcs(OLD),funcs(NEW)
        for k in old.keys()-{'validate_config','run'}:
            self.assertEqual(old[k],new[k],k)
    def test_unchanged_training_body_except_scale(self):
        a=OLD[OLD.index('            student.train()'):OLD.index('        if successful_updates != budget')]
        b=NEW[NEW.index('            student.train()'):NEW.index('        if successful_updates != budget')]
        self.assertEqual(a.replace('scaler.update()','scaler.update(new_scale=1024.0)'),b)
    def test_final_evaluation_unchanged(self):
        start='        # Sole test entry point:'; end='        result = '
        self.assertEqual(OLD[OLD.index(start):OLD.index(end)],NEW[NEW.index(start):NEW.index(end)])
    def test_optimizer_unchanged(self):
        start='        optimizer = torch.optim.AdamW(['; end='        step_calls = '
        self.assertEqual(OLD[OLD.index(start):OLD.index(end)],NEW[NEW.index(start):NEW.index(end)])
    def test_budget(self):
        self.assertIn('for epoch in range(43, 60):',NEW)
        self.assertEqual((60-43)*346,5882)
        self.assertEqual(43*346+5882,20760)
        self.assertIn('scaler.update(new_scale=1024.0)',NEW)
    def test_boundary_positive_and_reject_mutations(self):
        cp={'successful_updates':14878,'completed_epochs':43,'next_epoch':43,'test_evaluations':0,
            'config':{'a':1},'config_sha256':'abc','source_manifest':{'s':2}}
        support.validate_boundary(cp,{'a':1},{'s':2},'abc')
        for key in cp:
            bad=copy.deepcopy(cp); bad[key]=None
            with self.assertRaises(RuntimeError):
                support.validate_boundary(bad,{'a':1},{'s':2},'abc')
    def test_no_replay_or_test_in_restore(self):
        text=(HERE/'recovery_support_20261019.py').read_text()
        self.assertNotIn('optimizer.step(',text)
        self.assertNotIn('make_strict_dataset(',text)
        self.assertIn('continuation_replay_verified',text)
    def test_guard_core_unchanged(self):
        a=funcs((HERE/'public_basic_sup_guard_20261018.py').read_text())
        b=funcs((HERE/'public_recovery_guard_20261019.py').read_text())
        for k in a.keys()-{'targets','validate_target'}:
            self.assertEqual(a[k],b[k],k)
    def test_guard_target_reject(self):
        import public_recovery_guard_20261019 as g
        for label,path in [(5,g.STORE/'sup05/final.pth'),(100,g.STORE/'sup100/latest.pth'),
            (100,g.HPQ),(100,g.RECOVERY/'unexpected.pth')]:
            with self.assertRaises(RuntimeError):
                g.validate_target(path,{'label_percent':label,'artifact_dir':str(path.parent)})
        with patch.object(g,'no_symlink'):
            self.assertEqual(g.validate_target(g.RECOVERY/'latest.pth',{'label_percent':100,
                'artifact_dir':str(g.RECOVERY)}),g.RECOVERY/'latest.pth')
if __name__=='__main__':
    unittest.main(verbosity=2)
