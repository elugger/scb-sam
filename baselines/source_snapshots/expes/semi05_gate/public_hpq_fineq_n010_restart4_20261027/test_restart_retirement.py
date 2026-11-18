"""Successor transport isolation and conservative retirement regression tests."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import hpq_contract as c
import hpq_preparation as prep
import launch_hpq as launch
import reservation_history as history

class RestartTests(unittest.TestCase):
    def setUp(self):
        self.expected = {'/nonexistent/successor-test/retired%d/final.pth' % i for i in range(8)}
        self.proof = dict(status='AUTHORIZED_DELETED_GPU0_SAFETY_RESERVATIONS_RETIRED',
            files_deleted=0, other_rows_exactly_preserved=True, retired_reserved_bytes=16*1024**3,
            removed=[dict(path=p) for p in self.expected],
            before_guard_targets=[dict(path='/keep',bound=12),dict(path='/other_history',bound=20)] +
                                 [dict(path=p,bound=2*1024**3) for p in self.expected])
        self.deleted = dict(status='AUTHORIZED_THREE_CATEGORIES_DELETED_VERIFIED',
            deleted=[dict(path=p,group='weights') for p in self.expected])

    def apply(self, registered=(), proof=None):
        with patch.object(history.os.path, 'lexists', return_value=False), patch.object(Path, 'is_symlink', return_value=False):
            return dict((str(p),b) for p,b in history.apply_deleted_retirement(
                {'/keep':30,'/new_run':40}, set(registered), set(), self.expected,
                self.proof if proof is None else proof, self.deleted))

    def test_full_union_preserved(self):
        self.assertEqual(self.apply(), {'/keep':30,'/new_run':40,'/other_history':20})

    def test_registered_retirement_rejected(self):
        with self.assertRaises(ValueError): self.apply([next(iter(self.expected))])

    def test_wrong_retirement_set_rejected(self):
        p=copy.deepcopy(self.proof);p['removed'].pop()
        with self.assertRaises(ValueError): self.apply(proof=p)

    def test_changed_bound_rejected(self):
        p=copy.deepcopy(self.proof);p['retired_reserved_bytes']+=1
        with self.assertRaises(ValueError): self.apply(proof=p)

    def test_occupied_retirement_rejected(self):
        with patch.object(history.os.path,'lexists',return_value=True):
            with self.assertRaises(ValueError):
                history.apply_deleted_retirement({},set(),set(),self.expected,self.proof,self.deleted)

    def test_independent_phase_paths(self):
        self.assertIn('RESTART4_20261027',str(c.BASE))
        self.assertEqual(c.STORE.name,'r2')
        self.assertEqual(launch.paths('bridge')['store'].name,'bridge_restart4_20261027')
        self.assertIn('restart4_20261027',str(prep.SRC))
        self.assertNotEqual(launch.paths('formal')['store'],launch.paths('bridge')['store'])

    def test_science_and_selection_unchanged(self):
        here=Path(__file__).parent
        self.assertEqual(c.sha_file(here/'fine_q.py'),'fb78c11ea08e2d5ffd2e85c61a8e90b2c2b0e3db8248b139d10c62119919b308')
        self.assertEqual(c.sha_file(here/'selection_protocol.py'),'098cdb19985d9e7172106671aa66aefcfd8a7d6d6f46c5de9e5b41c916234f75')

    def test_constructor_phase_transport(self):
        from hpq_initialization import AcceptedSupervisedMixin
        import hpq_inputs
        import hpq_initialization
        from test_hpq_contract import splits
        class ReachedModel(Exception): pass
        class FakeModel:
            def __init__(self,cfg): raise ReachedModel()
        class Candidate(AcceptedSupervisedMixin,FakeModel): pass
        for phase in ('bridge','formal'):
            cfg={'dataset':{},'experiment':{},'semi':{}}
            with patch.object(prep,'validate_scientific_config'), patch.object(hpq_initialization,'validate_scientific_config'), patch.object(hpq_inputs,'read_bound_splits',return_value=splits()), patch.object(hpq_inputs,'read_accepted_n010'):
                cfg=prep.phase_config(phase,0,cfg)
                self.assertEqual(cfg['semi']['paired_checkpoint_dir'],str(launch.paths(phase)['store']))
                with self.assertRaises(ReachedModel): Candidate(cfg,splits=splits(),ancestor={},phase=phase)

    def test_constructor_rejects_foreign_bridge(self):
        from hpq_initialization import AcceptedSupervisedMixin
        import hpq_inputs
        import hpq_initialization
        from test_hpq_contract import splits
        cfg={'experiment':{'output_dir':str(c.BASE/'bridge_r1')},'semi':{'paired_checkpoint_dir':str(c.STORE.parent/'bridge_r1')}}
        with patch.object(hpq_initialization,'validate_scientific_config'), patch.object(hpq_inputs,'read_bound_splits',return_value=splits()), patch.object(hpq_inputs,'read_accepted_n010'):
            with self.assertRaisesRegex(ValueError,'Foreign checkpoint path'):
                AcceptedSupervisedMixin(cfg,splits=splits(),ancestor={},phase='bridge')

if __name__ == '__main__': unittest.main()
