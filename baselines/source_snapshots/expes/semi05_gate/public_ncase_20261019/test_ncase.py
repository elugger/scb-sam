"""Targeted CPU checks; these are not real CUDA qualification or completion evidence."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ncase_protocol as p
import ncase_inputs as data
import ncase_resources as resources
import train_ncase as runner
import inspect_ncase as monitor

HERE=Path(__file__).resolve().parent


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg=p.make_config(HERE,5)
    def test_nested_selection(self):
        sets=p.validate_splits(HERE/'splits')
        self.assertLess(set(sets[5]['lab']),set(sets[10]['lab']))
        self.assertEqual((len(sets[5]['unlab']),len(sets[10]['unlab'])),(341,336))
        self.assertFalse(set(sets[5]['lab']+sets[5]['unlab']) & set(sets[5]['test']))
    def test_rank_manifest_exact(self):
        selection=json.loads((HERE/'selection.json').read_text())
        for row in selection['ranking']:
            self.assertEqual(row['rank'],hashlib.sha256((p.RANK_PREFIX+row['id']).encode()).hexdigest())
        self.assertEqual([r['id'] for r in selection['ranking'][:5]],p.ids(HERE/'splits/n005_lab.txt'))
    def test_supervised_budget(self):
        self.assertEqual(runner.validate_config(self.cfg)['total_updates'],300)
        self.assertEqual(runner.validate_config(p.make_config(HERE,10))['total_updates'],600)
    def test_hpq_budget_complete_unlab(self):
        b=p.hpq_budget(5)
        self.assertEqual((b['total_updates'],b['total_microbatches'],b['legal_unlabeled_replays_per_epoch']),(2484,12420,4))
        self.assertEqual(p.hpq_budget(10)['total_updates'],2448)
    def test_reject_percent_label(self):
        self.cfg['label_percent']=5
        with self.assertRaises(ValueError): runner.validate_config(self.cfg)
    def test_reject_seventeen_case_weight_and_count(self):
        self.cfg['train_count']=17
        with self.assertRaises(ValueError): runner.validate_config(self.cfg)
        self.cfg=p.make_config(HERE,5); self.cfg['init_from']='old_sup05/final.pth'
        with self.assertRaises(ValueError): runner.validate_config(self.cfg)
    def test_reject_wrong_paths_and_policies(self):
        for key,value in [('artifact_dir','/tmp/final'),('output_dir','/tmp/run'),('amp_policy','growth'),
                          ('epochs',61),('test_policy','best'),('checkpoint_policy','latest')]:
            cfg=copy.deepcopy(self.cfg);cfg[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): runner.validate_config(cfg)
    def test_reject_overlap_and_duplicates(self):
        with self.assertRaises(ValueError): runner.validate_case_sets(['a'],['a'],1,1)
        with self.assertRaises(ValueError): runner.validate_case_sets(['a','a'],['b'],2,1)
    def test_source_has_single_test_after_reload(self):
        source=(HERE/'train_ncase.py').read_text()
        self.assertLess(source.index('student.load_state_dict(reloaded["student"], strict=True)'),source.index('evaluation_marker ='))
        self.assertEqual(source.count('test_dataset = make_strict_dataset'),1)
        self.assertNotIn('latest_payload',source)
        self.assertIn('scaler.update(new_scale=1024.0)',source)
    def test_all_sources_parse(self):
        for source in HERE.glob('*.py'):
            ast.parse(source.read_text(),filename=source.name)
    def test_guard_rejects_early_final(self):
        with self.assertRaises(ValueError):
            resources.guarded_torch_save(payload=dict(completed_epochs=1),target=resources.target_for(self.cfg),config=self.cfg,save_kind='final')
    def test_guard_rejects_other_checkpoint(self):
        with self.assertRaises(ValueError):
            resources.guarded_torch_save(payload={},target='/tmp/final.pth',config=self.cfg,save_kind='final')
    def test_identity_rejects_outside_root(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp)/'data'; root.mkdir()
            outside=Path(tmp)/'PUBLIC_NATIVE_CASE.nii.gz';outside.write_bytes(b'a')
            with self.assertRaises(ValueError): data.identity(outside,root)
    def test_identity_detects_content_metadata_change(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path=Path(tmp)/'PUBLIC_NATIVE_CASE.nii.gz';path.write_bytes(b'a')
            before=data.identity(path,tmp);path.write_bytes(b'changed')
            self.assertNotEqual(before,data.identity(path,tmp))
    def test_launch_has_no_ten_case_or_gpu1_dispatch(self):
        source=(HERE/'launch_ncase.py').read_text()
        self.assertIn('make_config(HERE, 5)',source)
        self.assertIn("'--gpu','0'",source)
        self.assertNotIn("'--gpu','1'",source)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.cfg=p.make_config(HERE,5)
        self.lab=p.ids(HERE/'splits/n005_lab.txt')
        self.test=p.ids(HERE/'splits/test.txt')
        self.steps=[dict(successful_update=i+1,epoch=0,epoch_position=i,case_id=c,requested_index=i,
                         optimizer_steps=1,skipped=False,gradient_finite=True,amp_scale=1024,loss=.1)
                    for i,c in enumerate(self.lab)]
        self.epochs=[dict(epoch=0,epoch_number=1,successful_updates=5,actual_case_ids=self.lab,test_evaluations=0)]
    def validate(self):
        with patch.object(monitor,'ids',side_effect=[self.lab,self.test]):
            return monitor.verify_ledgers(self.cfg,self.steps,self.epochs)
    def test_complete_epoch(self):
        self.assertEqual(self.validate()['completed_epochs'],1)
    def test_inflight_epoch_publication(self):
        self.epochs=[]
        self.assertEqual(self.validate()['successful_updates'],5)
    def test_reject_duplicate_case(self):
        self.steps[-1].update(case_id=self.lab[0],requested_index=0)
        with self.assertRaises(ValueError): self.validate()
    def test_reject_skipped_or_nan(self):
        self.steps[-1]['skipped']=True
        with self.assertRaises(ValueError): self.validate()
        self.steps[-1]['skipped']=False;self.steps[-1]['loss']=float('nan')
        with self.assertRaises(ValueError): self.validate()
    def test_reject_wrong_epoch_case_trace(self):
        self.epochs[0]['actual_case_ids']=list(reversed(self.lab))
        with self.assertRaises(ValueError): self.validate()


if __name__=='__main__':
    unittest.main(verbosity=2)
