import ast
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import by_protocol as p
import by_inputs
import by_resources
import by_runtime
import train_by5
HERE = Path(__file__).resolve().parent

@contextmanager
def synthetic_directory():
    owner = HERE.parent.resolve()
    temporary = tempfile.TemporaryDirectory(prefix='tmp_test_by5_', dir=str(owner))
    resolved = Path(temporary.name).resolve()
    if resolved.parent != owner or not resolved.name.startswith('tmp_test_by5_'):
        raise ValueError('Unsafe synthetic-test cleanup path')
    try:
        yield str(resolved)
    finally:
        temporary.cleanup()

def config():
    cfg = p.base_config()
    cfg['source_sha256'] = {'source.py': 'a' * 64}
    return cfg

class ContractTests(unittest.TestCase):

    def test_exact_budget(self):
        self.assertEqual(p.validate_supervised_config(config()), train_by5.training_budget(5, 60))
        self.assertEqual(p.validate_supervised_config(config())['total_updates'], 300)

    def test_reject_changed_scientific_or_execution_conditions(self):
        changes = dict(labeled_cases=6, train_count=6, test_count=24, fold=1, seed=2026, epochs=59, batch_size=2, mode='semi', use_hpq=True, use_unlabeled=True, resume=True, amp_init_scale=512.0, checkpoint_policy='best', lr=0.0002, lr_decoder_scale=0.2, lr_coarse_scale=1.0, weight_decay=0.01, aux_weight=0.1, model_type='vit_l', img_size=64, deep_supervision=True, project_root='/other', output_dir='/other', artifact_dir='/other', sam_checkpoint='/old/fold2.pth', sam_checkpoint_sha256='b' * 64, train_list='/other', resource_guard_module='other', init_from='old.pth', resume_from='old.pth', label_percent=5, train_count_type=True)
        for key, value in changes.items():
            with self.subTest(key=key):
                cfg = config()
                cfg['train_count' if key == 'train_count_type' else key] = value
                with self.assertRaises(ValueError):
                    p.validate_supervised_config(cfg)

    def test_reject_unbound_sources_and_changed_split(self):
        for field in ('source_sha256', 'input_sha256'):
            cfg = config()
            cfg[field] = {}
            with self.assertRaises(ValueError):
                p.validate_supervised_config(cfg)
        cfg = config()
        cfg['input_sha256'][str(p.SPLIT / 'train_lab.txt')] = 'b' * 64
        with self.assertRaises(ValueError):
            p.validate_supervised_config(cfg)

    def test_reservations_keep_old_targets(self):
        old = [('/historic/final.pth', 123), ('/historic2/final.pth', 456)]
        merged = dict(((str(path), bound) for path, bound in by_resources.merge(old, p.targets())))
        self.assertEqual({k: merged[str(Path(k))] for k, _ in old}, dict(old))
        self.assertEqual(sum((b for _, b in p.targets())), 9 * p.GIB + 320 * 1024 ** 2)
        self.assertEqual(len(p.targets()), 10)
        self.assertEqual(len({path for path, _ in p.targets()}), 10)
        for path, bound in p.targets():
            self.assertEqual(Path(path).name, 'final.pth')
            self.assertLessEqual(bound, 2 * p.GIB)
        with self.assertRaises(ValueError):
            by_resources.merge(p.targets(), [(p.targets()[0][0], 123)])
        with self.assertRaises(ValueError):
            by_resources.merge(old + old, [])

    def test_save_rejects_incomplete_budget_before_infra_or_serialization(self):
        good = dict(completed_epochs=60, epoch=59, successful_updates=300, test_evaluations_before_save=0)
        with patch.object(by_resources, 'infra', side_effect=AssertionError('must reject before infrastructure')):
            for key, value in (('successful_updates', 299), ('completed_epochs', 59), ('epoch', 58), ('test_evaluations_before_save', 1), ('optimizer', {})):
                payload = dict(good)
                payload[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    by_resources.guarded_torch_save(payload=payload, target=str(p.STORE / 'supervised/final.pth'), config=config(), save_kind='final')
            with self.assertRaises(ValueError):
                by_resources.guarded_torch_save(payload=good, target='/other/final.pth', config=config(), save_kind='final')

    def test_bundle_detects_mutation_and_traversal(self):
        with synthetic_directory() as folder:
            root = Path(folder)
            item = root / 'test.py'
            item.write_text('x = 1\n')
            manifest = root / 'source_manifest.json'
            manifest.write_text(json.dumps({'files': {'test.py': p.sha(item)}}))
            digest = p.sha(manifest)
            self.assertEqual(len(p.verify_bundle(digest, root)), 1)
            item.write_text('x = 2\n')
            with self.assertRaises(ValueError):
                p.verify_bundle(digest, root)
            manifest.write_text(json.dumps({'files': {'../other.py': 'a' * 64}}))
            with self.assertRaises(ValueError):
                p.verify_bundle(p.sha(manifest), root)
            manifest.write_text(json.dumps({'files': {}}))
            with self.assertRaises(ValueError):
                p.verify_bundle(p.sha(manifest), root)

class InputTests(unittest.TestCase):

    def test_reject_unsafe_or_duplicate_lists(self):
        with synthetic_directory() as folder:
            path = Path(folder) / 'cases.txt'
            for text in ('', '../case\n', 'case.nii.gz\n', 'a\na\n', 'C:case\n', 'a/b\n', 'a\\b\n'):
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        train_by5.read_case_ids(path)

    def test_disjoint_exact_case_counts(self):
        train = [str(i) for i in range(5)]
        test = [str(i) for i in range(5, 30)]
        self.assertTrue(train_by5.validate_case_sets(train, test, 5, 25)['case_sets_disjoint'])
        for t, d in ((train[:-1], test), (train, test[:-1]), (train, ['0'] + test[1:]), (['0'] * 5, test)):
            with self.assertRaises(ValueError):
                train_by5.validate_case_sets(t, d, 5, 25)

    def test_case_identity_detects_change_and_escape(self):
        with synthetic_directory() as folder:
            root = Path(folder)
            data = root / 'data'
            data.mkdir()
            case = data / 'case.nii.gz'
            case.write_bytes(b'input')
            old = by_inputs.identity(case, data)
            case.write_bytes(b'changed-input')
            self.assertNotEqual(old, by_inputs.identity(case, data))
            outside = root / 'outside.nii.gz'
            outside.write_bytes(b'input')
            with self.assertRaises(ValueError):
                by_inputs.identity(outside, data)

class GpuTests(unittest.TestCase):

    def test_only_idle_expected_uuid_allowed(self):
        idle = dict(uuid=p.GPU_UUID, index=0, process_counts=dict(compute=0, graphics=0), graphics=[], utilization_percent=0, memory_bytes=0)
        self.assertTrue(by_runtime.is_idle(idle))
        for key, value in (('uuid', 'GPU-other'), ('index', 1), ('utilization_percent', 1), ('memory_bytes', 512 * 1024 ** 2), ('process_counts', dict(compute=1, graphics=0))):
            row = copy.deepcopy(idle)
            row[key] = value
            with self.subTest(key=key):
                self.assertFalse(by_runtime.is_idle(row))
        row = copy.deepcopy(idle)
        row['process_counts']['graphics'] = 1
        row['graphics'] = [dict(name='Xorg', memory_bytes=20 * 1024 ** 2)]
        self.assertTrue(by_runtime.is_idle(row))
        row['graphics'][0]['name'] = 'other'
        self.assertFalse(by_runtime.is_idle(row))
        row['graphics'] = []
        self.assertFalse(by_runtime.is_idle(row))

class ComputationParityTests(unittest.TestCase):

    def test_scientific_functions_unchanged_from_pinned_supervisor(self):
        remote = Path(p.ROOT / 'expes/semi05_gate/public_ncase_20261019/train_ncase.py')
        reference = remote if remote.is_file() else HERE.parents[1] / 'public_ncase_20261019/train_ncase.py'
        original = ast.parse(reference.read_text(encoding='utf-8'))
        derived_text = (HERE / 'train_by5.py').read_text(encoding='utf-8')
        derived_text = derived_text.replace('    from by_runtime import ensure_gpu0_idle\n    ensure_gpu0_idle()\n', '')
        derived_text = derived_text.replace('        from by_runtime import own_gpu0_fds\n        atomic_json(output_dir / "own_gpu_fds.json", own_gpu0_fds())\n', '')
        derived_text = derived_text.replace('from by_inputs import check_case', 'from ncase_inputs import check_case')
        derived_text = derived_text.replace('by5_shared_sup_fixed_final_student_v1', 'public_basic_sup_fixed_final_student_v1')
        derived_text = derived_text.replace('cohort_checkpoint_loaded', 'public_checkpoint_loaded')
        derived = ast.parse(derived_text)
        a = {n.name: ast.dump(n, include_attributes=False) for n in original.body if isinstance(n, ast.FunctionDef)}
        b = {n.name: ast.dump(n, include_attributes=False) for n in derived.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(set(a), set(b))
        for name in set(a) - {'validate_config', 'main'}:
            with self.subTest(function=name):
                self.assertEqual(a[name], b[name])

    def test_all_new_python_files_parse_without_import_side_effects(self):
        for path in HERE.glob('*.py'):
            with self.subTest(file=path.name):
                ast.parse(path.read_text(encoding='utf-8'))
if __name__ == '__main__':
    unittest.main()
