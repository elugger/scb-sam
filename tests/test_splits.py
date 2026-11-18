import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify_splits', ROOT / 'scripts/verify_splits.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SplitIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.value = json.loads((ROOT / 'splits/by/fold2_n005.json').read_text())

    def test_all_current_splits_are_bound_without_claiming_image_identity(self):
        result = module.verify()
        self.assertEqual(len(result['splits']), 4)
        self.assertFalse(result['native_data_binding_verified'])

    def test_overlap_duplicate_and_unknown_alias_are_rejected(self):
        for change in ('overlap', 'duplicate', 'unsafe'):
            value = copy.deepcopy(self.value)
            if change == 'overlap':
                value['test'][0] = value['train'][0]
            elif change == 'duplicate':
                value['labeled'][1] = value['labeled'][0]
            else:
                value['test'][0] = '../../unlisted_case'
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.validate_split(value)


if __name__ == '__main__':
    unittest.main()
