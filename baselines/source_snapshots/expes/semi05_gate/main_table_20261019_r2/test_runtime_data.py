"""Case mapping tests use tiny files, never training/test inference."""
from pathlib import Path
import tempfile
import unittest
from runtime_data import case_file_identity


class CaseMappingTests(unittest.TestCase):
    def test_named_regular_file_is_bound_and_change_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); p=root/'case.nii.gz'; p.write_bytes(b'one')
            first=case_file_identity(p,root)
            p.write_bytes(b'changed')
            self.assertNotEqual(first,case_file_identity(p,root))

    def test_named_symlink_within_data_root_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); raw=root/'raw'; raw.mkdir()
            target=raw/'case.nii.gz'; target.write_bytes(b'image')
            p=root/'case.nii.gz'
            try: p.symlink_to(target)
            except OSError: self.skipTest('Host lacks symlink privilege')
            result=case_file_identity(p,root)
            self.assertTrue(result['symlink'])
            self.assertEqual(result['resolved'],str(target.resolve()))

    def test_wrong_case_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); target=root/'other.nii.gz'; target.write_bytes(b'image')
            p=root/'case.nii.gz'
            try: p.symlink_to(target)
            except OSError: self.skipTest('Host lacks symlink privilege')
            with self.assertRaises(ValueError): case_file_identity(p,root)

    def test_outside_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); allowed=root/'data'; allowed.mkdir()
            p=root/'case.nii.gz'; p.write_bytes(b'image')
            with self.assertRaises(ValueError): case_file_identity(p,allowed)


if __name__=='__main__': unittest.main()
