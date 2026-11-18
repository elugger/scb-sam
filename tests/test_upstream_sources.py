import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('source_fetch', Path(__file__).resolve().parents[1] / 'scripts/upstream_sources.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SourceFetchTests(unittest.TestCase):
    def row(self):
        return dict(local_path='dependency/code.py', bytes=4, sha256=hashlib.sha256(b'pass').hexdigest(), repository='owner/repo', revision='a'*40, upstream_path='code.py')

    def test_verified_download_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            self.assertEqual(module.fetch_one(root, self.row(), lambda *a, **k: io.BytesIO(b'pass')), 'downloaded_verified')
            self.assertEqual(module.fetch_one(root, self.row(), lambda *a, **k: self.fail('unnecessary network')), 'already_verified')
            (root/'dependency/code.py').write_bytes(b'edit')
            with self.assertRaises(ValueError):
                module.fetch_one(root, self.row())
            self.assertEqual((root/'dependency/code.py').read_bytes(), b'edit')

    def test_corruption_and_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with self.assertRaises(ValueError):
                module.fetch_one(root, self.row(), lambda *a, **k: io.BytesIO(b'evil'))
            self.assertFalse((root/'dependency/code.py').exists())
            row = self.row()
            row['local_path'] = '../escape.py'
            with self.assertRaises(ValueError):
                module.fetch_one(root, row)

    def test_transformation_checks_both_original_and_final_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            row = self.row()
            original = b'pass\r\n'
            final = b'pass\n'
            row.update(bytes=len(final), sha256=hashlib.sha256(final).hexdigest(), upstream_bytes=len(original), upstream_sha256=hashlib.sha256(original).hexdigest(), transforms=[dict(operation='lf')])
            root = Path(folder).resolve()
            module.fetch_one(root, row, lambda *a, **k: io.BytesIO(original))
            self.assertEqual((root/'dependency/code.py').read_bytes(), final)
            row['local_path'] = 'other.py'
            with self.assertRaises(ValueError):
                module.fetch_one(root, row, lambda *a, **k: io.BytesIO(final))
            self.assertFalse((root/'other.py').exists())
