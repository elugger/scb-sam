import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('checkpoint_assets', Path(__file__).resolve().parents[1] / 'scripts/checkpoint_assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)

class Response(io.BytesIO):
    def geturl(self):
        return 'https://huggingface.co/verified'

class AssetTests(unittest.TestCase):
    def setUp(self):
        self.data = b'exact-checkpoint-fixture'
        self.row = {'name': 'model.pth', 'original_bytes': len(self.data), 'expected_original_sha256': hashlib.sha256(self.data).hexdigest(), 'download_url': 'https://huggingface.co/model'}
        self.tmp = tempfile.TemporaryDirectory(dir=assets.ROOT.parent)
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def opener(self, url, timeout):
        return Response(self.data)

    def test_verified_download_and_idempotent_existing_asset(self):
        self.assertEqual(assets.download(self.row, self.root, self.opener)['status'], 'VERIFIED')
        self.assertEqual((self.root / 'model.pth').read_bytes(), self.data)
        self.assertEqual(assets.download(self.row, self.root, lambda *a, **k: self.fail('existing asset fetched again'))['status'], 'VERIFIED')

    def test_corrupted_payload_is_never_published(self):
        for payload in (self.data[:-1], b'x' * len(self.data), self.data + b'x'):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    assets.download(self.row, self.root, lambda *a, **k: Response(payload))
                self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_wrong_asset_is_preserved(self):
        target = self.root / 'model.pth'
        target.write_bytes(b'user-file')
        with self.assertRaises(ValueError):
            assets.download(self.row, self.root, self.opener)
        self.assertEqual(target.read_bytes(), b'user-file')

    def test_unpublished_weights_have_no_fabricated_download(self):
        self.row['download_url'] = None
        with self.assertRaisesRegex(ValueError, 'NO_VERIFIED_DOWNLOAD_URL'):
            assets.download(self.row, self.root, self.opener)

    def test_origin_and_path_validation(self):
        for key, value in [('name', '../outside.pth'), ('download_url', 'http://huggingface.co/model'), ('download_url', 'https://example.com/model')]:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    assets.download(dict(self.row, **{key: value}), self.root, self.opener)
        with self.assertRaisesRegex(ValueError, 'OUTSIDE_CODE_REPOSITORY'):
            assets.download(self.row, assets.ROOT / 'weights', self.opener)

if __name__ == '__main__':
    unittest.main()
