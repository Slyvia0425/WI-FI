import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


class LitePackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / 'scripts/prepare_spatial_lite.py'
        spec = importlib.util.spec_from_file_location('lite_package_test', path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def archive(self, root, payload=b'model', extra=None):
        path = root / 'runtime.zip'
        spec = {'files': {'outputs/model.joblib': {'bytes': 5, 'sha256': hashlib.sha256(b'model').hexdigest()}}}
        with zipfile.ZipFile(path, 'w') as package:
            package.writestr('bundle.json', json.dumps(spec))
            package.writestr('outputs/model.joblib', payload)
            if extra:
                package.writestr(extra, b'other')
        return path

    def test_restore_checks_model_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.module.restore(self.archive(root), root / 'restored')
            self.assertEqual((root / 'restored/outputs/model.joblib').read_bytes(), b'model')

    def test_corrupt_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                self.module.restore(self.archive(root, b'wrong'), root / 'restored')

    def test_unlisted_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'Unexpected'):
                self.module.restore(self.archive(root, extra='../outside'), root / 'restored')
            self.assertFalse((root / 'outside').exists())
