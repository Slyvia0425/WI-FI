import contextlib
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib


class SpatialDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        script = Path(__file__).resolve().parents[2] / 'datasets/space_proximity_sensing/get_data.py'
        spec = importlib.util.spec_from_file_location('spatial_dataset_test', script)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.module.HERE = self.root
        self.module.DATA = self.root / 'data'
        self.archive = self.root / 'source.zip'
        self.payloads = [('single_object/a.npz', 'wifi space/data/a.npz', b'first'), ('multi_object/b.npz', 'wifi space/multi_object/data/b.npz', b'second')]
        with zipfile.ZipFile(self.archive, 'w') as archive:
            for _, original, content in self.payloads:
                archive.writestr(original, content)
            archive.writestr('another_module/private.dat', b'not ours')
        self.write_index(self.payloads)

    def tearDown(self):
        self.temp.cleanup()

    def write_index(self, files):
        output = io.StringIO(newline='')
        writer = csv.writer(output)
        writer.writerow(['dataset_path', 'archive_path', 'bytes', 'crc32'])
        for path, original, content in files:
            writer.writerow([path, original, len(content), f'{zlib.crc32(content):08x}'])
        index = gzip.compress(output.getvalue().encode(), mtime=0)
        (self.root / 'file_index.csv.gz').write_bytes(index)
        metadata = {'index_sha256': hashlib.sha256(index).hexdigest(), 'file_count': len(files), 'release': {'archive_bytes': self.archive.stat().st_size, 'archive_sha256': hashlib.sha256(self.archive.read_bytes()).hexdigest()}}
        (self.root / 'manifest.json').write_text(json.dumps(metadata), encoding='utf-8')

    def run_main(self, *args):
        with patch('sys.argv', ['get_data.py', *args]), contextlib.redirect_stdout(io.StringIO()):
            self.module.main()

    def test_only_own_data_is_extracted_and_verified(self):
        self.run_main('--archive', str(self.archive))
        self.run_main('--verify-only')
        for path, _, payload in self.payloads:
            self.assertEqual((self.root / 'data' / path).read_bytes(), payload)
        self.assertFalse((self.root / 'data/another_module').exists())

    def test_damaged_file_is_detected(self):
        self.run_main('--archive', str(self.archive))
        (self.root / 'data/single_object/a.npz').write_bytes(b'wrong')
        with self.assertRaisesRegex(RuntimeError, 'Missing or damaged'):
            self.run_main('--verify-only')

    def test_index_tampering_is_rejected(self):
        with (self.root / 'file_index.csv.gz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'index checksum'):
            self.run_main('--archive', str(self.archive))

    def test_cross_module_destination_is_rejected(self):
        self.write_index([('../breathing_detection/data/a.npz', self.payloads[0][1], b'first')])
        with self.assertRaisesRegex(RuntimeError, 'Invalid dataset path'):
            self.run_main('--archive', str(self.archive))
