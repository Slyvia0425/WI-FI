"""Download only FallDetection from the official Kaggle v12 ZIP using HTTP ranges.

Checks ZIP CRC32 and uncompressed size for every file, including resumed files.
No complete 81 GB dataset download, credential, or archive copy is required.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib

import requests
from remotezip import RemoteZip

URL = 'https://www.kaggle.com/api/v1/datasets/download/guozhenjennzhu/csi-bench?datasetVersionNumber=12'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True, help='CSI-Bench-v12 parent directory')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with RemoteZip(URL, timeout=90) as archive:
        entries = [dict(name=i.filename, size=i.file_size, compressed=i.compress_size,
                        offset=i.header_offset, method=i.compress_type, crc=i.CRC)
                   for i in archive.infolist() if i.filename.startswith('FallDetection/') and not i.is_dir()]
    if not entries:
        raise ValueError('No FallDetection entries in official ZIP')
    entries.sort(key=lambda i: i['offset'])
    (root / 'official_fall_zip_index.json').write_text(json.dumps(entries, indent=2))
    blocks = []
    for item in entries:
        if not blocks or item['offset'] - blocks[-1][0]['offset'] > 16 * 1024**2:
            blocks.append([])
        blocks[-1].append(item)

    def valid(p, item):
        return p.is_file() and p.stat().st_size == item['size'] and zlib.crc32(p.read_bytes()) == item['crc']

    def download(block):
        if all(valid(root / item['name'], item) for item in block):
            return len(block)
        start = block[0]['offset']
        # ZIP local header length can include a 65535-byte extra field.
        last = block[-1]
        end = last['offset'] + 30 + len(last['name'].encode()) + 65535 + last['compressed'] - 1
        for attempt in range(5):
            try:
                with requests.get(URL, headers={'Range': f'bytes={start}-{end}'}, timeout=120, stream=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                        raise ValueError('Server did not honor range; refusing full dataset download')
                    chunk = response.content
                for item in block:
                    pos = item['offset'] - start
                    if chunk[pos:pos+4] != b'PK\x03\x04':
                        raise ValueError('Invalid ZIP local header')
                    name_len, extra_len = struct.unpack_from('<HH', chunk, pos+26)
                    data_start = pos + 30 + name_len + extra_len
                    raw = chunk[data_start:data_start+item['compressed']]
                    if item['method'] == 8:
                        raw = zlib.decompress(raw, -15)
                    elif item['method'] != 0:
                        raise ValueError('Unsupported ZIP compression')
                    if len(raw) != item['size'] or zlib.crc32(raw) != item['crc']:
                        raise ValueError(f"CRC or size mismatch: {item['name']}")
                    p = (root / item['name']).resolve()
                    if not p.is_relative_to(root):
                        raise ValueError('Unsafe ZIP member path')
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tmp = p.with_suffix(p.suffix+'.partial')
                    tmp.write_bytes(raw)
                    tmp.replace(p)
                return len(block)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
    count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, block) for block in blocks]
        for future in concurrent.futures.as_completed(futures):
            count += future.result()
            print(f'CRC-verified {count}/{len(entries)} files', flush=True)
    receipt = {'source_url': URL, 'dataset': 'guozhenjennzhu/csi-bench', 'version': 12,
               'downloaded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'files': len(entries), 'uncompressed_bytes': sum(i['size'] for i in entries),
               'verification': 'ZIP CRC32 and exact byte length for every entry',
               'index_sha256': hashlib.sha256((root/'official_fall_zip_index.json').read_bytes()).hexdigest()}
    (root/'download_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
