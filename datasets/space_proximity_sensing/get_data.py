import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile
import zlib

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def rows():
    with gzip.open(HERE / 'file_index.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            relative = PurePosixPath(row['dataset_path'])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in {'single_object', 'multi_object'}:
                raise RuntimeError('Invalid dataset path')
            yield row


def matches(path, row):
    if not path.is_file() or path.stat().st_size != int(row['bytes']):
        return False
    crc = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            crc = zlib.crc32(block, crc)
    return f'{crc & 0xffffffff:08x}' == row['crc32']


def archive_path(spec, supplied):
    if supplied:
        path = Path(supplied).resolve()
        if path.stat().st_size != spec['archive_bytes'] or sha256(path) != spec['archive_sha256']:
            raise RuntimeError('Archive checksum mismatch')
        return path
    cache = HERE / '.downloads'
    cache.mkdir(exist_ok=True)
    path = cache / 'wifi-space.zip'
    if path.exists() and path.stat().st_size == spec['archive_bytes'] and sha256(path) == spec['archive_sha256']:
        return path
    for part in spec['parts']:
        target = cache / part['name']
        if target.exists() and target.stat().st_size == part['bytes'] and sha256(target) == part['sha256']:
            continue
        partial = target.with_name(target.name + '.partial')
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {'User-Agent': 'wifi-space-dataset'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        url = f"https://github.com/{spec['repository']}/releases/download/{spec['tag']}/{part['name']}"
        print('Downloading', part['name'], flush=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as response:
            if response.status != 206:
                offset = 0
            with partial.open('ab' if offset else 'wb') as output:
                shutil.copyfileobj(response, output, 4 * 1024 * 1024)
        if partial.stat().st_size != part['bytes'] or sha256(partial) != part['sha256']:
            raise RuntimeError('Part checksum mismatch')
        partial.replace(target)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as output:
        for part in spec['parts']:
            with (cache / part['name']).open('rb') as source:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    if temporary.stat().st_size != spec['archive_bytes'] or sha256(temporary) != spec['archive_sha256']:
        raise RuntimeError('Archive checksum mismatch')
    temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', help='Existing complete wifi space.zip')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    metadata = json.loads((HERE / 'manifest.json').read_text(encoding='utf-8'))
    if sha256(HERE / 'file_index.csv.gz') != metadata['index_sha256']:
        raise RuntimeError('File index checksum mismatch')
    archive = None if args.verify_only else zipfile.ZipFile(archive_path(metadata['release'], args.archive))
    checked = 0
    base = DATA.resolve()
    safe_parents = set()
    try:
        for row in rows():
            target = DATA / row['dataset_path']
            if target.parent not in safe_parents:
                if not target.parent.resolve().is_relative_to(base):
                    raise RuntimeError('Dataset target outside its own folder')
                safe_parents.add(target.parent)
            if target.is_symlink():
                raise RuntimeError('Symbolic file links are not accepted')
            if not matches(target, row):
                if args.verify_only:
                    raise RuntimeError('Missing or damaged file: ' + row['dataset_path'])
                info = archive.getinfo(row['archive_path'])
                if info.file_size != int(row['bytes']) or f'{info.CRC:08x}' != row['crc32']:
                    raise RuntimeError('Archive/index mismatch')
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.partial')
                with archive.open(info) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                if not matches(temporary, row):
                    raise RuntimeError('Extracted file checksum mismatch')
                temporary.replace(target)
            checked += 1
            if checked % 10000 == 0:
                print('Verified', checked, '/', metadata['file_count'], flush=True)
    finally:
        if archive:
            archive.close()
    if checked != metadata['file_count']:
        raise RuntimeError('File count mismatch')
    print(f'Verified {checked} files in {DATA}', flush=True)


if __name__ == '__main__':
    main()
