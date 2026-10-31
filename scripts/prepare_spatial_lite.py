import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILES = ['active_profile.json', 'home_v5/model.joblib', 'home_v5/results.json']
OUTPUT_FILES += ['spatial_v6/default/' + name for name in ['model.joblib', 'APPROVED.json', 'frozen.json', 'results.json', 'api_verification.json']]
OUTPUT_FILES += ['layout_t1_v3/' + name for name in ['model.joblib', 'APPROVED.json', 'frozen.json', 'results.json', 'data_audit.json', 'evaluation_binding.json', 'api_verification.json']]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(folder, source_project=None):
    spec = json.loads((folder / 'bundle.json').read_text(encoding='utf-8'))
    for name, expected in spec['files'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise RuntimeError('Missing runtime file: ' + name)
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise RuntimeError('Runtime checksum mismatch: ' + name)
    if source_project:
        for name, expected in spec['source_sha256'].items():
            if digest(source_project / 'multi_object' / name) != expected:
                raise RuntimeError('Frozen source mismatch: ' + name)
    return spec


def build(source, destination, archive=None):
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    def copy(original, relative):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = digest(original)
        if not target.exists() or digest(target) != expected:
            shutil.copy2(original, target)
        files[relative] = {'bytes': target.stat().st_size, 'sha256': expected}
    for name in OUTPUT_FILES:
        copy(source / 'multi_object/outputs' / name, 'outputs/' + name)
    examples = ROOT / 'datasets/space_proximity_sensing/sample'
    samples = json.loads((examples / 'manifest.json').read_text(encoding='utf-8'))
    for item in samples:
        copy(examples / item['path'], 'samples/' + item['path'])
    copy(examples / 'manifest.json', 'samples/manifest.json')
    frozen_sources = {}
    for name in ['spatial_v6/default/frozen.json', 'layout_t1_v3/frozen.json']:
        frozen_sources.update(json.loads((source / 'multi_object/outputs' / name).read_text(encoding='utf-8'))['source_sha256'])
    spec = {'format': 1, 'mode': 'lite', 'models': ['spatial_count_v6', 't1_count_v3'], 'full_dataset_included': False,
            'sample_count': len(samples), 'payload_bytes': sum(item['bytes'] for item in files.values()),
            'source_sha256': frozen_sources, 'files': files}
    (destination / 'bundle.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    verify(destination, source)
    if archive:
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix('.partial')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as package:
            for name in ['bundle.json', *files]:
                package.write(destination / name, name)
        temporary.replace(archive)
    return spec


def restore(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        spec = json.loads(package.read('bundle.json'))
        allowed = {'bundle.json', *spec['files']}
        for info in package.infolist():
            if info.filename not in allowed:
                raise RuntimeError('Unexpected runtime archive member')
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError('Invalid runtime archive path')
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(info) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    return verify(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-project', type=Path)
    parser.add_argument('--restore', type=Path)
    parser.add_argument('--destination', type=Path, default=ROOT / 'runtime_assets/spatial_lite')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.from_project:
        spec = build(args.from_project.resolve(), args.destination.resolve(), args.archive)
    elif args.restore:
        spec = restore(args.restore, args.destination.resolve())
    else:
        spec = verify(args.destination.resolve(), ROOT / 'backend/modules/space_proximity_sensing/wifi space')
    print(json.dumps({'ready': True, 'payload_bytes': spec['payload_bytes'], 'sample_count': spec['sample_count'], 'full_dataset_included': False}))


if __name__ == '__main__':
    main()
