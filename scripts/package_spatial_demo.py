import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assets = ROOT / 'runtime_assets/spatial_lite'
    from prepare_spatial_lite import verify
    spec = verify(assets, ROOT / 'backend/modules/space_proximity_sensing/wifi space')
    git = 'git'
    import shutil
    if not shutil.which(git):
        fallback = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
        git = str(fallback) if fallback.exists() else git
    source_files = subprocess.check_output([git, 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    prefix = 'wifi space light/'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.partial')
    count = 0
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as package:
        for name in sorted(set(source_files)):
            if not name or not (ROOT / name).is_file():
                continue
            if name.startswith('datasets/') and '/sample/' in name and not name.startswith('datasets/space_proximity_sensing/'):
                continue
            if name.startswith(('.runtime/', 'runtime_assets/', 'node_modules/', '.venv/')) or '/data/' in name and not name.startswith('frontend/') and not '/sample/' in name and not '/fixtures/' in name:
                continue
            package.write(ROOT / name, prefix + name)
            count += 1
        for file in (ROOT / 'frontend/dist').rglob('*'):
            if file.is_file():
                package.write(file, prefix + file.relative_to(ROOT).as_posix())
        for name in ['bundle.json', *spec['files']]:
            package.write(assets / name, prefix + 'runtime_assets/spatial_lite/' + name)
        package.writestr(prefix + 'START_HERE.md', '# WiFi Sense light demo\n\n1. Install Python 3.12.\n2. Run INSTALL_LIGHT.bat once to install the ML and ray-tracing libraries.\n3. Run START_LIGHT.bat.\n4. Open http://127.0.0.1:5173/ and select Spatial sensing.\n\nThe website is prebuilt; Node.js is not required. The package includes unchanged models and six example scenes. No full training/test dataset download is needed. The full research assets can be added separately.\n')
    temporary.replace(args.output)
    with zipfile.ZipFile(args.output) as package:
        invalid = package.testzip()
        if invalid:
            raise RuntimeError('Package CRC failed: ' + invalid)
        assert not any('/datasets/space_proximity_sensing/data/' in name for name in package.namelist())
    value = hashlib.sha256()
    with args.output.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(block)
    report = {'archive': str(args.output), 'bytes': args.output.stat().st_size, 'sha256': value.hexdigest(), 'source_files': count, 'runtime_payload_bytes': spec['payload_bytes'], 'full_dataset_included': False, 'zip_crc_passed': True}
    (ROOT / '.runtime/lite_package.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
