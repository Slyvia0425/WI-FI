import argparse
import importlib.util
import json
from pathlib import Path
import sys
from urllib.parse import urlparse


def block_full_data(project):
    repo = Path(__file__).resolve().parents[3]
    forbidden = [repo / 'datasets/space_proximity_sensing/data', project / 'data', project / 'multi_object/data']
    forbidden = [p.resolve() for p in forbidden]
    def guard(event, arguments):
        if event != 'open' or not arguments or not isinstance(arguments[0], (str, bytes)):
            return
        path = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]).resolve()
        if any(path.is_relative_to(folder) for folder in forbidden):
            raise RuntimeError('Full dataset access is disabled in light mode')
    sys.addaudithook(guard)


def load_core(project, mode, assets=None):
    source = project / 'multi_object'
    sys.path.insert(0, str(source))
    import multi_common
    samples = {}
    if mode == 'lite':
        block_full_data(project)
        if assets is None or not (assets / 'bundle.json').exists():
            raise RuntimeError('Light runtime assets are not installed')
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root / 'scripts'))
        from prepare_spatial_lite import verify
        verify(assets, project)
        multi_common.OUT = assets / 'outputs'
        multi_common.DATA = assets / 'unused_data'
        entries = json.loads((assets / 'samples/manifest.json').read_text(encoding='utf-8'))
        for item in entries:
            identifier = 'demo_' + item['layout_id'] + '_' + str(len(item['objects']))
            samples[identifier] = dict(item, id=identifier, count=len(item['objects']), region_count=len({o['zone'] for o in item['objects']}), shared=len({o['zone'] for o in item['objects']}) < len(item['objects']))
    spec = importlib.util.spec_from_file_location('spatial_core', source / 'app.py')
    core = importlib.util.module_from_spec(spec)
    sys.modules['spatial_core'] = core
    spec.loader.exec_module(core)
    if mode == 'lite':
        core.PROFILE_OVERRIDE = 5
        core.catalog = lambda: samples
        core.active_record = lambda row, version=None: row['id'] in samples
        core.record_path = lambda row: assets / 'samples' / samples[row['id']]['path']
    return core, samples


def handler_class(core, mode, samples):
    runtime = {'mode': mode, 'full_dataset_required': mode == 'research', 'full_replay': mode == 'research',
               'sample_count': len(samples), 'simulation': True, 'signal_import': True,
               'saved_evaluation': True, 'models_retrained': False,
               'full_dataset_access_blocked': mode == 'lite'}
    class Handler(core.Handler):
        def send(self, data, code=200):
            if isinstance(data, dict) and urlparse(self.path).path in ['/api/state', '/api/health', '/api/ready']:
                data = dict(data, runtime=runtime)
            return super().send(data, code)

        def do_GET(self):
            if urlparse(self.path).path == '/api/runtime':
                return self.send(runtime)
            return super().do_GET()
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['lite', 'research'], required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    core, samples = load_core(args.project.resolve(), args.mode, args.assets.resolve() if args.assets else None)
    server = core.LocalHTTPServer(('127.0.0.1', args.port), handler_class(core, args.mode, samples))
    print(f'Spatial {args.mode}: http://127.0.0.1:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
