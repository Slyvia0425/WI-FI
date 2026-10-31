import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.api.server import Handler


class IntegratedHandler(Handler, SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / 'frontend/dist'), **kwargs)

    def do_GET(self):
        if self.path.startswith(('/api/', '/spatial/', '/spatial-lite/')):
            return Handler.do_GET(self)
        return SimpleHTTPRequestHandler.do_GET(self)

    def end_headers(self):
        if not self.path.startswith(('/api/', '/spatial/', '/spatial-lite/')):
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()


def start_spatial(port, mode='research'):
    settings_file = ROOT / '.runtime/config.json'
    settings = json.loads(settings_file.read_text(encoding='utf-8')) if settings_file.exists() else {}
    project = Path(os.environ.get('WIFI_SPACE_ROOT') or settings.get('spatial_root') or str(ROOT / 'backend/modules/space_proximity_sensing/wifi space')).resolve()
    assets = Path(os.environ.get('WIFI_SPATIAL_LITE_ROOT', str(ROOT / 'runtime_assets/spatial_lite'))).resolve()
    if mode == 'lite':
        project = ROOT / 'backend/modules/space_proximity_sensing/wifi space'
    model = assets / 'bundle.json' if mode == 'lite' else project / 'multi_object/outputs/spatial_v6/default/model.joblib'
    if not model.exists():
        print(f'Spatial {mode} assets missing. See docs/SPATIAL_MODES.md.', flush=True)
        return None, None
    if mode == 'research' and not all((project / 'multi_object/data' / name).exists() for name in ['home_v5/manifest.json', 'layout_t1_v3/manifest.json', 'spatial_v6/manifest.json']):
        print('Full research data missing; light mode remains independent.', flush=True)
        return None, None
    python = os.environ.get('WIFI_SPATIAL_PYTHON') or settings.get('spatial_python')
    if not python:
        candidates = [ROOT / '.venv/Scripts/python.exe', ROOT / '.venv/bin/python', project / '.venv/Scripts/python.exe', project / '.venv/bin/python']
        python = str(next((p for p in candidates if p.exists()), Path(sys.executable)))
    logs = ROOT / '.runtime'
    logs.mkdir(exist_ok=True)
    log = (logs / ('spatial-' + mode + '.log')).open('a', encoding='utf-8')
    command = [python, '-u', str(ROOT / 'backend/modules/space_proximity_sensing/runtime.py'), '--mode', mode, '--project', str(project), '--port', str(port)]
    if mode == 'lite':
        command += ['--assets', str(assets)]
    process = subprocess.Popen(command, cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    for _ in range(90):
        if process.poll() is not None:
            print(f'Spatial {mode} startup failed. See .runtime/spatial-{mode}.log.', flush=True)
            break
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/ready', timeout=3) as response:
                ready = json.load(response)
            expected = hashlib.sha256(str((project / 'multi_object').resolve()).casefold().encode()).hexdigest()[:16]
            if ready.get('ready') and ready.get('runtime', {}).get('mode') == mode and ready.get('project_id') == expected:
                print(f'Spatial {mode} model ready.', flush=True)
                return process, log
        except (OSError, ValueError):
            time.sleep(0.5)
    return process, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5173)
    parser.add_argument('--spatial-port', type=int, default=8780)
    parser.add_argument('--api-only', action='store_true')
    parser.add_argument('--lite-port', type=int, default=8781)
    parser.add_argument('--spatial-mode', choices=['both', 'lite', 'research'], default='both')
    args = parser.parse_args()
    if not args.api_only and not (ROOT / 'frontend/dist/index.html').exists():
        raise SystemExit('Run npm install --prefix frontend, then npm run build first.')
    os.environ['WIFI_SPATIAL_PORT'] = str(args.spatial_port)
    os.environ['WIFI_SPATIAL_LITE_PORT'] = str(args.lite_port)
    os.environ['WIFI_SPATIAL_MODES'] = 'lite,research' if args.spatial_mode == 'both' else args.spatial_mode
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler if args.api_only else IntegratedHandler)
    services = []
    try:
        for mode, port in [('lite', args.lite_port), ('research', args.spatial_port)]:
            if args.spatial_mode in ['both', mode]:
                services.append(start_spatial(port, mode))
        print(f'WiFi Sense: http://127.0.0.1:{args.port}/', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        for process, log in services:
            if process is not None and process.poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if log:
                log.close()


if __name__ == '__main__':
    main()
