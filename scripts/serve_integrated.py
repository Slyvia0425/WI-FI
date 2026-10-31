import argparse
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
        if self.path.startswith(('/api/', '/spatial/')):
            return Handler.do_GET(self)
        return SimpleHTTPRequestHandler.do_GET(self)


def start_spatial(port):
    settings_file = ROOT / '.runtime/config.json'
    settings = json.loads(settings_file.read_text(encoding='utf-8')) if settings_file.exists() else {}
    project = Path(os.environ.get('WIFI_SPACE_ROOT') or settings.get('spatial_root') or str(ROOT / 'backend/modules/space_proximity_sensing/wifi space')).resolve()
    app = project / 'multi_object/app.py'
    model = project / 'multi_object/outputs/spatial_v6/default/model.joblib'
    if not model.exists():
        print('Spatial models missing. Run install.bat inside wifi space to restore them.', flush=True)
        return None, None
    python = os.environ.get('WIFI_SPATIAL_PYTHON') or settings.get('spatial_python')
    if not python:
        candidates = [project / '.venv/Scripts/python.exe', project / '.venv/bin/python']
        python = str(next((p for p in candidates if p.exists()), Path(sys.executable)))
    logs = ROOT / '.runtime'
    logs.mkdir(exist_ok=True)
    log = (logs / 'spatial.log').open('a', encoding='utf-8')
    process = subprocess.Popen([python, '-u', str(app), '--port', str(port)], cwd=app.parent,
                               stdout=log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    for _ in range(90):
        if process.poll() is not None:
            print('Spatial startup failed. See .runtime/spatial.log.', flush=True)
            break
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/ready', timeout=3) as response:
                ready = json.load(response)
            if ready.get('ready'):
                print('Spatial model ready.', flush=True)
                return process, log
        except (OSError, ValueError):
            time.sleep(0.5)
    return process, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5173)
    parser.add_argument('--spatial-port', type=int, default=8780)
    parser.add_argument('--api-only', action='store_true')
    args = parser.parse_args()
    if not args.api_only and not (ROOT / 'frontend/dist/index.html').exists():
        raise SystemExit('Run npm install --prefix frontend, then npm run build first.')
    os.environ['WIFI_SPATIAL_PORT'] = str(args.spatial_port)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler if args.api_only else IntegratedHandler)
    process, log = None, None
    try:
        process, log = start_spatial(args.spatial_port)
        print(f'WiFi Sense: http://127.0.0.1:{args.port}/', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if process is not None and process.poll() is None:
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
