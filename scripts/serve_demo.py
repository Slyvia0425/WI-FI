"""Serve the prebuilt website and module replay APIs using Python stdlib only."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.api.server import Handler


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--web-port',type=int,default=5173);ap.add_argument('--api-port',type=int,default=8000);args=ap.parse_args()
    dist=ROOT/'frontend/dist'
    if not (dist/'index.html').exists():raise SystemExit('Built website missing. Run npm ci --prefix frontend, then npm run build.')
    api=ThreadingHTTPServer(('127.0.0.1',args.api_port),Handler)
    try:web=ThreadingHTTPServer(('127.0.0.1',args.web_port),partial(SimpleHTTPRequestHandler,directory=str(dist)))
    except BaseException:api.server_close();raise
    thread=threading.Thread(target=api.serve_forever,daemon=True);thread.start()
    print(f'Website: http://localhost:{args.web_port}/#/module/fall_detection',flush=True)
    print(f'Module B: http://localhost:{args.web_port}/#/module/human_activity_recognition',flush=True)
    print(f'API: http://localhost:{args.api_port}/api/modules',flush=True)
    print('Recorded/fixture replay only. Ctrl+C to stop.',flush=True)
    try:web.serve_forever()
    except KeyboardInterrupt:pass
    finally:web.server_close();api.shutdown();api.server_close()

if __name__=='__main__':main()
