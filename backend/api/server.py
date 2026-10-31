"""HTTP server: turns requests into routes.dispatch calls. Standard library only."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import routes
from .integration import integration_status, proxy_spatial


class Handler(BaseHTTPRequestHandler):
    def send_json(self, body, status=200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split('?')[0] == '/api/integration':
            return self.send_json(integration_status())
        if self.path.startswith('/spatial/'):
            return proxy_spatial(self)
        status, body = routes.dispatch("GET", self.path)
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")  # the website runs on another port
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path.startswith('/spatial/'):
            return proxy_spatial(self)
        self.send_json({'error': 'Not found'}, 404)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}")


def serve(port: int = 8000) -> None:
    print(f"Serving {len(routes.MODULES)} modules on http://localhost:{port}: {', '.join(sorted(routes.MODULES))}")
    ThreadingHTTPServer(("", port), Handler).serve_forever()
