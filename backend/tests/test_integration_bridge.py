import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from backend.api.server import Handler


class Upstream(BaseHTTPRequestHandler):
    def do_GET(self):
        responses = {
            '/': ('text/html', b'<head><link href="/style.css"></head><body><script src="/app.js"></script></body>'),
            '/app.js': ('text/javascript', b"fetch('/api/state');const x=\"/api/predict\";"),
            '/api/health': ('application/json', b'{"service":"roomsense-multi","ready":true}'),
        }
        content_type, body = responses.get(self.path, ('application/json', b'{"error":"missing"}'))
        self.send_response(200 if self.path in responses else 404)
        self.send_header('Content-Type', content_type)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'path': self.path, 'body': json.loads(body)}).encode())

    def log_message(self, *args):
        pass


class QuietHandler(Handler):
    def log_message(self, *args):
        pass


class IntegrationBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        cls.env = patch.dict(os.environ, WIFI_SPATIAL_PORT=str(cls.upstream.server_port))
        cls.env.start()
        cls.threads = []
        for server in [cls.upstream, cls.server]:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            cls.threads.append(thread)
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        for server in [cls.server, cls.upstream]:
            server.shutdown()
            server.server_close()
        for thread in cls.threads:
            thread.join(timeout=3)
        cls.env.stop()

    def get(self, path):
        return urllib.request.urlopen(self.base + path, timeout=10)

    def test_all_five_modes_without_joint_claim(self):
        with self.get('/api/integration') as response:
            data = json.load(response)
        self.assertEqual(len(data['modules']), 5)
        self.assertTrue(data['spatial']['available'])
        self.assertEqual(data['spatial']['mode'], 'Signal simulation')
        self.assertFalse(data['joint_recording'])

    def test_html_assets_and_embedding(self):
        with self.get('/spatial/') as response:
            html = response.read().decode()
        self.assertIn('/spatial/style.css', html)
        self.assertIn('/spatial/app.js', html)
        self.assertIn('.page-heading', html)
        self.assertIn('/spatial/effects.css', html)
        self.assertIn('/spatial/effects.js', html)

    def test_effects_assets_are_local(self):
        with patch('backend.api.integration.spatial_origin', return_value='http://127.0.0.1:1'):
            for path, kind in [('effects.css', 'text/css'), ('effects.js', 'text/javascript')]:
                with self.get('/spatial/' + path) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(kind, response.headers['Content-Type'])
                    self.assertGreater(len(response.read()), 100)

    def test_javascript_api_prefix(self):
        with self.get('/spatial/app.js') as response:
            data = response.read().decode()
        self.assertIn("fetch('/spatial/api/state')", data)
        self.assertIn('"/spatial/api/predict"', data)

    def test_post_body_and_query_unchanged(self):
        payload = {'objects': [{'x': 1.2, 'y': 2.4}], 'seed': 42}
        request = urllib.request.Request(self.base + '/spatial/api/simulate?layout=default',
                                         data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.load(response)
        self.assertEqual(data, {'path': '/api/simulate?layout=default', 'body': payload})

    def test_upstream_error_status_preserved(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.get('/spatial/missing')
        self.assertEqual(error.exception.code, 404)

    def test_unavailable_model_is_503(self):
        with patch('backend.api.integration.spatial_origin', return_value='http://127.0.0.1:1'):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.get('/spatial/api/state')
        self.assertEqual(error.exception.code, 503)

    def test_unavailable_model_does_not_break_replays(self):
        with patch('backend.api.integration.spatial_origin', return_value='http://127.0.0.1:1'):
            with self.get('/api/integration') as response:
                data = json.load(response)
        self.assertFalse(data['spatial']['available'])
        self.assertTrue(data['modules']['fall_detection']['available'])


if __name__ == '__main__':
    unittest.main()
