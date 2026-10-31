import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from .routes import MODULES


def spatial_origin(mode='research'):
    name, fallback = ('WIFI_SPATIAL_LITE_PORT', '8781') if mode == 'lite' else ('WIFI_SPATIAL_PORT', '8780')
    return f"http://127.0.0.1:{int(os.environ.get(name, fallback))}"


def integration_status():
    modules = {}
    for name, module in MODULES.items():
        if name == 'space_proximity_sensing':
            continue
        try:
            data = module.replay()
            modules[name] = {
                'available': bool(data and data.get('frames')),
                'frames': len(data.get('frames', [])) if data else 0,
                'mode': 'Example replay' if data and data.get('mock') else 'Recorded results',
                'dataset': data.get('dataset') if data else None,
            }
        except Exception:
            modules[name] = {'available': False, 'frames': 0, 'mode': 'Unavailable'}
    spatial = {'available': False, 'mode': 'Signal simulation'}
    enabled = os.environ.get('WIFI_SPATIAL_MODES', 'lite,research').split(',')
    try:
        if 'research' not in enabled:
            raise ValueError('Research profile is not started')
        with urllib.request.urlopen(spatial_origin() + '/api/health', timeout=2) as response:
            state = json.load(response)
        spatial['available'] = state.get('service') == 'roomsense-multi' and state.get('ready') is True
    except (OSError, ValueError):
        pass
    profiles = {'research': dict(spatial)}
    profiles['lite'] = {'available': False, 'mode': 'lite', 'full_dataset_required': False, 'sample_count': 6}
    try:
        if 'lite' not in enabled:
            raise ValueError('Light profile is not started')
        with urllib.request.urlopen(spatial_origin('lite') + '/api/health', timeout=2) as response:
            state = json.load(response)
        profiles['lite'].update(state.get('runtime', {}), available=state.get('service') == 'roomsense-multi' and state.get('ready') is True)
    except (OSError, ValueError):
        pass
    profiles['research'].update(mode='research', full_dataset_required=True)
    spatial['profiles'] = profiles
    spatial['available'] = any(p['available'] for p in profiles.values())
    modules['space_proximity_sensing'] = spatial
    return {'modules': modules, 'spatial': spatial, 'joint_recording': False}


def proxy_spatial(handler):
    parsed = urlsplit(handler.path)
    prefix = '/spatial-lite' if parsed.path.startswith('/spatial-lite/') else '/spatial'
    mode = 'lite' if prefix == '/spatial-lite' else 'research'
    if mode not in os.environ.get('WIFI_SPATIAL_MODES', 'lite,research').split(','):
        return handler.send_json({'error': 'This runtime profile is not started'}, 503)
    path = parsed.path.removeprefix(prefix) or '/'
    if path in ('/effects.css', '/effects.js', '/runtime-ui.js'):
        name = 'spatial-runtime.js' if path == '/runtime-ui.js' else 'spatial-effects.css' if path.endswith('.css') else 'spatial-effects.js'
        file = Path(__file__).parent / 'assets' / name
        data = file.read_bytes()
        handler.send_response(200)
        handler.send_header('Content-Type', 'text/css; charset=utf-8' if path.endswith('.css') else 'text/javascript; charset=utf-8')
        handler.send_header('Cache-Control', 'no-store')
        handler.send_header('Content-Length', str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)
        return
    origin = spatial_origin('lite') if prefix == '/spatial-lite' else spatial_origin()
    target = origin + path + ('?' + parsed.query if parsed.query else '')
    try:
        length = int(handler.headers.get('Content-Length', '0'))
        if length < 0 or length > 8_000_000:
            return handler.send_json({'error': 'Request too large'}, 413)
        body = handler.rfile.read(length) if handler.command == 'POST' else None
        request = urllib.request.Request(target, data=body, method=handler.command,
                                         headers={'Content-Type': 'application/json'})
        try:
            response = urllib.request.urlopen(request, timeout=180)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read()
            status = response.status
            content_type = response.headers.get('Content-Type', 'application/octet-stream')
            disposition = response.headers.get('Content-Disposition')
        if path == '/app.js' and status == 200:
            data = data.replace(b"'/api/", ("'" + prefix + '/api/').encode()).replace(b'"/api/', ('"' + prefix + '/api/').encode())
        if path == '/' and status == 200:
            data = data.replace(b'href="/style.css"', b'href="/spatial/style.css"').replace(b'src="/app.js"', b'src="/spatial/app.js"').replace(b'href="/"', b'href="/spatial/"')
            data = data.replace(b'</head>', b'<style>body>header,.page-heading{display:none!important}main{padding:0!important;max-width:none!important}.workspace{margin-top:0!important}body{background:#070b14!important}</style></head>')
            data = data.replace(b'</head>', b'<link rel="stylesheet" href="/spatial/effects.css"></head>')
            data = data.replace(b'</body>', b'<script src="/spatial/effects.js"></script></body>')
            data = data.replace(b'</body>', b'<script src="/spatial/runtime-ui.js"></script></body>')
            if prefix == '/spatial-lite':
                data = data.replace(b'/spatial/', b'/spatial-lite/')
        handler.send_response(status)
        handler.send_header('Content-Type', content_type)
        handler.send_header('Cache-Control', 'no-store')
        handler.send_header('Content-Length', str(len(data)))
        if disposition:
            handler.send_header('Content-Disposition', disposition)
        handler.end_headers()
        handler.wfile.write(data)
    except (OSError, ValueError):
        handler.send_json({'error': 'Spatial service unavailable'}, 503)
