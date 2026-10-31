import argparse
import json
from pathlib import Path
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def normalized(value):
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items() if k not in {'elapsed_ms', 'simulation_ms'}}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def request(url, payload):
    req = urllib.request.Request(url + '/api/predict', data=json.dumps(payload, allow_nan=False).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lite', default='http://127.0.0.1:8781')
    parser.add_argument('--research', default='http://127.0.0.1:8780')
    parser.add_argument('--report', type=Path, default=ROOT / '.runtime/mode_equivalence.json')
    args = parser.parse_args()
    folder = ROOT / 'datasets/space_proximity_sensing/sample'
    rows = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    reports = []
    for row in rows:
        with np.load(folder / row['path'], allow_pickle=False) as sample:
            h = np.repeat(sample['h'][None], 16, axis=0).astype(np.complex64)
        for snr in [30, 10]:
            rng = np.random.default_rng(421)
            noise = (rng.normal(size=h.shape) + 1j * rng.normal(size=h.shape)) * np.sqrt(np.mean(np.abs(h) ** 2)) * 10 ** (-snr / 20) / np.sqrt(2)
            signal = (h + noise).astype(np.complex64)
            payload = {'real': signal.real.tolist(), 'imag': signal.imag.tolist(), 'layout_id': row['layout_id']}
            lite, research = request(args.lite, payload), request(args.research, payload)
            if normalized(lite) != normalized(research):
                raise RuntimeError('Mode predictions differ: ' + row['path'])
            reports.append({'sample': row['path'], 'snr_db': snr, 'status': lite['status'], 'identical': True})
    for layout in ['default', 't1_v1']:
        signal = np.zeros((16, 32, 64), np.float32).tolist()
        payload = {'real': signal, 'imag': signal, 'layout_id': layout}
        lite, research = request(args.lite, payload), request(args.research, payload)
        assert normalized(lite) == normalized(research) and lite['status'] == 'Unknown'
        reports.append({'sample': 'missing_signal', 'layout': layout, 'status': 'Unknown', 'identical': True})
    result = {'passed': True, 'compared_inputs': len(reports), 'comparison': 'All response fields except computation timing', 'cases': reports}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'compared_inputs': len(reports)}))


if __name__ == '__main__':
    main()
