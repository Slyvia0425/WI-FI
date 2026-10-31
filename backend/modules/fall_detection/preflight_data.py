"""Inspect locally available evidence only; does not establish official splits or axes."""
import argparse
import collections
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspection-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.inspection_root.resolve()
    with (root / 'fall_metadata.csv').open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    task = root / 'FallDetection'
    result = {
        'status': 'incomplete_local_inspection_not_official_audit',
        'metadata': str(root / 'fall_metadata.csv'), 'fields': fields,
        'metadata_records': len(rows),
        'counts': {key: dict(collections.Counter(r[key] for r in rows))
                   for key in ('label', 'device', 'Difficulty')},
        'device_label': [dict(device=d, original_label=l, count=n)
                         for (d, l), n in sorted(collections.Counter(
                             (r['device'], r['label']) for r in rows).items())],
        'duplicate_id_excess': len(rows) - len({r['id'] for r in rows}),
        'missing_files': sum(not (task / r['file_path']).is_file() for r in rows),
        'official_metadata_present': (task / 'metadata/sample_metadata.csv').is_file(),
        'label_mapping_present': (task / 'metadata/label_mapping.json').is_file(),
        'split_files': [str(p) for p in sorted((task / 'splits').glob('*.json'))],
        'available_hdf5': [], 'axes': 'unverified',
    }
    for path in sorted(task.rglob('*.h5')):
        record = {'file': str(path), 'datasets': [], 'error': None}
        try:
            with h5py.File(path) as handle:
                def inspect(key, obj):
                    if isinstance(obj, h5py.Dataset):
                        data = obj[()]
                        record['datasets'].append({
                            'key': key, 'shape': list(data.shape), 'dtype': str(data.dtype),
                            'finite_ratio': float(np.isfinite(data).mean()) if data.size else None,
                            'attributes': {k: str(v) for k, v in obj.attrs.items()},
                        })
                handle.visititems(inspect)
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        result['available_hdf5'].append(record)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
