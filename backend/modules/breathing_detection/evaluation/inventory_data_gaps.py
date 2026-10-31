"""Read-only local metadata inventory for external-data acquisition planning."""
import csv
import json
from collections import Counter
from pathlib import Path

import h5py

from .evaluate_spike_preprocessing import ROOT, digest


def main():
    output = ROOT/'data_gap_inventory_v1'
    if output.exists():
        raise FileExistsError('Preserve data-gap inventory')
    base = Path('data/BreathingDetection')
    metadata = base/'metadata/sample_metadata.csv'
    rows = list(csv.DictReader(metadata.open()))
    splits = {p.stem: json.loads(p.read_text()) for p in sorted((base/'splits').glob('*.json'))}
    fan_deployments = ('LOS_FanOff', 'LOS_FanOn', 'NLOS_FanOff', 'NLOS_FanOn')
    fan = {}
    sample_rows = [rows[0]]
    for deployment in fan_deployments:
        members = [r for r in rows if r['deplyoment'] == deployment]
        ids = {r['id'] for r in members}
        fan[deployment] = {'metadata_samples': len(members), 'labels': dict(Counter(r['label'] for r in members)),
            'users': sorted({r['user'] for r in members}),
            'official_split_memberships': {name: len(ids & set(values)) for name, values in splits.items()},
            'note': 'Sample/window counts, NOT verified independent recordings; session field uniqueness is not proof.'}
        for label in sorted({r['label'] for r in members}):
            sample_rows.append(next(r for r in members if r['label'] == label))
    schemas = []
    for row in sample_rows:
        path = base/row['file_path']
        with h5py.File(path, 'r') as handle:
            schemas.append({'sample_id': row['id'], 'deployment': row['deplyoment'], 'label': row['label'],
                'datasets': {key: {'shape': list(handle[key].shape), 'dtype': str(handle[key].dtype)} for key in handle},
                'file_attribute_names': sorted(handle.attrs)})
    wire = json.loads((ROOT/'spike_preprocessing_v1/raw_audit.json').read_text())
    protected = json.loads((ROOT/'phase_feasibility_v1/config.json').read_text())['protected_artifact_hashes']
    hashes = {str(metadata): digest(metadata), **{str(p): digest(p) for p in (base/'splits').glob('*.json')}}
    report = {'date': '2026-10-29', 'local_data_directories': sorted(p.name for p in Path('data').iterdir() if p.is_dir()),
        'csi_bench': {'metadata_rows': len(rows), 'labels': dict(Counter(r['label'] for r in rows)),
            'split_counts': {name: len(values) for name, values in splits.items()},
            'fan_deployments': fan, 'sampled_hdf5_schemas': schemas,
            'schema_scope': 'Eight deterministic file checks, not exhaustive HDF5 validation.',
            'current_release_version': 'unknown locally; do not infer from directory name or replace files'},
        'wire_breath': {'source_files': len(wire), 'recordings': len({a['recording_id'] for a in wire}),
            'quality_passing_30s': sum(a['assessable'] for a in wire),
            'quality_excluded_30s': sum(not a['assessable'] for a in wire),
            'participants': sorted({a['metadata']['participant_id'] for a in wire}),
            'environments': dict(Counter(a['metadata']['environment'] for a in wire)),
            'status': 'Repeatedly observed development data, not new independent evaluation.'},
        'protected_artifact_hashes': protected, 'local_metadata_sha256': hashes,
        'no_model_evaluation_or_download': True, 'application_enabled': False}
    output.mkdir()
    (output/'inventory.json').write_text(json.dumps(report, indent=2))
    if any(digest(path) != sha for path, sha in {**hashes, **protected}.items()):
        raise ValueError('Protected model/application/metadata changed')
    print(json.dumps({k: v for k, v in report.items() if k not in ('protected_artifact_hashes', 'local_metadata_sha256')}, indent=2))


if __name__ == '__main__':
    main()
