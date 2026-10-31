"""Audit native CSI from subjects2..4; reuse subject01's hash-matched audit."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
sys.path.insert(0, str(MODULE / 'tcs_fall_audit_v1'))
from audit_local_archives import audit_record, fields, reference_check, sha, write_json


def protected():
    expected = json.loads((MODULE / 'final_selection_v1/preserved_artifacts_before.json').read_text())
    for rel in ('outputs/final_selection_v1/final_research_model.pt',
                'outputs/tcs_subject01_gcv_cnn_seed42_v1/run/best.pt'):
        expected[rel] = sha(MODULE / rel)
    for rel, digest in expected.items():
        if sha(MODULE / rel) != digest:
            raise ValueError('Protected artifact changed: ' + rel)
    return expected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--intake', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing audit; choose a fresh output directory')
    intake = json.loads((args.intake / 'data_readiness.json').read_text())
    if not intake['all_archive_integrity_checks_accepted']:
        raise ValueError('Required archives have not passed intake')
    protocol = json.loads((HERE / 'protocol.json').read_text())
    if sha(HERE / 'protocol.json') != intake['protocol_sha256']:
        raise ValueError('Protocol changed after intake')
    annotations = json.loads((MODULE / 'reports/tcs_fall_audit_v1/annotations_extracted.json').read_text())
    index = {(r['subject_id'], r['activity_id'], r['repeat_index_in_cell_zero_based']): r for r in annotations}
    before = protected()
    args.output.mkdir(parents=True)
    rows, issues, reference_done = [], [], set()
    reference_source = MODULE / 'outputs/tcs_fall_local_v1/source_package/TCS-Fall-main/code/wifilib.py'
    start = time.perf_counter()
    for archive_info in intake['archives']:
        person = archive_info['subject']; role = archive_info['role']; path = Path(archive_info['path'])
        if sha(path) != archive_info['sha256']:
            raise ValueError('Archive changed since intake: ' + str(path))
        if person == 1:
            source = json.loads((MODULE / 'reports/tcs_fall_local_v1/records.json').read_text())
            for r in source:
                rows.append({**r, 'role': role, 'source_archive_sha256': archive_info['sha256'],
                             'reused_previous_native_audit': True})
            print(json.dumps({'subject': 1, 'reused_records': len(source)}), flush=True)
            continue
        with zipfile.ZipFile(path) as archive:
            entries = sorted((e for e in archive.infolist() if e.filename.endswith('.dat')), key=lambda e: e.filename)
            for i, entry in enumerate(entries, 1):
                try:
                    data = archive.read(entry)
                    r = audit_record(data, entry, index, args.output, reference_source)
                    r.update({'role': role, 'source_archive_sha256': archive_info['sha256'], 'reused_previous_native_audit': False})
                    if r['nominal_subject_id'] != person:
                        raise ValueError('Unexpected subject identifier')
                    if r['source_kind'] != 'unknown_activity_excluded':
                        if r['nonincreasing_timestamp_intervals'] or r['rf_scaled_amplitude_finite_ratio'] != 1:
                            r['errors'].append('Nonincreasing timestamps or nonfinite RF amplitude')
                        if r['packet_count'] < 8000:
                            r['errors'].append('Fewer than8000 packets')
                        if r['source_kind'] == 'annotated_continuous_candidate':
                            if not r['annotation_available'] or not r.get('nominal_fall_time_inside_raw_duration'):
                                r['errors'].append('Missing annotation or nominal event outside recording')
                            elif r['annotation_record']['recorded_fall_time_seconds'] <= 120:
                                r['errors'].append('Fall annotation is not after background interval')
                    key = (person, r['source_kind'])
                    if key not in reference_done and r['source_kind'] != 'unknown_activity_excluded':
                        packet_rows = np.frombuffer(data, dtype=np.uint8).reshape(-1, 575)
                        _, _, perm = fields(packet_rows)
                        _, indices = np.unique(packet_rows[:, 18], return_index=True)
                        selected = np.unique(np.concatenate((np.arange(min(8, len(packet_rows))), indices)))
                        r['reference_check'] = reference_check(reference_source, packet_rows, perm, selected)
                        reference_done.add(key)
                    del data
                except Exception as exc:
                    r = {'archive_member': entry.filename, 'role': role, 'nominal_subject_id': person,
                         'source_archive_sha256': archive_info['sha256'], 'errors': [f'{type(exc).__name__}: {exc}']}
                rows.append(r)
                if r['errors']:
                    issues.append({'member': entry.filename, 'errors': r['errors']})
                if i == 1 or i % 30 == 0 or i == len(entries):
                    print(json.dumps({'subject': person, 'audited': i, 'total': len(entries),
                                      'issues': len(issues), 'elapsed_seconds': round(time.perf_counter() - start, 1)}), flush=True)
                    write_json(args.output / 'progress.json', {'subject': person, 'audited': i, 'total': len(entries), 'issues': issues})
            write_json(args.output / f'subject{person:02d}_records.json', [r for r in rows if r.get('nominal_subject_id') == person])
    good = [r for r in rows if not r['errors']]
    groups = defaultdict(list)
    for r in good:
        groups[r['canonical_csi_sha256']].append(r)
    duplicates = [[{'member': r['archive_member'], 'role': r['role']} for r in v] for v in groups.values() if len(v) > 1]
    cross_role = [v for v in duplicates if len({r['role'] for r in v}) > 1]
    if cross_role:
        issues.append({'cross_role_exact_content_duplicates': cross_role})
    for rel, digest in before.items():
        if sha(MODULE / rel) != digest:
            raise ValueError('Protected artifact changed during audit: ' + rel)
    subject_summary = []
    for person, role in protocol['subject_roles'].items():
        subset = [r for r in good if r.get('nominal_subject_id') == int(person)]
        sa = [r for r in subset if r['source_kind'] == 'single_action']
        ct = [r for r in subset if r['source_kind'] == 'annotated_continuous_candidate']
        subject_summary.append({'subject': int(person), 'role': role, 'single_action': len(sa),
                                'fall': sum(r['single_action_binary_label'] == 1 for r in sa),
                                'nonfall': sum(r['single_action_binary_label'] == 0 for r in sa),
                                'continuous': len(ct), 'unknown_excluded': sum(r['source_kind'] == 'unknown_activity_excluded' for r in subset),
                                'continuous_raw_minutes': sum(r['timestamp_span_seconds'] for r in ct) / 60})
    summary = {'subject_summary': subject_summary, 'records_total_including_known_errors': len(rows),
               'new_issues': issues, 'ready_for_feature_preparation': not issues,
               'prior_authorized_errors': [r['archive_member'] for r in rows if r['errors'] and r['role'] == 'train' and r.get('reused_previous_native_audit')],
               'canonical_exact_duplicate_groups': duplicates, 'cross_role_exact_duplicate_groups': cross_role,
               'data_shape_counts': dict(Counter(str(r['packet_csi_shape']) for r in good)),
               'all_readable_amplitude_finite': all(r['rf_scaled_amplitude_finite_ratio'] == 1 for r in good),
               'reference_checks': [r['reference_check'] for r in good if r.get('reference_check') and not r.get('reused_previous_native_audit')],
               'time_origin_verified': False, 'subject_source_session_independence_beyond_identifiers': 'not independently established',
               'raw_archives_extracted': False, 'test_model_predictions': False, 'training_run': False,
               'protected_artifacts_unchanged': True, 'protected_artifact_count': len(before),
               'audit_helper_sha256': sha(MODULE / 'tcs_fall_audit_v1/audit_local_archives.py'),
               'protocol_sha256': sha(HERE / 'protocol.json'), 'archives': intake['archives'],
               'elapsed_seconds': time.perf_counter() - start}
    write_json(args.output / 'records.json', rows)
    write_json(args.output / 'audit_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary['ready_for_feature_preparation'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
