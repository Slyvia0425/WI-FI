"""Inventory required subject archives; check integrity without training or scoring.

Only explicitly named1.zip..4.zip are opened. No extraction or raw CSI caching.
Existing subject01 audit is reused only when the full archive hash matches.
"""
import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def inspect(path, subject, role, protocol, prior, annotations):
    if not path.is_file():
        return {'subject': subject, 'role': role, 'path': str(path), 'status': 'missing'}, []
    digest = sha(path)
    reuse = subject == 1 and digest == protocol['existing_subject01']['allowed_partial_archive_sha256']
    exclusions = set(protocol['existing_subject01']['authorized_excluded_members']) if reuse else set()
    rows, errors = [], []
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate ZIP member names')
            for entry in z.infolist():
                if entry.is_dir():
                    continue
                member = PurePosixPath(entry.filename)
                if member.is_absolute() or '..' in member.parts or entry.flag_bits & 1:
                    raise ValueError('Unsafe or encrypted member: ' + entry.filename)
                if member.suffix.lower() != '.dat':
                    continue
                match = re.fullmatch(r'a_(\d+)_(\d+)_(\d+)\.dat', member.name)
                if not match:
                    errors.append('Unrecognized DAT identifier: ' + entry.filename)
                    continue
                person, activity, repeat = map(int, match.groups())
                if person != subject:
                    errors.append(f'Subject mismatch in {entry.filename}: expected{subject}, observed{person}')
                kind = 'single_action' if activity in range(7) else 'continuous' if activity in range(10, 16) else 'unknown_excluded'
                row = {'archive': str(path), 'member': entry.filename, 'subject': person,
                       'activity': activity, 'repeat': repeat, 'role': role, 'kind': kind,
                       'bytes': entry.file_size, 'integrity': '', 'sha256': '', 'error': '',
                       'annotation_structural_match': (person, activity, repeat) in annotations,
                       'native_audit': 'pending'}
                old = prior.get(entry.filename) if reuse else None
                if old is not None:
                    if old['errors']:
                        row['integrity'] = 'known_corrupt_excluded' if entry.filename in exclusions else 'failed'
                        row['error'] = '; '.join(old['errors'])
                    else:
                        row['integrity'] = 'verified_prior_audit_same_archive_hash'
                        row['sha256'] = old['raw_file_sha256']
                        row['native_audit'] = 'verified_prior_audit'
                else:
                    try:
                        h = hashlib.sha256()
                        with z.open(entry) as f:
                            for part in iter(lambda: f.read(1024 * 1024), b''):
                                h.update(part)
                        row['sha256'] = h.hexdigest()
                        row['integrity'] = 'crc_verified'
                    except Exception as exc:
                        row['integrity'] = 'failed'
                        row['error'] = f'{type(exc).__name__}: {exc}'
                if row['integrity'] == 'failed':
                    errors.append(entry.filename + ': ' + row['error'])
                rows.append(row)
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    accepted = [r for r in rows if r['integrity'] not in ('failed', 'known_corrupt_excluded')]
    for activity in range(7):
        if not any(r['activity'] == activity for r in accepted):
            errors.append(f'No usable SA recording for activity{activity}')
    for activity in range(10, 16):
        for repeat in (0, 1):
            if not any(r['activity'] == activity and r['repeat'] == repeat and r['annotation_structural_match'] for r in accepted):
                errors.append(f'Missing readable annotated continuous activity{activity}/repeat{repeat}')
    ids = [(r['subject'], r['activity'], r['repeat']) for r in rows]
    if len(ids) != len(set(ids)):
        errors.append('Duplicate subject/activity/repeat identifiers')
    known_excluded = [r['member'] for r in rows if r['integrity'] == 'known_corrupt_excluded']
    status = 'needs_review' if errors else 'accepted_with_known_exclusions' if known_excluded else 'integrity_checked'
    return {'subject': subject, 'role': role, 'path': str(path), 'bytes': path.stat().st_size,
            'sha256': digest, 'status': status,
            'errors': errors, 'reused_prior_audit': reuse,
            'dat_entries': len(rows), 'readable_by_kind': dict(Counter(r['kind'] for r in accepted)),
            'excluded_known_corrupt': known_excluded,
            'native_audit_pending': not reuse}, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=Path('/Users/chenxiaohe/Downloads'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; preserve previous intake checks')
    protocol = json.loads((HERE / 'protocol.json').read_text())
    prior = {r['archive_member']: r for r in json.loads((MODULE / 'reports/tcs_fall_local_v1/records.json').read_text())}
    annotations = {(r['subject_id'], r['activity_id'], r['repeat_index_in_cell_zero_based'])
                   for r in json.loads((MODULE / 'reports/tcs_fall_audit_v1/annotations_extracted.json').read_text())}
    args.output.mkdir(parents=True)
    inventories, manifest = [], []
    for subject, role in protocol['subject_roles'].items():
        item, rows = inspect(args.data_dir / f'{subject}.zip', int(subject), role, protocol, prior, annotations)
        inventories.append(item); manifest.extend(rows)
        print(json.dumps({k: item[k] for k in ('subject', 'role', 'status', 'path')}), flush=True)
    missing = [r['path'] for r in inventories if r['status'] == 'missing']
    result = {'protocol_sha256': sha(HERE / 'protocol.json'), 'archives': inventories,
              'missing_archives': missing, 'training_started': False, 'model_evaluation_started': False,
              'all_archive_integrity_checks_accepted': all(r['status'] in ('integrity_checked', 'accepted_with_known_exclusions') for r in inventories),
              'training_ready': False,
              'remaining_gates': ['Obtain missing subject archives', 'Audit new native CSI/timestamps/annotations and cross-role content duplicates',
                                  'Build disjoint subject manifests, fix sampling protocol, and verify preprocessing', 'Only then begin new comparison runs'],
              'note': 'CRC/hash acceptance is not a replacement for native data audit; structural annotation joins do not establish exact synchronization.'}
    (args.output / 'data_readiness.json').write_text(json.dumps(result, indent=2))
    if manifest:
        with (args.output / 'archive_members.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
            writer.writeheader(); writer.writerows(manifest)
    print(json.dumps({'missing': missing, 'training_ready': False, 'report': str(args.output.resolve())}, indent=2))
    return 0 if result['all_archive_integrity_checks_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
