"""Apply explicit input eligibility to the two inspected native audit findings.

Raw archives are never edited. A short SA is ineligible for the frozen8000-packet
input. One verified incomplete trailing header lies after8980 complete frames;
its complete prefix is audited and its first8000 packets remain usable.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
sys.path.insert(0, str(MODULE / 'tcs_fall_audit_v1'))
from audit_local_archives import audit_record, fields, reference_check, sha, write_json

SHORT = '2/a_02_03_05.dat'
TRAILER = '3/a_03_05_04.dat'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Use fresh eligibility output')
    summary = json.loads((args.audit / 'audit_summary.json').read_text())
    if {r.get('member') for r in summary['new_issues']} != {SHORT, TRAILER}:
        raise ValueError('Findings differ from the reviewed cases; review new findings separately')
    rows = json.loads((args.audit / 'records.json').read_text())
    short = next(r for r in rows if r['archive_member'] == SHORT)
    if short['packet_count'] != 6973 or short['errors'] != ['Fewer than8000 packets']:
        raise ValueError('Short record differs from inspected evidence')
    args.output.mkdir(parents=True)
    archive3 = next(a for a in summary['archives'] if a['subject'] == 3)
    if sha(Path(archive3['path'])) != archive3['sha256']:
        raise ValueError('Subject03 archive changed')
    with zipfile.ZipFile(archive3['path']) as archive:
        entry = archive.getinfo(TRAILER)
        data = archive.read(entry)
    pos = 0; complete = 0
    while pos + 2 <= len(data):
        length = int.from_bytes(data[pos:pos+2], 'big')
        if pos + length + 2 > len(data):
            break
        if length != 573 or data[pos+2] != 187:
            raise ValueError('Unexpected frame in reviewed complete prefix')
        pos += length + 2; complete += 1
    if complete != 8980 or len(data) - pos != 2 or data[pos:] != bytes.fromhex('023d'):
        raise ValueError('Trailing record is not the reviewed two-byte unfinished header')
    extracted = json.loads((MODULE / 'reports/tcs_fall_audit_v1/annotations_extracted.json').read_text())
    annotation_index = {(r['subject_id'], r['activity_id'], r['repeat_index_in_cell_zero_based']): r for r in extracted}
    source = MODULE / 'outputs/tcs_fall_local_v1/source_package/TCS-Fall-main/code/wifilib.py'
    fixed = audit_record(data[:pos], entry, annotation_index, args.output, source)
    if fixed['rf_scaled_amplitude_finite_ratio'] != 1 or fixed['nonincreasing_timestamp_intervals']:
        raise ValueError('Complete prefix failed numeric/time audit')
    packet_rows = np.frombuffer(data[:pos], dtype=np.uint8).reshape(-1, 575)
    _, _, permutation = fields(packet_rows)
    _, unique_indices = np.unique(packet_rows[:,18], return_index=True)
    fixed['reference_check'] = reference_check(source, packet_rows, permutation,
                                              np.unique(np.concatenate((np.arange(8), unique_indices))))
    fixed.update({'decoded_complete_prefix_sha256': fixed['raw_file_sha256'],
                  'raw_file_sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                  'decoded_complete_prefix_bytes': pos, 'trailing_bytes_excluded_from_decode': 2,
                  'source_archive_sha256': archive3['sha256'], 'role': 'validation',
                  'source_issue': 'Incomplete trailing frame header after8980 intact frames',
                  'adaptation': 'Use the verified complete frame prefix; unchanged first8000-packet model input; original DAT/ZIP preserved'})
    rows = [fixed if r['archive_member'] == TRAILER else r for r in rows]
    for r in rows:
        if r['archive_member'] == SHORT:
            r['input_eligibility'] = 'excluded_short_record'
        elif r['errors']:
            r['input_eligibility'] = 'excluded_known_archive_corruption'
        elif r['source_kind'] == 'unknown_activity_excluded':
            r['input_eligibility'] = 'excluded_unknown_semantics'
        else:
            r['input_eligibility'] = 'eligible_verified_prefix' if r['archive_member'] == TRAILER else 'eligible'
    eligible = [r for r in rows if r['input_eligibility'].startswith('eligible')]
    groups = defaultdict(list)
    for r in rows:
        if r.get('canonical_csi_sha256'):
            groups[r['canonical_csi_sha256']].append(r)
    duplicates = [[{'member':r['archive_member'], 'role':r['role']} for r in g] for g in groups.values() if len(g)>1]
    cross_role = [g for g in duplicates if len({r['role'] for r in g})>1]
    if cross_role:
        raise ValueError('Cross-role duplicates require source review')
    counts=[]
    for person, role in ((1,'train'),(2,'train'),(3,'validation'),(4,'test')):
        sub=[r for r in eligible if r['nominal_subject_id']==person]
        sa=[r for r in sub if r['source_kind']=='single_action']
        counts.append({'subject':person,'role':role,'single_action':len(sa),
                       'fall':sum(r['single_action_binary_label']==1 for r in sa),
                       'nonfall':sum(r['single_action_binary_label']==0 for r in sa),
                       'continuous':sum(r['source_kind']=='annotated_continuous_candidate' for r in sub)})
    write_json(args.output/'records.json',rows)
    decision={'status':'eligible cohort ready for feature preparation', 'subject_summary':counts,
              'eligible_record_count':len(eligible), 'eligibility_counts':dict(Counter(r['input_eligibility'] for r in rows)),
              'decision_basis':'Apply fixed8000-packet input requirement uniformly; no padding, time stretch, relabeling, subject reassignment or raw-file edits.',
              'short_record':{'member':SHORT,'packets':6973,'seconds':short['timestamp_span_seconds'],'role':'train','label':'Nonfall'},
              'verified_prefix':{'member':TRAILER,'complete_packets':8980,'incomplete_tail_bytes':2,'input_packets':8000,
                                 'raw_sha256':fixed['raw_file_sha256'],'reference_check':fixed['reference_check']},
              'canonical_duplicate_groups':duplicates,'cross_role_duplicate_groups':cross_role,
              'audit_summary_sha256':sha(args.audit/'audit_summary.json'),'records_sha256':sha(args.output/'records.json'),
              'archives':summary['archives'],'test_model_evaluated':False,'training_run':False}
    write_json(args.output/'eligibility.json',decision)
    with (args.output/'record_manifest.csv').open('w',newline='') as f:
        columns=['archive_member','role','nominal_subject_id','source_kind','single_action_binary_label','input_eligibility','packet_count','timestamp_span_seconds','raw_file_sha256']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':
    main()
