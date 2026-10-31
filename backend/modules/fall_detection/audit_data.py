"""Audit actual FallDetection HDF5 content and official ID splits, without changing either."""
import argparse
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path

import h5py
import numpy as np

FIELDS = ['sample_id', 'file_path', 'original_label', 'binary_label', 'split',
          'device', 'user', 'environment', 'session', 'h5_key', 'shape', 'dtype',
          'finite_ratio', 'content_hash', 'status']


def digest(data):
    # Comparable logical numeric data, independent of HDF5 compression and source byte order.
    # Preserve shape; use float64 for real and paired float64 for complex values.
    a = np.asarray(data, dtype='<c16' if np.iscomplexobj(data) else '<f8').copy(order='C')
    a[a == 0] = 0  # canonicalize signed zero
    if np.iscomplexobj(a):
        parts = a.view('<f8')
        parts[np.isnan(parts)] = np.nan
    else:
        a[np.isnan(a)] = np.nan
    return hashlib.sha256(json.dumps(list(a.shape)).encode() + a.dtype.str.encode() + a.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='FallDetection task root')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-version', default='unknown')
    args = parser.parse_args()
    root = args.data_root.resolve()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (root / 'metadata/sample_metadata.csv').open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    mapping = json.loads((root / 'metadata/label_mapping.json').read_text())
    # Explicit semantic names, never reinterpret official numeric labels as our binary target.
    if set(mapping['label_to_idx']) != {'Fall', 'Nonfall'}:
        raise ValueError('Unverified label vocabulary; review mapping before converting')
    binary = {'Fall': 1, 'Nonfall': 0}
    splits = {}
    issues = []
    for path in sorted((root / 'splits').glob('*.json')):
        value = json.loads(path.read_text())
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError(f'{path}: expected inspected list of string sample IDs')
        splits[path.stem] = value
        if len(value) != len(set(value)):
            issues.append({'type': 'duplicate_ids_in_split', 'split': path.stem})
    for required in ('train_id', 'val_id', 'test_id'):
        if required not in splits:
            raise ValueError(f'Missing required official split: {required}')
    ids = collections.Counter(r['id'] for r in rows)
    for sid, count in ids.items():
        if count > 1:
            issues.append({'type': 'duplicate_metadata_id', 'sample_id': sid, 'count': count})
    for name, members in splits.items():
        for sid in set(members) - ids.keys():
            issues.append({'type': 'split_id_without_metadata', 'split': name, 'sample_id': sid})
    memberships = collections.defaultdict(list)
    for name, members in splits.items():
        for sid in set(members):
            memberships[sid].append(name)
    manifests = []
    h5_inventory = []
    for i, row in enumerate(rows, 1):
        sid = row['id']
        path = (root / row['file_path']).resolve()
        m = dict.fromkeys(FIELDS, 'unknown')
        m.update(sample_id=sid, file_path=str(path), original_label=row['label'],
                 binary_label=binary.get(row['label'], 'unknown'),
                 split=';'.join(memberships[sid]) or 'unassigned', status='ok')
        for key in ('device', 'user', 'environment', 'session'):
            m[key] = row.get(key) or 'unknown'
        if m['binary_label'] == 'unknown':
            issues.append({'type': 'unmatched_label', 'sample_id': sid, 'label': row['label']})
        if not memberships[sid]:
            issues.append({'type': 'unassigned_sample', 'sample_id': sid})
        try:
            with h5py.File(path, 'r') as handle:
                datasets = []
                handle.visititems(lambda key, obj: datasets.append({
                    'key': key, 'shape': list(obj.shape), 'dtype': str(obj.dtype),
                    'attrs': {k: str(v) for k, v in obj.attrs.items()}
                }) if isinstance(obj, h5py.Dataset) else None)
                h5_inventory.append({'sample_id': sid, 'datasets': datasets,
                                     'attrs': {k: str(v) for k, v in handle.attrs.items()}})
                data = handle['CSI_amps'][()]
                if not np.issubdtype(data.dtype, np.number) or not data.size:
                    raise ValueError('Empty or nonnumeric CSI_amps')
                m.update(h5_key='CSI_amps', shape=json.dumps(list(data.shape)), dtype=str(data.dtype),
                         finite_ratio=float(np.isfinite(data).mean()), content_hash=digest(data))
                if m['finite_ratio'] < 1:
                    issues.append({'type': 'nonfinite_data', 'sample_id': sid,
                                   'nan': int(np.isnan(data).sum()), 'inf': int(np.isinf(data).sum())})
        except Exception as exc:
            m['status'] = 'error'
            issues.append({'type': 'file_error', 'sample_id': sid, 'file': str(path),
                           'error': f'{type(exc).__name__}: {exc}'})
        manifests.append(m)
        if i % 1000 == 0:
            print(f'Audited {i}/{len(rows)}', flush=True)
    with (out / 'sample_manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(manifests)
    sets = {k: set(v) for k, v in splits.items()}
    summary = {}
    for name, members in sets.items():
        selected = [m for m in manifests if m['sample_id'] in members]
        labels = collections.Counter(m['original_label'] for m in selected)
        summary[name] = {'n': len(selected), 'unique_split_ids': len(members), 'labels': dict(labels),
                         'fall_ratio': labels['Fall'] / len(selected) if selected else None,
                         'devices': dict(collections.Counter(m['device'] for m in selected))}
    overlaps = [{'a': a, 'b': b, 'count': len(sets[a] & sets[b])}
                for a, b in itertools.combinations(sets, 2)]
    hashes = collections.defaultdict(list)
    for m in manifests:
        if m['status'] == 'ok':
            hashes[m['content_hash']].append(m)
    duplicate_groups = []
    for h, group in hashes.items():
        if len(group) < 2:
            continue
        duplicate_groups.append({'hash': h, 'records': [{k: m[k] for k in (
            'sample_id', 'original_label', 'split', 'device', 'user', 'environment', 'session')}
            for m in group], 'interpretation': 'Equal CSI arrays; recording provenance needs review, not automatically leakage'})
    # Session identifiers are not documented as globally unique. Include all known provenance.
    source_groups = collections.defaultdict(list)
    for m in manifests:
        source_groups[tuple(m[k] for k in ('user', 'environment', 'device', 'session'))].append(m)
    source_cross = []
    for key, group in source_groups.items():
        primary = {s for m in group for s in m['split'].split(';') if s in ('train_id', 'val_id', 'test_id')}
        if len(primary) > 1:
            source_cross.append({'source_key': key, 'primary_splits': sorted(primary),
                                 'ids': [m['sample_id'] for m in group]})
    duplicate_cross = []
    for a, b in itertools.combinations(sets, 2):
        shared = [h for h, group in hashes.items()
                  if any(m['sample_id'] in sets[a] for m in group)
                  and any(m['sample_id'] in sets[b] for m in group)]
        duplicate_cross.append({'a': a, 'b': b, 'shared_content_hashes': len(shared)})
    version_files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in ('metadata', 'splits') for p in sorted((root / folder).glob('*')) if p.is_file()}
    result = {
        'data_root': str(root), 'data_version': args.data_version, 'metadata_split_sha256': version_files,
        'official_label_mapping': mapping, 'internal_label_mapping': binary,
        'records': len(rows), 'splits': summary, 'id_overlaps': overlaps,
        'devices': dict(collections.Counter(m['device'] for m in manifests)),
        'device_label_counts': [{'device': d, 'original_label': l, 'count': n}
                                for (d, l), n in sorted(collections.Counter(
                                    (m['device'], m['original_label']) for m in manifests).items())],
        'shapes': dict(collections.Counter(m['shape'] for m in manifests if m['status'] == 'ok')),
        'dtypes': dict(collections.Counter(m['dtype'] for m in manifests if m['status'] == 'ok')),
        'issues': issues, 'issue_counts': dict(collections.Counter(i['type'] for i in issues)),
        'duplicate_content_groups': duplicate_groups, 'content_overlap_by_split': duplicate_cross,
        'data_content_sha256': hashlib.sha256(json.dumps(sorted((m['sample_id'], m['content_hash']) for m in manifests)).encode()).hexdigest(),
        'candidate_source_key_collisions_across_primary_splits': source_cross,
        'source_limit': 'Composite user/environment/device/session only. No verified parent recording ID; adjacent session numbers do not establish a continuous recording.',
        'hash_representation': 'shape + canonical little endian float64 (complex128 if complex); C-order; signed zeros and NaNs normalized; no amplitude conversion or normalization',
    }
    (out / 'audit_details.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (out / 'hdf5_inventory.json').write_text(json.dumps(h5_inventory, ensure_ascii=False, indent=2))
    lines = ['# FallDetection 官方数据审计', '', f'版本：{args.data_version}；样本数：{len(rows)}。',
             '', '官方数字 Fall=0、Nonfall=1；内部二分类 Fall=1、Non-fall=0，保留原始字符串。',
             '', '## 官方集合', '', '|集合|样本数|Fall|Nonfall|Fall 比例|', '|---|---:|---:|---:|---:|']
    for name, s in summary.items():
        lines.append(f"|{name}|{s['n']}|{s['labels'].get('Fall', 0)}|{s['labels'].get('Nonfall', 0)}|{s['fall_ratio']:.2%}|")
    lines += ['', 'split 文件实际为字符串 sample ID 的 JSON 列表。manifest 的 split 使用分号保存所有成员关系，不把重叠测试集算成互斥划分。',
              '', '## 设备与标签', '', '|设备|原始标签|数量|', '|---|---|---:|']
    for entry in result['device_label_counts']:
        lines.append(f"|{entry['device']}|{entry['original_label']}|{entry['count']}|")
    lines += ['', '## 全量输入形状', '', '```json', json.dumps(result['shapes'], indent=2), '```',
              '', f"dtype：{result['dtypes']}。", '', '## 集合间 ID 交集', '', '|集合 A|集合 B|交集|', '|---|---|---:|']
    for entry in overlaps:
        lines.append(f"|{entry['a']}|{entry['b']}|{entry['count']}|")
    lines += ['', '## 集合间内容交集', '', '|集合 A|集合 B|共同内容哈希数|', '|---|---|---:|']
    for entry in duplicate_cross:
        lines.append(f"|{entry['a']}|{entry['b']}|{entry['shared_content_hashes']}|")
    lines += ['', '## 数据质量与来源', '', f'异常项：{len(issues)}；重复内容组：{len(duplicate_groups)}；候选来源字段组合跨主集合组：{len(source_cross)}。',
              '', '每项异常与每个重复组的样本、标签和来源见 audit_details.json；未静默跳过文件，未修改官方划分。',
              '', '内容相同只证明数组一致，须进一步核查录制来源，不能直接把所有重复组认定为泄漏。',
              '', '同源检查使用 user/environment/device/session 组合。缺少经过确认的连续录制父 ID；不能证明相邻片段已按父录制隔离。',
              '', '哈希使用统一 float64（复数使用 complex128）、小端、C 顺序及 shape，并统一符号零和 NaN；不用 HDF5 文件字节哈希判定内容重复。',
              '', '## 边界', '', '本审计不训练模型、不选择阈值、不报告性能。轴语义证据与审计解释见 inspection_notes.md。']
    lines += ['', '## 当前判断', '',
              '文件可用性检查完成；不能据此宣称官方划分没有泄漏风险。跨集合重复内容计数见上表；本次人工重新读取与来源核查见 duplicate_review.md。',
              '', '来源字段组合碰撞可能来自活动内 session 编号重复使用，不能直接作为同源录制的定论。',
              '', '后续官方实验保留现有划分并披露风险；如另行进行按内容/经核实父录制分组的纠正实验，必须另存划分和结果，不能称为官方基准。当前没有生成纠正划分。',
              '', '各测试子集设备组成见 audit_details.json；难度与设备可能混杂，解读难度分层结果时必须一并检查。']
    (out / 'data_audit.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: result[k] for k in ('records', 'shapes', 'devices')}, indent=2))
    print('issues', len(issues), 'duplicate_groups', len(duplicate_groups), 'source_cross', len(source_cross))
    if any(i['type'] in ('file_error', 'unmatched_label', 'split_id_without_metadata', 'nonfinite_data', 'duplicate_metadata_id') for i in issues):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
