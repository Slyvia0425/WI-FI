"""Immutable nested development splits for the already-observed WiRe-Breath data."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .wire_cache import load_wire_breath_cache

SEED = 20261028
VERSION = 1
DEFAULT_PATH = Path('backend/modules/breathing_detection/evaluation/protocols/wire_breath_development_v1.json')


def fingerprint(samples):
    payload = json.dumps(samples, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def distribution(samples, indices):
    rows = [samples[i] for i in indices]
    return {
        'windows': len(rows),
        'recordings': len({row['recording_id'] for row in rows}),
        'assessable': sum(bool(row['assessable']) for row in rows),
        **{field: dict(sorted(Counter(str(row[field]) for row in rows).items()))
           for field in ('participant_id', 'room', 'environment', 'posture')},
    }


def validate_protocol(protocol, samples):
    if protocol.get('version') != VERSION:
        raise ValueError('Unsupported protocol version')
    if protocol['sample_fingerprint_sha256'] != fingerprint(samples):
        raise ValueError('Protocol does not match the current ordered cache metadata')
    if protocol.get('sample_ids') != [row['sample_id'] for row in samples]:
        raise ValueError('Protocol sample IDs do not match cache order')
    n = len(samples)
    all_ids = set(range(n))
    if len({row['sample_id'] for row in samples}) != n:
        raise ValueError('Duplicate sample IDs')
    subjects = {row['participant_id'] for row in samples}
    if set(protocol['folds']) != {f'subject_{subject}' for subject in subjects}:
        raise ValueError('Expected exactly one outer fold per participant')
    for name, fold in protocol['folds'].items():
        assignments = {}
        seen = set()
        for partition in ('train', 'validation', 'assessment'):
            indices = fold[partition]
            if not indices or len(indices) != len(set(indices)) or not set(indices) <= all_ids:
                raise ValueError(f'Invalid indices: {name}/{partition}')
            if seen & set(indices):
                raise ValueError(f'Window leakage in {name}')
            seen.update(indices)
            for index in indices:
                row = samples[index]
                recording = row['recording_id']
                if recording in assignments and assignments[recording] != partition:
                    raise ValueError(f'Recording leakage in {name}: {recording}')
                assignments[recording] = partition
                if (partition == 'assessment') != (row['participant_id'] == fold['held_out_subject']):
                    raise ValueError(f'Outer subject leakage in {name}')
        if seen != all_ids:
            raise ValueError(f'Incomplete fold {name}')


def build_protocol(samples):
    recordings = defaultdict(list)
    for i, row in enumerate(samples):
        recordings[row['recording_id']].append(i)
    for recording, indices in recordings.items():
        if len({tuple(samples[i][key] for key in ('participant_id', 'room', 'environment', 'posture')) for i in indices}) != 1:
            raise ValueError(f'Inconsistent recording metadata: {recording}')
    rng = np.random.default_rng(SEED)
    folds = {}
    for subject in sorted({row['participant_id'] for row in samples}):
        strata = defaultdict(list)
        development = []
        assessment = []
        for recording, indices in sorted(recordings.items()):
            row = samples[indices[0]]
            if row['participant_id'] == subject:
                assessment.extend(indices)
            else:
                development.extend(indices)
                strata[(row['participant_id'], row['environment'])].append(recording)
        # Fixed metadata-only search balances window counts and posture coverage.
        # Every subject/environment with >=2 records is represented on both sides.
        total_posture = Counter(samples[i]['posture'] for i in development)
        best = None
        for _ in range(1000):
            chosen = set()
            for records in strata.values():
                count = min(len(records) - 1, max(1, round(len(records) * .2))) if len(records) > 1 else 0
                chosen.update(rng.choice(records, count, replace=False).tolist())
            validation = sorted(i for recording in chosen for i in recordings[recording])
            posture = Counter(samples[i]['posture'] for i in validation)
            error = abs(len(validation) / len(development) - .2)
            error += np.mean([abs(posture[key] / count - .2) for key, count in total_posture.items()])
            # Prefer complete posture coverage on both sides when feasible.
            error += sum(posture[key] == 0 or posture[key] == count for key, count in total_posture.items())
            candidate = (float(error), tuple(sorted(chosen)), validation)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        validation = best[2]
        selected = set(validation)
        fold = {
            'held_out_subject': subject,
            'train': sorted(set(development) - selected),
            'validation': validation,
            'assessment': sorted(assessment),
        }
        fold['distribution'] = {key: distribution(samples, fold[key]) for key in ('train', 'validation', 'assessment')}
        folds[f'subject_{subject}'] = fold
    protocol = {
        'version': VERSION, 'seed': SEED, 'purpose': 'development_cross_validation_already_observed_data',
        'inner_validation_target': .2,
        'selection': '1000 metadata-only candidates; subject/environment-stratified recordings; window/posture balance with posture coverage penalty',
        'sample_fingerprint_sha256': fingerprint(samples),
        'sample_ids': [row['sample_id'] for row in samples],
        'folds': folds,
    }
    validate_protocol(protocol, samples)
    return protocol


def load_protocol(path, samples):
    protocol = json.loads(Path(path).read_text())
    validate_protocol(protocol, samples)
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, default=Path('backend/modules/breathing_detection/outputs/wire_breath_cache'))
    parser.add_argument('--output', type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    cache = load_wire_breath_cache(args.cache_dir, 1401)
    samples = cache.manifest['samples']
    protocol = build_protocol(samples)
    if args.output.exists():
        existing = load_protocol(args.output, samples)
        if existing != protocol:
            raise ValueError('Existing protocol differs; preserve v1 and create a reviewed new version')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({name: fold['distribution'] for name, fold in protocol['folds'].items()}, indent=2))


if __name__ == '__main__':
    main()
