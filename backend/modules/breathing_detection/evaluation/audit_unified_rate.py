"""Verify fixed-rate integration against preserved original streaming artifacts."""
import json
from pathlib import Path
from ..preprocessing.unified_rate import ESTIMATOR_ID

ROOT = Path('backend/modules/breathing_detection/outputs')


def main():
    target = ROOT / 'respiratory_rate_unified_v1/integration_audit.json'
    if target.exists():
        raise FileExistsError('Preserve integration audit')
    old = json.loads((ROOT / 'streaming_validation_v2/stream_events.json').read_text())
    new = json.loads((ROOT / 'streaming_validation_unified_v1/stream_events.json').read_text())
    if len(old) != len(new) or len(new) != 467:
        raise ValueError('Segment count changed')
    unchanged, changed_rates, available = 0, 0, 0
    for before, after in zip(old, new):
        if before['sample_id'] != after['sample_id'] or len(before['events']) != len(after['events']):
            raise ValueError('Replay alignment changed')
        for a,b in zip(before['events'], after['events']):
            if any(a.get(k) != b.get(k) for k in ('generation','window_start','window_end','status','reason','breathing_probability','temporal_instability')):
                raise ValueError('Breathing detection or quality decision changed')
            if b.get('rate_estimator_id') != ESTIMATOR_ID:
                raise ValueError('Mixed estimator versions')
            unchanged += 1
            if (a['rate_bpm'] is None) != (b['rate_bpm'] is None):
                raise ValueError('Rate availability changed')
            if a['rate_bpm'] is not None:
                available += 1
                changed_rates += a['rate_bpm'] != b['rate_bpm']
        if after['after_timeout']['rate_bpm'] is not None or not after['after_timeout']['last_result_stale']:
            # Some files never generated a result? This dataset has at least one
            # window per segment, so every terminal result must be historical.
            raise ValueError('Timeout did not expire current evidence')
    payload = json.loads((ROOT / 'replay.json').read_text())
    pairs = 0
    for record in payload['records']:
        for sample in record['samples']:
            for event in sample['streaming']['recorded']['events']:
                if event.get('rate_bpm') is not None:
                    if event['rate_bpm'] != sample['candidate_bpm']:
                        raise ValueError('Research and monitoring rate differ')
                    pairs += 1
    if payload['rate_estimator']['id'] != ESTIMATOR_ID or len(payload['csi_bench']['scenes']) != 2:
        raise ValueError('Dashboard estimator/CSI experience mismatch')
    result = dict(estimator_id=ESTIMATOR_ID, segments=467, unchanged_detection_events=unchanged,
                  available_stream_rates=available, changed_stream_rates=changed_rates,
                  timeout_checks_passed=467, curated_exact_rate_pairs=pairs,
                  original_streaming_artifacts_preserved=True, csi_scenarios_preserved=True)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
