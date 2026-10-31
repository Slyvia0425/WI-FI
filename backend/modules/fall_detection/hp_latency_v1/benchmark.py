"""HP-only batch-one timing; does not train, tune thresholds, or evaluate tests."""
from time import perf_counter_ns
PROCESS_CODE_START = perf_counter_ns()

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

MODULE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE / 'final_selection_v1'))
from runtime import load_candidate, load_frozen, read_amplitude, transform_file

CHECKPOINTS = {
    'original_1d_cnn': MODULE / 'outputs/final_selection_v1/final_research_model.pt',
    'official_resnet18': MODULE / 'outputs/official_resnet18_seed42_recovery_v1/reproduction_bundle.pt',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def hp_rows():
    manifest = MODULE / 'reports/sample_manifest.csv'
    with manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for label in ('0', '1'):
        matches = sorted((r for r in rows if r['device'] == 'HP'
                          and 'val_id' in r['split'].split(';')
                          and r['binary_label'] == label), key=lambda r: r['sample_id'])
        if len(matches) < 8:
            raise ValueError('Need eight audited HP validation samples of each class')
        selected.extend(matches[:8])
    for row in selected:
        if row['status'] != 'ok' or float(row['finite_ratio']) != 1:
            raise ValueError('Audited sample failed')
    return selected


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def load_model(kind, device):
    checkpoint = CHECKPOINTS[kind]
    if kind == 'original_1d_cnn':
        model, state, loaded_kind = load_frozen(checkpoint, device)
        if loaded_kind != kind:
            raise ValueError('Wrong frozen model')
    else:
        receipt = json.loads((MODULE / 'reports/final_selection_v1_resnet_seed42_cpu/receipt.json').read_text())
        if sha(checkpoint) != receipt['checkpoint_sha256']:
            raise ValueError('ResNet checkpoint hash mismatch')
        model, state = load_candidate(checkpoint, kind, device)
    sync(device)
    return model, state


def make_preprocessor(state, kind):
    if kind == 'original_1d_cnn':
        cfg = state['preprocess_config']
        stats = cfg['devices']['HP']
        mean = np.array(stats['mean'])[:, None]
        scale = np.array(stats['scale'])[:, None]

        def preprocess(raw):
            if raw.shape != (232, 500) or not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError('Invalid HP amplitude')
            output = np.zeros((232, 500), dtype=np.float32)
            output[:] = (raw - mean) / scale
            if not np.isfinite(output).all():
                raise ValueError('Nonfinite normalized data')
            return torch.from_numpy(output)
    else:
        def preprocess(raw):
            if raw.shape != (232, 500) or not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError('Invalid HP amplitude')
            x = torch.from_numpy(raw.astype(np.float32)).transpose(0, 1).unsqueeze(0)
            normalized = (x - x.mean(dim=(1, 2), keepdim=True)) / x.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            result = torch.zeros((1, 500, 232), dtype=torch.float32)
            result[:] = normalized
            if not torch.isfinite(result).all():
                raise ValueError('Nonfinite normalized data')
            return result
    return preprocess


def score_output(output, kind):
    return (output.reshape(-1).sigmoid()[0] if kind == 'original_1d_cnn'
            else output.softmax(1)[0, 0]).item()


@torch.inference_mode()
def worker(args):
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no silent CPU fallback')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows = hp_rows()
    load_start = perf_counter_ns()
    model, state = load_model(args.kind, args.device)
    load_ms = (perf_counter_ns() - load_start) / 1e6
    threshold = float(state['selected_threshold']) if args.kind == 'original_1d_cnn' else 0.5

    def file_pipeline(i):
        x = transform_file(rows[i % len(rows)]['file_path'], 'HP', state, args.kind)
        output = model(x.unsqueeze(0).to(args.device))
        score = score_output(output, args.kind)
        return score, score >= threshold

    first_start = perf_counter_ns()
    first_score, first_prediction = file_pipeline(0)
    sync(args.device)
    first_ms = (perf_counter_ns() - first_start) / 1e6
    base = {
        'kind': args.kind, 'backend': args.device, 'seed': int(state['seed']),
        'checkpoint': str(CHECKPOINTS[args.kind]), 'checkpoint_sha256': sha(CHECKPOINTS[args.kind]),
        'threshold_used_for_timing_only': threshold, 'batch_size': 1,
        'load_model_and_integrity_check_ms': load_ms, 'first_file_pipeline_ms': first_ms,
        'process_code_to_first_result_ms': (perf_counter_ns() - PROCESS_CODE_START) / 1e6,
        'first_fall_score': first_score, 'first_prediction_is_fall': first_prediction,
    }
    if args.cold:
        write_json(args.output, base)
        return

    raw = [read_amplitude(row['file_path']) for row in rows]
    preprocess = make_preprocessor(state, args.kind)
    tensors = [preprocess(x) for x in raw]
    adaptation_error = 0.0
    for row, tensor in zip(rows, tensors):
        expected = transform_file(row['file_path'], 'HP', state, args.kind)
        adaptation_error = max(adaptation_error, float((expected - tensor).abs().max()))
    if adaptation_error != 0.0:
        raise ValueError(f'In-memory preprocessing differs from established runtime: {adaptation_error}')
    resident = [x.unsqueeze(0).to(args.device) for x in tensors]
    sync(args.device)
    source_kind = 'original' if args.kind == 'original_1d_cnn' else 'resnet'
    with (MODULE / f'reports/final_selection_v1_{source_kind}_seed42_cpu/validation_predictions.csv').open(newline='') as stream:
        reference = {r['sample_id']: float(r['fall_score']) for r in csv.DictReader(stream)}
    errors = [abs(score_output(model(x), args.kind) - reference[row['sample_id']])
              for row, x in zip(rows, resident)]
    if max(errors) > 1e-4:
        raise ValueError('Backend score differs excessively from saved CPU validation scores')

    def forward_only(i):
        return model(resident[i % len(resident)])

    def memory_pipeline(i):
        x = preprocess(raw[i % len(raw)]).unsqueeze(0).to(args.device)
        score = score_output(model(x), args.kind)
        return score, score >= threshold

    records, summaries = [], []
    functions = {'forward_only': forward_only, 'memory_pipeline': memory_pipeline,
                 'file_pipeline_warm_os_cache': file_pipeline}
    for scope, fn in functions.items():
        for i in range(args.warmup):
            fn(i)
        sync(args.device)
        for trial in range(args.trials):
            values = []
            for i in range(args.iterations):
                sync(args.device)
                start = perf_counter_ns()
                fn(i)
                sync(args.device)
                elapsed = (perf_counter_ns() - start) / 1e6
                values.append(elapsed)
                records.append({'kind': args.kind, 'backend': args.device, 'scope': scope,
                                'trial': trial + 1, 'iteration': i + 1,
                                'sample_id': rows[i % len(rows)]['sample_id'], 'milliseconds': elapsed})
            summaries.append({'scope': scope, 'trial': trial + 1, 'n': len(values),
                              'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95)),
                              'p99_ms': float(np.percentile(values, 99))})
            print(json.dumps({'kind': args.kind, 'backend': args.device, **summaries[-1]}), flush=True)
    combined = []
    for scope in functions:
        values = [r['milliseconds'] for r in records if r['scope'] == scope]
        combined.append({'scope': scope, 'n': len(values), 'median_ms': float(np.median(values)),
                         'p95_ms': float(np.percentile(values, 95)), 'p99_ms': float(np.percentile(values, 99)),
                         'min_ms': min(values), 'max_ms': max(values)})
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    base.update({'scopes': combined, 'trials': summaries, 'preprocess_max_abs_difference': adaptation_error,
                 'backend_score_max_abs_difference_vs_cpu': max(errors),
                 'parameter_count': sum(p.numel() for p in model.parameters()),
                 'parameter_and_buffer_bytes': sum(p.numel() * p.element_size() for p in model.state_dict().values()),
                 'checkpoint_file_bytes': CHECKPOINTS[args.kind].stat().st_size,
                 'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'input_shape': list(resident[0].shape), 'dtype': str(resident[0].dtype),
                 'sample_ids': [r['sample_id'] for r in rows], 'test_evaluated': False,
                 'completed_utc': datetime.now(timezone.utc).isoformat()})
    write_json(args.output, base)


def orchestrate(args):
    out = args.output
    if out.exists():
        raise FileExistsError('Use a new output directory; prior results are preserved')
    out.mkdir(parents=True)
    sources = [MODULE / p for p in ('models/cnn.py', 'preprocess.py', 'final_selection_v1/runtime.py',
               'final_selection_v1/infer.py', 'outputs/final_selection_v1/final_research_model.json')]
    sources += list(CHECKPOINTS.values())
    before = {str(p): sha(p) for p in sources}
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'HP validation samples only; latency only',
        'models': list(CHECKPOINTS), 'backends': ['cpu', 'mps'], 'batch_size': 1,
        'cpu_threads': 2, 'interop_threads': 1, 'warmup_per_scope': args.warmup,
        'trials': args.trials, 'iterations_per_trial_per_scope': args.iterations,
        'new_process_startup_trials': 3, 'sample_selection': 'first eight sorted HP val_id IDs per class',
        'manifest_sha256': sha(MODULE / 'reports/sample_manifest.csv'),
        'source_and_artifact_sha256_before': before, 'benchmark_source_sha256': sha(__file__),
        'python': platform.python_version(), 'platform': platform.platform(), 'torch': torch.__version__,
        'numpy': np.__version__, 'machine': platform.machine(),
        'hardware': {k: subprocess.check_output(['sysctl', '-n', k], text=True).strip()
                     for k in ('hw.model', 'machdep.cpu.brand_string', 'hw.memsize', 'hw.ncpu')},
        'notes': ['Backends run sequentially; FP32, eval and inference_mode; no compilation/quantization.',
                  'Forward-only input already on device; scoring/transfer excluded, synchronization included.',
                  'Memory pipeline begins with parsed amplitude in RAM; includes preprocessing, transfer, scoring and local threshold.',
                  'File pipeline uses exact existing runtime, OS file cache warm; no guaranteed physical disk read.',
                  'New-process startup is not a power-on/cold-disk experiment.',
                  'Peak RSS is whole Python process including imports/loading; not isolated model memory and not GPU allocation.',
                  'No acquisition wait, rolling-window wait, alarm confirmation, network/UI delivery or event latency measured.',
                  'No ESP32 records, training, threshold changes, test evaluation or continuous-monitoring claims.'],
    }
    write_json(out / 'protocol.json', protocol)
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'VECLIB_MAXIMUM_THREADS': '2',
           'PYTORCH_ENABLE_MPS_FALLBACK': '0'}
    runs, cold = [], []
    for backend in ('cpu', 'mps'):
        for kind in CHECKPOINTS:
            path = out / f'{kind}_{backend}.json'
            common = [sys.executable, str(Path(__file__).resolve()), '--worker', '--kind', kind,
                      '--device', backend, '--warmup', str(args.warmup), '--trials', str(args.trials),
                      '--iterations', str(args.iterations)]
            subprocess.run(common + ['--output', str(path)], check=True, env=env)
            runs.append(json.loads(path.read_text()))
            for trial in range(1, 4):
                cold_path = out / f'{kind}_{backend}_startup_{trial}.json'
                start = perf_counter_ns()
                subprocess.run(common + ['--cold', '--output', str(cold_path)], check=True, env=env)
                elapsed = (perf_counter_ns() - start) / 1e6
                data = json.loads(cold_path.read_text())
                data.update({'trial': trial, 'parent_process_wall_through_exit_ms': elapsed})
                write_json(cold_path, data)
                cold.append(data)
    after = {p: sha(p) for p in before}
    if before != after:
        raise ValueError('Protected model/source artifact changed')
    write_json(out / 'results.json', {'protocol': protocol, 'runs': runs, 'startup': cold,
                                    'protected_artifacts_unchanged': True, 'completed': True})
    print('HP latency benchmark complete', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker', action='store_true'); ap.add_argument('--cold', action='store_true')
    ap.add_argument('--kind', choices=list(CHECKPOINTS)); ap.add_argument('--device', choices=['cpu', 'mps'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--warmup', type=int, default=30); ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--iterations', type=int, default=200)
    args = ap.parse_args()
    if args.worker:
        worker(args)
    else:
        orchestrate(args)


if __name__ == '__main__':
    main()
