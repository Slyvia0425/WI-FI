"""A.3 step 5: how long does one window take from the raw signal to the four class probabilities, and how big is what has to be shipped? One CPU thread.

Candidates (the models of split 0): the three frozen ones and the two with physical features. For every window of a random sample the building blocks are timed
one by one (statistics, MiniRocket transform, spectral shape, physical raw numbers, assembling the physical row with its dynamics), and the whole path of each candidate is
timed from the raw window to the probabilities. Reading the file is not included: the raw window is assumed to be in memory. Smoothing over windows is a few additions and is not timed.

The sample is 200 windows of the validation rows of split 0 (no label is used; the test rows are not touched). Each window is processed on its own, never in a batch, after a warm-up.
The first call of the MiniRocket transform (compiling) is reported separately as the cold start. Times are for this computer and one thread; compare the candidates with each other,
and the total with the 5 s between two windows.

The dynamics of the physical route are computed here the way a running system would, from the stored numbers of the last windows of the same run; the check at the start compares
them with the training code (preprocessing/physical.py) and stops if they differ.

Run from the repository root:
    backend/modules/motion_source_recognition/.venv/bin/python -m backend.modules.motion_source_recognition.evaluation.latency
Writes outputs/evaluation/latency.json and latency.md.
"""
import json
import pickle
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..preprocessing.config import EVALUATION, EVALUATION_DIR, MODELS, PREP_DIR, TRAINING, data_root
from ..preprocessing.features import N_CHANNELS, statistics_features
from ..preprocessing.physical import SPECTRAL, channel_numbers, dynamic_features, raw_numbers, spectral_shape, RAW_NAMES
from ..training.common import MODELS_DIR


SAMPLE, WARMUP, SEED = 200, 5, TRAINING["seed"]
CANDIDATES = {name: (m["label"], m["route"]) for name, m in MODELS.items()}   # model folder -> (label, feature path)
HISTORY = EVALUATION["latency_history_windows"]   # windows the dynamics look back


def dynamics_row(history: dict, last: int = HISTORY) -> list:
    """The 9 dynamics of the newest window from the stored numbers of the last windows of the run (oldest first, newest last), as preprocessing/physical.dynamic_features computes them.

    The 'change' numbers sum the absolute steps that end in the window, including the step into its first value, and divide by one less than the number of values: the same
    formula as the training code, which a deployed system has to repeat.
    """
    def roll(values, n):
        v = np.asarray(values[-(n + 1):], dtype=np.float64)       # n values plus the one before them, when it exists
        w = v[-n:]
        mean = w.mean()
        steps = np.abs(np.diff(v)).sum() if len(v) > 1 else 0.0
        return mean, w.std() / (mean + 1e-9), (steps / max(len(w) - 1, 1)) / (mean + 1e-9)
    _, cv6, ch6 = roll(history["std"], 6)
    _, cv24, ch24 = roll(history["std"], 24)
    return [cv6, ch6, roll(history["spec_share_0.5-1"], 6)[0], roll(history["spec_peak_to_mean"], 6)[0], cv24, ch24,
            roll(history["pca_share_1"], 6)[0], roll(history["ac_peak"], 6)[0], roll(history["permutation_entropy"], 6)[0]]


def physical_row(x: np.ndarray, stats: np.ndarray, history: dict, columns: list) -> tuple:
    """The 60 physical numbers of one window (features_physical_columns.json order) and the timing of its parts. `history` is updated with this window."""
    t0 = time.perf_counter_ns()
    spectral = spectral_shape(x[None]).reshape(1, len(SPECTRAL), N_CHANNELS).mean(axis=2)[0]
    t1 = time.perf_counter_ns()
    raw = raw_numbers(x[None])[0]
    t2 = time.perf_counter_ns()
    numbers = channel_numbers(stats).iloc[0].to_dict()
    numbers.update({"spec_" + n: v for n, v in zip(SPECTRAL, spectral)})
    numbers.update(dict(zip(RAW_NAMES, raw)))
    for key in history:
        history[key].append(numbers[key])
        del history[key][:-(HISTORY + 1)]
    dyn = dynamics_row(history)
    t3 = time.perf_counter_ns()
    names = ["dyn_std_cv", "dyn_std_change", "dyn_share_slow", "dyn_peak_to_mean", "dyn24_std_cv", "dyn24_std_change", "dyn6_pca_share_1", "dyn6_ac_peak", "dyn6_permutation_entropy"]
    numbers.update(dict(zip(names, dyn)))
    row = np.array([[numbers[c] for c in columns]], dtype=np.float32)
    return np.nan_to_num(row), (t1 - t0, t2 - t1, t3 - t2)


def check_dynamics(columns: list) -> None:
    """The streaming dynamics must equal the training code on a made-up run of 40 windows."""
    rng = np.random.default_rng(0)
    keys = ["std", "spec_share_0.5-1", "spec_peak_to_mean", "pca_share_1", "ac_peak", "permutation_entropy"]
    table = pd.DataFrame({k: rng.uniform(0.2, 2.0, 40) for k in keys})
    reference = dynamic_features(table, np.zeros(40, dtype=int), np.arange(40)).iloc[-1].to_numpy()
    history = {k: list(table[k].to_numpy()) for k in keys}
    np.testing.assert_allclose(dynamics_row(history), reference, rtol=1e-6, err_msg="the streaming dynamics differ from preprocessing/physical.py")


def summary(ns: list) -> dict:
    ms = np.asarray(ns, dtype=np.float64) / 1e6
    return {"median_ms": round(float(np.median(ms)), 2), "p95_ms": round(float(np.percentile(ms, 95)), 2), "mean_ms": round(float(ms.mean()), 2)}


def run() -> dict:
    import numba
    numba.set_num_threads(1)
    columns = json.loads((PREP_DIR / "features_physical_columns.json").read_text())
    check_dynamics(columns)
    manifest, splits = pd.read_csv(PREP_DIR / "manifest.csv"), pd.read_csv(PREP_DIR / "splits.csv")
    val_ids = splits.loc[splits["split_0"] == "val", "id"]
    sample = manifest[manifest["id"].isin(val_ids)].sample(SAMPLE + WARMUP, random_state=SEED)
    from ..preprocessing.windows import read_windows
    windows = read_windows([str(data_root() / p) for p in sample["file_path"]])
    transformer = pickle.loads((PREP_DIR / "minirocket_split_0.pkl").read_bytes())
    transformer.n_jobs = transformer._n_jobs = 1
    models = {name: joblib.load(MODELS_DIR / name / "split_0" / "model.joblib") for name in CANDIDATES}
    for name, model in models.items():
        classifier = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(classifier, "n_jobs"):
            classifier.n_jobs = 1
    history_keys = ["std", "spec_share_0.5-1", "spec_peak_to_mean", "pca_share_1", "ac_peak", "permutation_entropy"]
    blocks = {k: [] for k in ("statistics", "minirocket", "spectral_shape", "physical_raw_numbers", "assemble_with_dynamics")}
    totals = {name: [] for name in CANDIDATES}
    classify = {name: [] for name in CANDIDATES}
    cold = None
    history = {k: [] for k in history_keys}
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, x in enumerate(windows):
            measured = i >= WARMUP
            t = time.perf_counter_ns(); stats = statistics_features(x[None]); t_stats = time.perf_counter_ns() - t
            t = time.perf_counter_ns(); mr = transformer.transform(x[None]).astype(np.float32); t_mr = time.perf_counter_ns() - t
            if cold is None:
                cold = t_mr
            row, (t_spec, t_raw, t_asm) = physical_row(x, stats, history, columns)
            if measured:
                for key, v in zip(blocks, (t_stats, t_mr, t_spec, t_raw, t_asm)):
                    blocks[key].append(v)
            features = {"statistics": stats, "minirocket": mr, "statistics_physical": np.concatenate([stats, row], axis=1)}
            for name, (_, path) in CANDIDATES.items():
                t = time.perf_counter_ns(); models[name].predict_proba(features[path]); t_cls = time.perf_counter_ns() - t
                if measured:
                    classify[name].append(t_cls)
            if measured:   # whole paths, timed again from the raw window on
                for name, (_, path) in CANDIDATES.items():
                    h = {k: list(v) for k, v in history.items()}
                    t = time.perf_counter_ns()
                    s = statistics_features(x[None])
                    if path == "statistics":
                        f = s
                    elif path == "minirocket":
                        f = transformer.transform(x[None]).astype(np.float32)
                    else:
                        r, _ = physical_row(x, s, h, columns)
                        f = np.concatenate([s, r], axis=1)
                    models[name].predict_proba(f)
                    totals[name].append(time.perf_counter_ns() - t)
    sizes = {name: round((MODELS_DIR / name / "split_0" / "model.joblib").stat().st_size / 1e6, 2) for name in CANDIDATES}
    transformer_mb = round((PREP_DIR / "minirocket_split_0.pkl").stat().st_size / 1e6, 2)
    result = {"sample_windows": SAMPLE, "threads": 1, "blocks_ms": {k: summary(v) for k, v in blocks.items()}, "minirocket_cold_start_ms": round(cold / 1e6, 1),
              "candidates": {name: {"label": CANDIDATES[name][0], "features": CANDIDATES[name][1], "classify_ms": summary(classify[name]), "total_ms": summary(totals[name]),
                                    "share_of_the_5_s_between_windows_percent": round(summary(totals[name])["median_ms"] / 5000 * 100, 2), "model_megabytes": sizes[name],
                                    "also_ship_megabytes": transformer_mb if CANDIDATES[name][1] == "minirocket" else 0.0} for name in CANDIDATES}}
    return result


def markdown(r: dict) -> str:
    L = [f"# Inference time of one window, one CPU thread ({r['sample_windows']} validation windows of split 0)\n", "| Candidate | Features | Classify (median ms) | Whole path: median ms | 95th percentile ms | Share of the 5 s | Model MB | Also ship MB |", "|---|---|---|---|---|---|---|---|"]
    for c in r["candidates"].values():
        L.append(f"| {c['label']} | {c['features']} | {c['classify_ms']['median_ms']} | {c['total_ms']['median_ms']} | {c['total_ms']['p95_ms']} | {c['share_of_the_5_s_between_windows_percent']}% | {c['model_megabytes']} | {c['also_ship_megabytes']} |")
    L += ["", "Building blocks (median / 95th percentile, ms):", "", "| Block | Median | 95th |", "|---|---|---|"]
    L += [f"| {k} | {v['median_ms']} | {v['p95_ms']} |" for k, v in r["blocks_ms"].items()]
    L += ["", f"MiniRocket first call (cold start, compiling): {r['minirocket_cold_start_ms']} ms. Reading the file and the memory in use at run time are not measured."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    out = run()
    (EVALUATION_DIR / "latency.json").write_text(json.dumps(out, indent=1))
    (EVALUATION_DIR / "latency.md").write_text(markdown(out))
    print(markdown(out))
