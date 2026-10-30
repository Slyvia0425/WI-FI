"""Physically motivated features of the CSI window: the "physical" feature route (60 numbers per window).

CSI here has amplitude only (no phase), so Doppler speed and angle of arrival cannot be computed. What can be read from the amplitude of a window (232 channels x 500 samples
at 100 Hz) are ratios and shapes that do not depend on signal size or level:

  from the statistics features, averaged over the channels (features_stats.npy): motion size (std, mean absolute change), dominant frequency, low-band share, range over std,
      skewness, kurtosis; how much the motion size varies across channels and across the 4 antenna links; spread of the dominant frequency across channels and the share of channels
      that agree on it
  spectral shape of the channels, averaged: share of the power in 0.5-1, 1-2, 2-4, 4-8, 8-20 Hz, entropy, flatness, peak-to-mean ratio, centroid, spread
  how the motion is spread over the channels (a large body or a moving object changes many subcarriers and links together):
      pca_share_1/2/3     share of the variance in the strongest 3 principal components of the 232 channels
      pca_effective_rank  participation ratio of the eigenvalues (1 = one common pattern, 232 = independent channels)
      pca_n90             components needed for 90% of the variance
      link_coherence_mean, link_coherence_min   correlation between the mean signals of the 4 antenna links
      adjacent_subcarrier_corr                  correlation between neighbouring subcarriers
  the time course of the strongest component (PC1; the same signal a person-sensing system would use):
      ac_peak, ac_lag     height and lag (0.2 to 2.5 s) of the largest autocorrelation peak: periodic motion such as steps, wheels, a rotating blade
      hjorth_mobility, hjorth_complexity, zero_crossings_per_s, permutation_entropy   how fast, how regular, how predictable
      rolloff85, centroid, share_20_35, share_35_50   where the spectrum lies, including 20 to 50 Hz where motors and blades vibrate
      duty_cycle, burstiness, block_cv   whether the motion is continuous or in bursts (10 blocks of 0.5 s)
  the whole signal: global_share_20_35, global_share_35_50, global_median_freq
  how the frequency content changes during the window (PC1, four 2-second frames): stft_centroid_std, stft_peak_freq_std, spectral_flux
  and how quickly the signal forgets itself: decorrelation_time (lag where the autocorrelation first falls below 1/e), pc1_skew, pc1_kurtosis

  across the recent windows of the same run (the current window included, never the future; a run ends where the session number jumps by more than 5):
      variation and mean absolute change of the motion size over 6 and 24 windows, mean 0.5-1 Hz share, mean spectral peak-to-mean ratio, and mean PC1 share, autocorrelation
      peak and permutation entropy over 6 windows

Steps (from the repository root, with the module's Python; the raw pass takes about 4 minutes):
    python -m backend.modules.motion_source_recognition.preprocessing.physical            # raw pass, then assembly
    python -m backend.modules.motion_source_recognition.preprocessing.physical --assemble # assembly only (needs features_stats.npy and the raw pass)

Writes outputs/prep/physical_raw.npy (30 raw numbers and 10 channel-averaged spectral numbers per window), features_physical.npy (the 60 numbers) and features_physical_columns.json,
rows in the order of manifest.csv. Only the CSI signal and the order of the windows are used: user, activity, environment and device never enter the features.
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .config import DATA, PREP_DIR, data_root
from .features import N_CHANNELS, SAMPLING_RATE_HZ
from .windows import read_windows

LINKS = DATA["antenna_links"]
PER_LINK = N_CHANNELS // LINKS
BAND_EDGES = [0.5, 1.0, 2.0, 4.0, 8.0, 20.0]
SPECTRAL = ["share_0.5-1", "share_1-2", "share_2-4", "share_4-8", "share_8-20", "entropy", "flatness", "peak_to_mean", "centroid", "spread"]
RAW_NAMES = ["pca_share_1", "pca_share_2", "pca_share_3", "pca_effective_rank", "pca_n90", "link_coherence_mean", "link_coherence_min", "adjacent_subcarrier_corr",
             "ac_peak", "ac_lag", "hjorth_mobility", "hjorth_complexity", "zero_crossings_per_s", "permutation_entropy", "rolloff85", "centroid", "share_20_35", "share_35_50",
             "duty_cycle", "burstiness", "block_cv", "global_share_20_35", "global_share_35_50", "global_median_freq",
             "stft_centroid_std", "stft_peak_freq_std", "spectral_flux", "decorrelation_time", "pc1_skew", "pc1_kurtosis"]
RAW_NAMES_V2 = RAW_NAMES[:24]   # the first 24 were evaluated first; the last 6 were added afterwards
EPS = 1e-12
RAW_FILE, FEATURES_FILE, COLUMNS_FILE = PREP_DIR / "physical_raw.npy", PREP_DIR / "features_physical.npy", PREP_DIR / "features_physical_columns.json"
RUN_GAP = DATA["run_gap"]   # a jump of the session number larger than this ends a run

def spectral_shape(x: np.ndarray) -> np.ndarray:
    """x: (samples, signals, time) -> (samples, 10 * signals), statistic-major (all signals of the first number, then the second, ...)."""
    x = x.astype(np.float32, copy=False)
    centred = x - x.mean(axis=2, keepdims=True)
    power = np.abs(np.fft.rfft(centred, axis=2)).astype(np.float32) ** 2
    freqs = np.fft.rfftfreq(x.shape[2], d=1.0 / SAMPLING_RATE_HZ)
    band = (freqs >= BAND_EDGES[0]) & (freqs < BAND_EDGES[-1])
    p, f = power[:, :, band], freqs[band]
    total = p.sum(axis=2, keepdims=True)
    with np.errstate(all="ignore"):   # a flat signal has no power in the band; its numbers become 0
        shares = [p[:, :, (f >= lo) & (f < hi)].sum(axis=2) / total[:, :, 0] for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
        q = p / total
        entropy = -(q * np.log(q + EPS)).sum(axis=2) / np.log(len(f))
        flatness = np.exp(np.log(p + EPS).mean(axis=2)) / (p.mean(axis=2) + EPS)
        peak = p.max(axis=2) / (p.mean(axis=2) + EPS)
        centroid = (q * f).sum(axis=2)
        spread = np.sqrt((q * (f - centroid[:, :, None]) ** 2).sum(axis=2))
    out = np.stack(shares + [entropy, flatness, peak, centroid, spread], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).reshape(len(x), -1).astype(np.float32)



def _spectrum(series: np.ndarray) -> tuple:
    power = np.abs(np.fft.rfft(series - series.mean(axis=-1, keepdims=True), axis=-1)) ** 2
    return power, np.fft.rfftfreq(series.shape[-1], d=1.0 / SAMPLING_RATE_HZ)


def _hjorth_mobility(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.diff(x, axis=-1).var(axis=-1) / (x.var(axis=-1) + EPS))


def _permutation_entropy(x: np.ndarray) -> np.ndarray:
    a, b, c = x[:, :-2], x[:, 1:-1], x[:, 2:]
    code = (a < b) * 4 + (b < c) * 2 + (a < c) * 1   # the ordering pattern of 3 consecutive samples (6 of the 8 codes occur)
    counts = np.stack([(code == k).sum(axis=1) for k in range(8)], axis=1) / code.shape[1]
    return -(counts * np.log(counts + EPS)).sum(axis=1) / np.log(6)


def _band_shares(power: np.ndarray, freqs: np.ndarray) -> tuple:
    band = freqs >= 0.5
    total = power[:, band].sum(axis=1) + EPS
    share = lambda lo, hi: power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) / total
    cumulative = np.cumsum(power[:, band], axis=1) / total[:, None]
    f = freqs[band]
    rolloff = f[(cumulative >= 0.85).argmax(axis=1)]
    centroid = (power[:, band] * f).sum(axis=1) / total
    median = f[(cumulative >= 0.5).argmax(axis=1)]
    return share(20, 35), share(35, 50.0001), rolloff, centroid, median


def raw_numbers(x: np.ndarray) -> np.ndarray:
    """x: (windows, 232, time) -> (windows, len(RAW_NAMES))."""
    x = x.astype(np.float32, copy=False)
    n, _, T = x.shape
    xc = x - x.mean(axis=2, keepdims=True)
    cov = np.matmul(xc, xc.transpose(0, 2, 1)) / T
    values, vectors = np.linalg.eigh(cov.astype(np.float64))
    values = np.clip(values[:, ::-1], 0, None)
    vectors = vectors[:, :, ::-1]
    total = values.sum(axis=1) + EPS
    shares = values / total[:, None]
    f = {"pca_share_1": shares[:, 0], "pca_share_2": shares[:, 1], "pca_share_3": shares[:, 2], "pca_effective_rank": total ** 2 / ((values ** 2).sum(axis=1) + EPS),
         "pca_n90": (np.cumsum(shares, axis=1) < 0.9).sum(axis=1) + 1.0}
    links = xc.reshape(n, LINKS, PER_LINK, T).mean(axis=2)
    lc = np.stack([np.corrcoef(l)[np.triu_indices(LINKS, 1)] if l.std() > 0 else np.zeros(6) for l in links])
    f["link_coherence_mean"], f["link_coherence_min"] = np.nan_to_num(lc).mean(axis=1), np.nan_to_num(lc).min(axis=1)
    sd = np.sqrt(np.einsum("nii->ni", cov)) + EPS
    adjacent = [(i, i + 1) for link in range(LINKS) for i in range(link * PER_LINK, (link + 1) * PER_LINK - 1)]
    a, b = np.array(adjacent).T
    f["adjacent_subcarrier_corr"] = (cov[:, a, b] / (sd[:, a] * sd[:, b])).mean(axis=1)

    pc1 = np.einsum("nc,nct->nt", vectors[:, :, 0].astype(np.float32), xc)
    power, freqs = _spectrum(pc1)
    ac = np.fft.irfft(np.abs(np.fft.rfft(pc1 - pc1.mean(axis=1, keepdims=True), n=2 * T, axis=1)) ** 2, axis=1)[:, :T]
    ac = ac / (ac[:, :1] + EPS)
    window = ac[:, 19:251]
    middle = window[:, 1:-1]
    local_peak = (middle > window[:, :-2]) & (middle >= window[:, 2:])   # true local maxima only: the slowly falling start of the curve is not a period
    peaks = np.where(local_peak, middle, 0.0)
    f["ac_peak"], f["ac_lag"] = peaks.max(axis=1), np.where(peaks.max(axis=1) > 0, (peaks.argmax(axis=1) + 20) / SAMPLING_RATE_HZ, 0.0)
    mobility = _hjorth_mobility(pc1)
    f["hjorth_mobility"], f["hjorth_complexity"] = mobility, _hjorth_mobility(np.diff(pc1, axis=1)) / (mobility + EPS)
    f["zero_crossings_per_s"] = (np.diff(np.sign(pc1 - pc1.mean(axis=1, keepdims=True)), axis=1) != 0).sum(axis=1) / (T / SAMPLING_RATE_HZ)
    f["permutation_entropy"] = _permutation_entropy(pc1)
    s2035, s3550, rolloff, centroid, _ = _band_shares(power, freqs)
    f["rolloff85"], f["centroid"], f["share_20_35"], f["share_35_50"] = rolloff, centroid, s2035, s3550
    blocks = pc1.reshape(n, 10, T // 10).std(axis=2)
    f["duty_cycle"], f["burstiness"], f["block_cv"] = (blocks > 0.25 * blocks.max(axis=1, keepdims=True)).mean(axis=1), blocks.max(axis=1) / (blocks.mean(axis=1) + EPS), blocks.std(axis=1) / (blocks.mean(axis=1) + EPS)
    global_power = (np.abs(np.fft.rfft(xc, axis=2)) ** 2).sum(axis=1)
    g2035, g3550, _, _, gmedian = _band_shares(global_power, freqs)
    f["global_share_20_35"], f["global_share_35_50"], f["global_median_freq"] = g2035, g3550, gmedian
    frames = np.stack([pc1[:, a:a + 200] for a in range(0, T - 199, 100)], axis=1)   # four 2-second frames, 0.5 Hz resolution
    fp = np.abs(np.fft.rfft(frames - frames.mean(axis=2, keepdims=True), axis=2)) ** 2
    ff = np.fft.rfftfreq(200, d=1.0 / SAMPLING_RATE_HZ)
    fp = fp[:, :, (ff >= 0.5)]
    fa = ff[ff >= 0.5]
    fnorm = fp / (fp.sum(axis=2, keepdims=True) + EPS)
    f["stft_centroid_std"] = (fnorm * fa).sum(axis=2).std(axis=1)
    f["stft_peak_freq_std"] = fa[fp.argmax(axis=2)].std(axis=1)
    f["spectral_flux"] = 0.5 * np.abs(np.diff(fnorm, axis=1)).sum(axis=2).mean(axis=1)
    below = ac < np.exp(-1)
    f["decorrelation_time"] = np.where(below.any(axis=1), below.argmax(axis=1), T) / SAMPLING_RATE_HZ
    z = (pc1 - pc1.mean(axis=1, keepdims=True)) / (pc1.std(axis=1, keepdims=True) + EPS)
    f["pc1_skew"], f["pc1_kurtosis"] = (z ** 3).mean(axis=1), (z ** 4).mean(axis=1) - 3.0
    out = np.stack([f[name] for name in RAW_NAMES], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)



def _work(paths: list) -> np.ndarray:
    x = read_windows(paths)
    channel_spectral = spectral_shape(x).reshape(len(x), len(SPECTRAL), N_CHANNELS).mean(axis=2)
    return np.concatenate([raw_numbers(x), channel_spectral], axis=1)


def extract(limit: int | None = None, workers: int = 2) -> None:
    manifest = pd.read_csv(PREP_DIR / "manifest.csv")
    if limit:
        manifest = manifest.head(limit)
    root = data_root()
    paths = [str(root / p) for p in manifest["file_path"]]
    chunks = [paths[i:i + 128] for i in range(0, len(paths), 128)]
    target = PREP_DIR / ("physical_raw_test.npy" if limit else RAW_FILE.name)
    out = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(len(paths), len(RAW_NAMES) + len(SPECTRAL)))
    start, row = time.time(), 0
    with Pool(workers) as pool:
        for part in pool.imap(_work, chunks):
            out[row:row + len(part)] = part
            row += len(part)
    out.flush()
    print(f"{row} windows in {time.time() - start:.0f} s; non-finite values: {int((~np.isfinite(out)).sum())}")


def run_ids(session: np.ndarray, gap: int = RUN_GAP) -> np.ndarray:
    """A number per window; windows share it when they follow each other in the session numbers (no jump larger than `gap`)."""
    order = np.argsort(session, kind="stable")
    new_run = np.ones(len(session), dtype=bool)
    new_run[1:] = np.diff(session[order]) > gap
    ids = np.empty(len(session), dtype=np.int64)
    ids[order] = np.cumsum(new_run)
    return ids


def rolling(values: np.ndarray, run: np.ndarray, session: np.ndarray, last: int) -> pd.DataFrame:
    """Over the last `last` windows of the same run (the current window included, nothing from the future): mean, coefficient of variation, mean absolute change over the mean."""
    frame = pd.DataFrame({"v": values, "g": run, "s": session}).sort_values(["g", "s"])
    g = frame.groupby("g")["v"]
    mean = g.rolling(last, min_periods=1).mean().reset_index(level=0, drop=True)
    std = g.rolling(last, min_periods=1).std(ddof=0).reset_index(level=0, drop=True)
    count = g.rolling(last, min_periods=1).count().reset_index(level=0, drop=True)
    change = frame.assign(d=g.diff().abs().fillna(0.0)).groupby("g")["d"].rolling(last, min_periods=1).sum().reset_index(level=0, drop=True) / np.maximum(count - 1, 1)
    result = pd.DataFrame({"mean": mean, "cv": std / (mean + 1e-9), "change": change / (mean + 1e-9)}, index=frame.index)
    return result.loc[np.arange(len(values))]


def dynamic_features(table: pd.DataFrame, run: np.ndarray, session: np.ndarray) -> pd.DataFrame:
    """The 4 short dynamics (6 windows) and, when the raw numbers are present, 5 more (24 windows, and 6 windows of three raw numbers)."""
    r6, r24 = rolling(table["std"].to_numpy(), run, session, 6), rolling(table["std"].to_numpy(), run, session, 24)
    out = pd.DataFrame({"dyn_std_cv": r6["cv"], "dyn_std_change": r6["change"], "dyn_share_slow": rolling(table["spec_share_0.5-1"].to_numpy(), run, session, 6)["mean"],
                        "dyn_peak_to_mean": rolling(table["spec_peak_to_mean"].to_numpy(), run, session, 6)["mean"]}).reset_index(drop=True)
    if "pca_share_1" in table:
        out["dyn24_std_cv"], out["dyn24_std_change"] = r24["cv"].to_numpy(), r24["change"].to_numpy()
        for name in ("pca_share_1", "ac_peak", "permutation_entropy"):
            out["dyn6_" + name] = rolling(table[name].to_numpy(), run, session, 6)["mean"].to_numpy()
    return out


def channel_numbers(stats: np.ndarray) -> pd.DataFrame:
    """Numbers from the 2,320 statistics features (statistic-major, 232 channels each), averaged over the channels."""
    s = lambda i: np.asarray(stats[:, i * N_CHANNELS:(i + 1) * N_CHANNELS])
    links = np.stack([s(1)[:, k * PER_LINK:(k + 1) * PER_LINK].mean(axis=1) for k in range(LINKS)])
    freq = s(7)
    q75, q25 = np.percentile(freq, [75, 25], axis=1)
    bins = np.rint(freq * 5).astype(np.int32)   # 0.2 Hz grid
    best = np.zeros(len(freq))
    for k in range(bins.max() + 1):
        best = np.maximum(best, (bins == k).sum(axis=1))
    return pd.DataFrame({"std": s(1).mean(1), "mean_abs_diff": s(6).mean(1), "dominant_freq": np.median(freq, axis=1), "low_share": (s(8) / (s(8) + s(9) + 1e-9)).mean(1),
                         "range_over_std": ((s(3) - s(2)) / (s(1) + 1e-9)).mean(1), "skew": s(4).mean(1), "kurtosis": s(5).mean(1),
                         "std_variation_across_channels": s(1).std(1) / (s(1).mean(1) + 1e-9), "link_std_spread": links.std(0) / (links.mean(0) + 1e-9),
                         "dominant_freq_spread": q75 - q25, "dominant_freq_mode_share": best / N_CHANNELS})


def assemble() -> None:
    manifest = pd.read_csv(PREP_DIR / "manifest.csv")
    raw = np.load(RAW_FILE)
    assert len(raw) == len(manifest)
    spectral = pd.DataFrame(raw[:, len(RAW_NAMES):], columns=["spec_" + n for n in SPECTRAL])
    raw_numbers_table = pd.DataFrame(raw[:, :len(RAW_NAMES)], columns=RAW_NAMES)
    ch = channel_numbers(np.load(PREP_DIR / "features_stats.npy", mmap_mode="r"))
    first = ["std", "mean_abs_diff", "dominant_freq", "low_share", "range_over_std", "skew", "kurtosis"]
    table = pd.concat([ch[first], spectral, ch[["std_variation_across_channels", "link_std_spread", "dominant_freq_spread", "dominant_freq_mode_share"]], raw_numbers_table], axis=1)
    session = manifest["session"].to_numpy()
    full = pd.concat([table, dynamic_features(table, run_ids(session), session)], axis=1)
    np.save(FEATURES_FILE, full.to_numpy(np.float32))
    COLUMNS_FILE.write_text(json.dumps(list(full.columns)))
    print(f"{full.shape[0]} windows x {full.shape[1]} numbers -> {FEATURES_FILE.name}; non-finite values: {int((~np.isfinite(full.to_numpy())).sum())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assemble", action="store_true", help="assembly only")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not args.assemble:
        extract(args.limit, args.workers)
    if not args.limit:
        assemble()
