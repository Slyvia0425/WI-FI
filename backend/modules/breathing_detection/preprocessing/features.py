"""Interpretable periodicity features for the breathing baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PeriodicityFeatureConfig:
    sampling_rate_hz: float = 30.0
    breathing_band_hz: tuple[float, float] = (0.1, 0.7)
    top_k_subcarriers: int = 64
    consistency_tolerance_hz: float = 0.1
    epsilon: float = 1e-8

    def to_dict(self) -> dict:
        return asdict(self)


FEATURE_NAMES = (
    "signal_std_median",
    "signal_std_iqr",
    "band_power_ratio_median",
    "band_power_ratio_q75",
    "band_power_ratio_max",
    "peak_frequency_median_hz",
    "peak_frequency_iqr_hz",
    "peak_power_share_median",
    "peak_power_share_q75",
    "spectral_entropy_median",
    "motion_power_ratio_median",
    "autocorr_peak_median",
    "autocorr_peak_q75",
    "autocorr_frequency_median_hz",
    "autocorr_frequency_iqr_hz",
    "peak_frequency_mad_hz",
    "peak_frequency_consistency",
    "median_waveform_correlation",
)

FEATURE_VERSION = "1"


def _iqr(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


def extract_periodicity_features(
    x: np.ndarray,
    config: PeriodicityFeatureConfig | None = None,
) -> np.ndarray:
    """Return fixed-length features from a project-layout ``[C,K,T]`` window.

    Channels are flattened into the subcarrier axis. The highest-variance
    subcarriers are analysed individually, then robustly aggregated. No label
    or dataset-wide statistic is used here.
    """
    config = config or PeriodicityFeatureConfig()
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected [C,K,T], got {values.shape}")
    channels, subcarriers, time_steps = values.shape
    if channels < 1 or subcarriers < 1 or time_steps < 3:
        raise ValueError(f"Invalid CSI shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Periodicity features require finite input")

    signals = values.reshape(channels * subcarriers, time_steps)
    signals = signals - signals.mean(axis=1, keepdims=True)
    signal_std = signals.std(axis=1)
    top_k = min(config.top_k_subcarriers, signals.shape[0])
    selected = np.argsort(signal_std)[-top_k:]
    signals = signals[selected]
    selected_std = signal_std[selected]
    standardized = signals / np.maximum(selected_std[:, None], config.epsilon)

    frequencies = np.fft.rfftfreq(time_steps, d=1.0 / config.sampling_rate_hz)
    spectrum = np.fft.rfft(standardized, axis=1)
    power = np.abs(spectrum) ** 2
    positive = frequencies > 0
    low, high = config.breathing_band_hz
    band = (frequencies >= low) & (frequencies <= high)
    if not band.any():
        raise ValueError("No FFT bins fall inside the configured breathing band")

    positive_power = power[:, positive].sum(axis=1) + config.epsilon
    band_power = power[:, band]
    band_sum = band_power.sum(axis=1)
    band_ratio = band_sum / positive_power
    peak_index = np.argmax(band_power, axis=1)
    band_frequencies = frequencies[band]
    peak_frequency = band_frequencies[peak_index]
    peak_power = band_power[np.arange(top_k), peak_index]
    peak_share = peak_power / (band_sum + config.epsilon)

    normalized_positive_power = power[:, positive] / positive_power[:, None]
    spectral_entropy = -np.sum(
        normalized_positive_power
        * np.log(np.maximum(normalized_positive_power, config.epsilon)),
        axis=1,
    ) / np.log(normalized_positive_power.shape[1])
    motion_band = frequencies > high
    motion_ratio = power[:, motion_band].sum(axis=1) / positive_power

    # Biased FFT autocorrelation is stable at long lags. Requiring at least two
    # cycles caps the largest lag at T/2, so a 10 s window supports >=0.2 Hz.
    fft_size = 1 << (2 * time_steps - 1).bit_length()
    fft_signal = np.fft.rfft(standardized, n=fft_size, axis=1)
    autocorr = np.fft.irfft(fft_signal * np.conj(fft_signal), n=fft_size, axis=1)
    autocorr = autocorr[:, :time_steps] / time_steps
    autocorr /= np.maximum(autocorr[:, :1], config.epsilon)
    min_lag = max(1, int(np.ceil(config.sampling_rate_hz / high)))
    max_lag = min(
        time_steps // 2,
        int(np.floor(config.sampling_rate_hz / max(low, config.epsilon))),
    )
    lag_values = np.arange(min_lag, max_lag + 1)
    lag_region = autocorr[:, min_lag : max_lag + 1]
    autocorr_index = np.argmax(lag_region, axis=1)
    autocorr_peak = lag_region[np.arange(top_k), autocorr_index]
    autocorr_frequency = config.sampling_rate_hz / lag_values[autocorr_index]

    median_peak_frequency = float(np.median(peak_frequency))
    peak_frequency_mad = float(
        np.median(np.abs(peak_frequency - median_peak_frequency))
    )
    peak_frequency_consistency = float(
        np.mean(
            np.abs(peak_frequency - median_peak_frequency)
            <= config.consistency_tolerance_hz
        )
    )
    median_waveform = np.median(standardized, axis=0)
    median_waveform_std = float(median_waveform.std())
    if median_waveform_std < config.epsilon:
        waveform_correlation = 0.0
    else:
        reference = (median_waveform - median_waveform.mean()) / median_waveform_std
        waveform_correlation = float(
            np.median(np.abs((standardized @ reference) / time_steps))
        )

    features = np.asarray(
        [
            np.median(selected_std),
            _iqr(selected_std),
            np.median(band_ratio),
            np.quantile(band_ratio, 0.75),
            np.max(band_ratio),
            median_peak_frequency,
            _iqr(peak_frequency),
            np.median(peak_share),
            np.quantile(peak_share, 0.75),
            np.median(spectral_entropy),
            np.median(motion_ratio),
            np.median(autocorr_peak),
            np.quantile(autocorr_peak, 0.75),
            np.median(autocorr_frequency),
            _iqr(autocorr_frequency),
            peak_frequency_mad,
            peak_frequency_consistency,
            waveform_correlation,
        ],
        dtype=np.float32,
    )
    if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
        raise ValueError("Feature extraction produced invalid output")
    return features
