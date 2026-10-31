"""Inspectable spectral respiratory-rate estimators; units are breaths/minute."""
from __future__ import annotations

import numpy as np
from scipy.signal import detrend

METHODS = ('mean_spectrum', 'median_spectrum', 'median_peak')


def spectra(signals, rate_hz=30., band_hz=(.1, .7)):
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 30 or not np.isfinite(values).all():
        raise ValueError('Expected finite [signals,time] with at least 30 samples')
    if rate_hz <= 1.4:
        raise ValueError('Sampling rate must exceed twice the upper respiratory band')
    centered = detrend(values, axis=1, type='linear')
    valid = centered.std(axis=1) > 1e-8
    if not valid.any():
        raise ValueError('No nonconstant signal')
    centered = centered[valid]
    centered /= centered.std(axis=1, keepdims=True)
    nfft = 2 ** int(np.ceil(np.log2(4 * values.shape[1])))
    frequencies = np.fft.rfftfreq(nfft, 1 / rate_hz)
    power = np.abs(np.fft.rfft(centered * np.hanning(values.shape[1]), n=nfft, axis=1)) ** 2
    band = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    total = (frequencies >= .1) & (frequencies <= min(5., rate_hz / 2))
    power /= np.maximum(power[:, total].sum(axis=1, keepdims=True), 1e-12)
    return frequencies[band], power[:, band]


def estimate_rates(x, rate_hz=30.):
    values = np.asarray(x)
    if values.ndim != 3 or values.shape[0] != 1:
        raise ValueError('Expected [1,subcarriers,time]')
    frequencies, power = spectra(values[0], rate_hz)
    return {
        'mean_spectrum': float(frequencies[np.argmax(power.mean(axis=0))] * 60),
        'median_spectrum': float(frequencies[np.argmax(np.median(power, axis=0))] * 60),
        'median_peak': float(np.median(frequencies[np.argmax(power, axis=1)]) * 60),
    }


def reference_rate(signal, rate_hz=30.):
    frequencies, power = spectra(np.asarray(signal)[None, :], rate_hz)
    return float(frequencies[np.argmax(power[0])] * 60)
