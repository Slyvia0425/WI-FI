"""Checks on the spectral-shape features and the link means of the spectral extension.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import numpy as np

from backend.modules.motion_source_recognition.experiments.spectral_evaluation import link_slice
from backend.modules.motion_source_recognition.experiments.spectral_features import link_means, spectral_features
from backend.modules.motion_source_recognition.preprocessing.physical import SPECTRAL

T = 500
TIME = np.arange(T) / 100.0


def one(signal):
    """Spectral numbers of a single signal as a dict."""
    return dict(zip(SPECTRAL, spectral_features(signal[None, None, :])[0]))


class SpectralTest(unittest.TestCase):
    def test_a_pure_tone_is_peaked_and_noise_is_flat(self):
        tone = one(np.sin(2 * np.pi * 3.0 * TIME))
        noise = one(np.random.default_rng(0).normal(size=T))
        self.assertLess(tone["entropy"], noise["entropy"])
        self.assertLess(tone["flatness"], noise["flatness"])
        self.assertGreater(tone["peak_to_mean"], noise["peak_to_mean"])
        self.assertAlmostEqual(tone["centroid"], 3.0, delta=0.3)
        self.assertGreater(tone["share_2-4"], 0.9)

    def test_band_shares_sum_to_one(self):
        r = one(np.random.default_rng(1).normal(size=T))
        self.assertAlmostEqual(sum(r[k] for k in SPECTRAL[:5]), 1.0, places=4)

    def test_signal_size_and_level_do_not_matter(self):
        x = np.random.default_rng(2).normal(size=T) + 5.0
        a, b = one(x), one(7.0 * x + 3.0)
        for key in SPECTRAL:
            self.assertAlmostEqual(a[key], b[key], delta=1e-3 * max(1.0, abs(a[key])), msg=key)

    def test_a_flat_signal_gives_finite_numbers(self):
        self.assertTrue(all(np.isfinite(list(one(np.full(T, 1.68)).values()))))

    def test_output_is_statistic_major(self):
        out = spectral_features(np.random.default_rng(3).normal(size=(2, 6, T)))
        self.assertEqual(out.shape, (2, 10 * 6))
        np.testing.assert_allclose(out[:, :6].sum(axis=1) + out[:, 6:12].sum(axis=1) + out[:, 12:18].sum(axis=1) + out[:, 18:24].sum(axis=1) + out[:, 24:30].sum(axis=1), 6.0, atol=1e-3)


class LinkTest(unittest.TestCase):
    def test_link_means_average_58_consecutive_channels(self):
        x = np.random.default_rng(4).normal(size=(3, 232, 20)).astype(np.float32)
        out = link_means(x)
        self.assertEqual(out.shape, (3, 4, 20))
        np.testing.assert_allclose(out[:, 2], x[:, 116:174].mean(axis=1), atol=1e-6)

    def test_link_slice_takes_one_link_of_every_block(self):
        x = np.arange(2320, dtype=np.float32)[None]
        out = link_slice(x, 3)
        self.assertEqual(out.shape, (1, 580))
        self.assertEqual(out[0, 0], 174.0)          # block 0, channel 174 = first channel of link 3
        self.assertEqual(out[0, 58], 232 + 174.0)   # block 1


if __name__ == "__main__":
    unittest.main()
