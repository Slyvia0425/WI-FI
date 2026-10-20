"""One window in, one output frame out: the whole path of the main model (Logistic Regression on MiniRocket features) from the raw CSI window to the module's fields.

    detector = Detector(split=0)            # loads the model, the MiniRocket transformer, the temperature and the threshold of that split
    frame = detector(window)                # window: (232, 500) or (232, 500, 1) array of CSI amplitudes at 100 Hz (5 seconds)

The frame has the fields of INTERFACE.md: label (Human, Pet, IRobot, Fan or unknown), confidence, probs and quality. `t` is added by the caller.

Steps: quality check -> MiniRocket transform -> standardise and classify (the saved pipeline) -> calibrate with the temperature of the split (the logits are divided by T)
-> the highest calibrated probability is the confidence -> below the threshold of the split the label is unknown. The threshold is the one frozen in step 7 of the evaluation
(outputs/evaluation/confidence_threshold_frozen.json): the 20th percentile of the calibrated confidences of the validation windows of that split.

Optional smoothing: pass `previous`, the calibrated probabilities of the last windows of the same run (oldest first); the mean of up to SMOOTHING_WINDOWS - 1 of them and
the current one is used instead of the current one. Windows of different runs must not be mixed.
"""
import json
import pickle

import joblib
import numpy as np

from .calibration import temper, to_logits
from .preprocessing.config import CLASSES, DATA, EVALUATION, EVALUATION_DIR, PREP_DIR, SAMPLE_SHAPE
from .training.common import MODELS_DIR

FROZEN = EVALUATION_DIR / "confidence_threshold_frozen.json"
SHAPE = SAMPLE_SHAPE[:2]          # (232, 500)
VALUE_RANGE = tuple(DATA["value_range"])          # the training windows lie in 0 to 8.95
SMOOTHING_WINDOWS = EVALUATION["smoothing_windows"]


def check_quality(window) -> dict:
    """invalid: unusable (not 232 x 500, or a value that is not a finite number); degraded: usable but unlike the training windows (zero or constant channels, values out of range)."""
    x = np.asarray(window, dtype=np.float64) if window is not None else None
    if x is not None and x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x is None or x.shape != SHAPE:
        return {"ok": False, "level": "invalid", "reasons": ["wrong_shape"]}
    if not np.isfinite(x).all():
        return {"ok": False, "level": "invalid", "reasons": ["non_finite_values"]}
    reasons = []
    zero = int((np.abs(x).max(axis=1) == 0).sum())
    flat = int(((x.std(axis=1) == 0) & (np.abs(x).max(axis=1) != 0)).sum())
    if zero:
        reasons.append(f"zero_channels:{zero}")
    if flat:
        reasons.append(f"constant_channels:{flat}")
    if x.min() < VALUE_RANGE[0] or x.max() > VALUE_RANGE[1]:
        reasons.append("value_out_of_range")
    return {"ok": not reasons, "level": "degraded" if reasons else "ok", "reasons": reasons}


def decide(probs: np.ndarray, threshold: float) -> tuple:
    """(label, confidence): the class with the highest probability, or "unknown" when that probability is below the threshold."""
    confidence = float(probs.max())
    return (CLASSES[int(probs.argmax())] if confidence >= threshold else "unknown"), confidence


class Detector:
    def __init__(self, split: int = 0, threads: int = 1):
        import numba
        numba.set_num_threads(threads)
        frozen = json.loads(FROZEN.read_text())
        self.split, self.temperature, self.threshold = split, frozen["splits"][f"split_{split}"]["temperature"], frozen["splits"][f"split_{split}"]["threshold"]
        self.model = joblib.load(MODELS_DIR / frozen["model"] / f"split_{split}" / "model.joblib")
        self.transformer = pickle.loads((PREP_DIR / f"minirocket_split_{split}.pkl").read_bytes())
        self.transformer.n_jobs = self.transformer._n_jobs = threads

    def probabilities(self, window) -> np.ndarray:
        """Calibrated probabilities of the four classes for a window that passed the quality check."""
        x = np.asarray(window, dtype=np.float32)
        x = x[..., 0] if x.ndim == 3 else x
        features = self.transformer.transform(x[None]).astype(np.float32)
        return temper(to_logits(self.model.predict_proba(features)), self.temperature)[0]

    def __call__(self, window, previous: list | None = None) -> dict:
        quality = check_quality(window)
        if quality["level"] == "invalid":
            return {"label": "unknown", "confidence": 0.0, "probs": None, "quality": quality}
        probs = self.probabilities(window)
        if previous:
            probs = np.mean([np.asarray(p, dtype=np.float64) for p in previous[-(SMOOTHING_WINDOWS - 1):]] + [probs], axis=0)
        label, confidence = decide(probs, self.threshold)
        return {"label": label, "confidence": round(confidence, 4), "probs": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)}, "quality": quality}
