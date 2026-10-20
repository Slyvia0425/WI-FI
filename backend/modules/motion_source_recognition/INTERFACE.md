# Motion source recognition: interface

What this module hands to the other modules: for every 5-second window of CSI, who or what is moving (Human, Pet, IRobot or Fan), how sure it is, and whether it should be trusted. Produced by `inference.Detector`; checked by `tests/test_inference.py`.

## Output (one frame per window; `t` is added by the caller)

| Field | Type and values | Meaning |
|---|---|---|
| `label` | `Human`, `Pet`, `IRobot`, `Fan` or `unknown` | The class with the highest probability, or `unknown` when that probability is below the threshold. `unknown` is a decision state ("not enough evidence"), not a fifth class |
| `probs` | `{Human, Pet, IRobot, Fan}`, each 0 to 1, sum 1; `null` when `quality.level` is `invalid` | Calibrated probabilities. Always given, also when the label is `unknown`; the best guess is the largest value |
| `confidence` | 0 to 1 | The largest value of `probs` (0 when invalid). The label is `unknown` exactly when this is below the threshold |
| `quality` | `{ok: bool, level: "ok" \| "degraded" \| "invalid", reasons: [...]}` | See below |

Required by the shared contract: `label`, `confidence`, `probs` (`contract.py`). `quality` is optional for the readers and follows the shape already used in the mock data (`quality.ok`).

## The unknown rule

`probs` are the model's probabilities calibrated with one temperature T (the logits are divided by T; the class never changes). The label is `unknown` when `confidence` is below a threshold. The threshold is chosen so that **a fifth of the validation windows** are answered unknown; it is a property of the trained model, found on that model's validation windows.

| Model trained on the training rows of | T | Threshold |
|---|---|---|
| split 0 (used for the examples) | 2.304 | 0.806 |
| split 1 | 6.323 | 0.689 |
| split 2 | 0.688 | 0.536 |

On windows the model has not seen the answer is unknown more often: 22% to 29% on the test rows, mostly for IRobot and Fan (43% to 93% of their windows). One threshold for all classes is used on purpose: the calibrated confidence already makes the classes the model knows less well come out as unknown more often, and thresholds fitted per class from precision targets were unstable on the test rows (unknown between 0.4% and 46%).

## Quality

| Level | When | What the module does |
|---|---|---|
| `invalid` | The window is not 232 x 500, or has a value that is not a finite number | Does not run the model: `label` is `unknown`, `confidence` is 0, `probs` is `null` |
| `degraded` | A channel that is all zero (`zero_channels:n`) or constant (`constant_channels:n`), or a value outside 0 to 9 (`value_out_of_range`); none of this occurred in the 60,858 windows of the dataset | Answers as usual; the answer is less reliable. `reasons` says why |
| `ok` | None of the above | |

## How downstream modules should use it (the baseline is the rule of the fusion page)

| Reader | Rule |
|---|---|
| Activity recognition, fall detection | Use the window when `label == "Human"` and `quality.level != "invalid"`. The fusion page's gate (Human with `confidence` at least 0.6) is compatible: a `Human` label already means `confidence` is at least the threshold above |
| Breathing detection | Treat `probs.Fan` as soft evidence that periodic signals may come from a fan (for example, trust times `1 - probs.Fan`), not as a switch: the `Fan` label is not reliable (see the limits) |
| `unknown` | Activity recognition and fall detection stay off for this window; breathing detection keeps its default trust |
| `Pet`, `IRobot` | No action; they only mean "not a person" |

## Calling it

```python
from backend.modules.motion_source_recognition.inference import Detector
detector = Detector(split=0)        # model, MiniRocket transformer, T and threshold of that split
frame = detector(window)            # window: array (232, 500) or (232, 500, 1) of CSI amplitudes at 100 Hz
```

Optional smoothing: `detector(window, previous=[...])` averages the calibrated probabilities of up to 5 previous windows of the same run (oldest first) with the current one. It improves the result slightly (macro-F1 about +0.005 to +0.01) and delays a change of source by a few windows.

One window takes about 22 ms on one CPU thread; the model is 0.4 MB and the MiniRocket transformer 0.1 MB. The first call compiles for about 5 seconds, so call the detector once on a dummy window at start-up. The trained files are in `outputs/` (not committed): build them with the steps in the README.

## Examples (real windows from the validation rows of split 0, picked without looking at the answers; see `outputs/evaluation/example_frames.json`)

| True class | Frame |
|---|---|
| Human | `{"label": "Human", "confidence": 0.9028, "probs": {"Human": 0.9028, "Pet": 0.0468, "IRobot": 0.0285, "Fan": 0.0219}, "quality": {"ok": true, "level": "ok", "reasons": []}}` |
| Pet | `{"label": "Pet", "confidence": 0.8845, "probs": {"Human": 0.0455, "Pet": 0.8845, "IRobot": 0.047, "Fan": 0.0231}, "quality": {"ok": true, "level": "ok", "reasons": []}}` |
| IRobot (answered unknown) | `{"label": "unknown", "confidence": 0.7224, "probs": {"Human": 0.7224, "Pet": 0.2367, "IRobot": 0.0216, "Fan": 0.0193}, "quality": {"ok": true, "level": "ok", "reasons": []}}` |
| IRobot (answered Human: a wrong answer) | `{"label": "Human", "confidence": 0.857, "probs": {"Human": 0.857, "Pet": 0.0906, "IRobot": 0.0304, "Fan": 0.0219}, "quality": {"ok": true, "level": "ok", "reasons": []}}` |
| Fan (answered unknown) | `{"label": "unknown", "confidence": 0.5744, "probs": {"Human": 0.0296, "Pet": 0.5744, "IRobot": 0.3099, "Fan": 0.0861}, "quality": {"ok": true, "level": "ok", "reasons": []}}` |
| Human (answered unknown: the model was not sure enough) | `{"label": "unknown", "confidence": 0.7366, "probs": {"Human": 0.7366, "Pet": 0.1259, "IRobot": 0.0836, "Fan": 0.0539}, "quality": {"ok": true, "level": "ok", "reasons": []}}` |
| Made-up window with one dead channel | `{"label": "Pet", "confidence": 0.8688, "probs": {...}, "quality": {"ok": false, "level": "degraded", "reasons": ["zero_channels:1"]}}` |
| Made-up window with a missing value | `{"label": "unknown", "confidence": 0.0, "probs": null, "quality": {"ok": false, "level": "invalid", "reasons": ["non_finite_values"]}}` |

In the 400-window draw behind these examples the module answered Human 169 times, Pet 156 times, unknown 75 times, and never IRobot or Fan.

## Limits (read before relying on it)

| # | Limit | Evidence |
|---|---|---|
| 1 | **Not checked together with the other modules.** The data of this module and of the others have no room in common, so the interface is an agreement, not an end-to-end test | |
| 2 | **`unknown` means "the probability is not high enough"**, not that an unknown object was recognised | |
| 3 | **Known rooms, people and pets: good. New ones: close to chance.** Test rows (people and pets also in training): Human recall 0.93, Pet 0.89. One fixed room, other recordings: Human 0.98, Pet 0.99. With a whole room, with its people, pets and devices, left out of training: accuracy 0.25, the same as guessing among four classes | `outputs/evaluation/test_scores.md`, `single_room_E11.md`, `leave_one_room_out.md` |
| 4 | **IRobot and Fan are not recognised on devices the model has not seen**: test recall 0.00 to 0.01. Most of their windows become `unknown` (43% to 93%); the rest are answered as Human or Pet | `test_scores.md`, `confidence_threshold_test.md` |
| 5 | **Most mistakes are confident ones.** In splits 0 and 1 about two thirds of the errors had a confidence of 0.9 or more before calibration; with the unknown rule 34% of the errors are still answered (as Human or Pet, mostly for IRobot and Fan windows). Calibration lowers the calibration error from 0.119 to 0.084 on the test rows, it does not remove it | `error_analysis.md`, `confidence_threshold_test.md` |
| 6 | **Test scores** (three splits, mean): macro-F1 0.430 and accuracy 0.766 with every window answered; with the unknown rule 0.462 and 0.870 on the windows that are answered, about a quarter of the windows being unknown | `test_scores.md`, `confidence_threshold_test.md` |
| 7 | One recording source (one device pair at 5 GHz, amplitude only, no phase); only these four classes | |
| 8 | A model trained on other data needs its own T and threshold (found on its own validation windows) | |

What was tried and is not part of the interface (physical-feature route, per-class thresholds, other classifiers) is in the README and in `outputs/evaluation/`.
