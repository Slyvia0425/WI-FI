# Genuine local peaks versus bounded spectral maximum

## Fixed experiment

The previous review found that 23 of 35 large errors selected the lower
respiratory search endpoint. This experiment keeps 30-second windows,
timestamp resampling, the reference rates and the 100 ms quality rule fixed.

The candidate searches a wider 0.05--1 Hz spectral context for genuine local
maxima, then accepts candidate rates only within the original 0.1--0.7 Hz
range. This lets a true near-boundary peak be recognized without raising the
minimum allowed rate. Synthetic tests cover a real 7.5 BPM signal.

Eight predefined candidates combine height or prominence ranking with minimum
relative prominences 0, 0.05, 0.1 or 0.2. Prominence is normalized by maximum
power in the allowed respiratory band. No valid peak means abstention.
The original bounded-maximum baseline is an eligible fallback.

Each fixed outer fold selects minimum record-macro MAE on inner validation,
requiring >=90% coverage of quality-passing windows and <=1% harm to originally
accurate estimates. Harm includes both rejection and changing an originally
<=2 BPM error to >2 BPM. Outer assessment labels never select the candidate.
The code saves these settings before feature extraction.

## Outer assessment results

| Measure | Original baseline | Inner-selected candidate |
|---|---:|---:|
| Retained quality-passing windows | 394 / 394 | 394 / 394 |
| Coverage of all 467 segments | 84.37% | 84.37% |
| Pooled MAE, BPM | 1.162 | 0.935 |
| Pooled RMSE, BPM | 3.663 | 3.466 |
| Equal-person mean MAE, BPM | 1.511 | 1.373 |
| Large errors >5 BPM | 35 | 24 |
| Within 2 BPM | 88.58% | 92.13% |
| Originally accurate windows rejected | 0 | 0 |
| Originally accurate windows made inaccurate | 0 | 1 / 349 (0.29%) |

Fourteen of the 35 original large errors become accurate to within 2 BPM;
21 remain large. Three previously non-large errors become large, giving 24
large errors overall. Of the 23 original lower-boundary large errors, ten
remain large after the selected correction.

| Outer subject | Inner-selected method | Baseline MAE | Candidate MAE | Large errors before -> after |
|---|---|---:|---:|---:|
| 01 | Highest prominence, min 0 | 0.851 | 0.271 | 6 -> 1 |
| 02 | Highest local peak, min 0 | 0.648 | 0.418 | 7 -> 4 |
| 03 | Highest local peak, min 0 | 1.225 | 0.727 | 6 -> 4 |
| 04 | Original baseline | 2.566 | 2.566 | 4 -> 4 |
| 05 | Highest prominence, min 0 | 2.267 | 2.883 | 12 -> 11 |

The selected candidate improves pooled Quiet MAE from 0.348 to 0.213 BPM and
Interference MAE from 2.055 to 1.725 BPM, but person-level effects differ.
Worst-person MAE worsens from 2.566 to 2.883 BPM. Positive minimum-prominence
thresholds were not selected in any fold; these results do not justify adding
a strong prominence rejection threshold.

## Failure and control interpretation

The highest-prominence rule introduces three large errors in Subject05.
One changes a previously accurate Soldier Interference segment004 from
0.44 BPM error to 23.73 BPM error. An isolated high-prominence peak can still
be the wrong physiological component. Good aggregate harm does not remove
this severe individual failure.

A predefined fixed highest-local-peak, zero-minimum-prominence control reaches
pooled MAE 0.683 BPM, 19 large errors, and no originally accurate estimate
spoiled or rejected. This is a useful diagnostic contrast, not a reason to
retroactively replace inner selection with the best outer result. It suggests
that prominence ranking itself can be unstable across people. The data remain
historically observed development data; a later final claim requires new data.

## Decision and next step

Retain the original application behavior. This controlled experiment supports
the usefulness of genuine local peaks, with no coverage loss in this dataset,
but does not establish a universally safe replacement. The subject05 failure
prevents interpreting the aggregate improvement as solved reliability.

The next conservative candidate should restrict correction to original
boundary maxima and prefer peak height over prominence alone, leaving already
valid interior peaks unchanged. Test the rule using the same fixed inner
selection/outer reporting discipline, including low-rate synthetic signals,
per-person worst error and original-good harm. Do not tune it directly to the
three observed Subject05 failures or simply raise the minimum breathing rate.

## Reproduction

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.evaluation.evaluate_local_rate_peaks
```

The runner preserves an existing configured run. Outputs are ignored under
`outputs/respiratory_rate_local_peaks_v1/`: config, every candidate's per-window
prediction/local peaks, validation selection and outer metrics. No LSTM
weights, reference labels, official test data or production API are changed.
