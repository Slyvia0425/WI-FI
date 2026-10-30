# Paired waveform review of the 35 large rate errors

## Scope and review procedure

All 35 quality-passing 30-second windows with absolute rate error above 5 BPM
were reviewed computationally against the aligned respiratory sensor waveform.
Each has a paired diagnostic figure containing reference waveform, filtered
reference peaks, illustrative CSI streams, mean CSI/reference spectra, and
three 10-second CSI/reference rates. Representative figures were additionally
visually inspected (cases 01, 09, 13 and 26).

Accurate controls have error <=2 BPM. Matching prefers the same original
record, then participant/posture/environment, then participant/posture, then
participant. Twenty-seven cases obtain a same-record control, seven match
participant/posture across environment, and one matches participant only.
There are 18 distinct controls, reused across some comparisons. Case numbers
sort by original absolute error and are diagnostic labels, not split IDs.

The reference cross-check uses a fixed 0.08--0.8 Hz Butterworth filter,
prominence 0.5 after standardization, and a minimum peak separation consistent
with 0.7 Hz. Median peak interval gives an alternative rate. This is a
cross-check of the same sensor, not independent ground truth. Illustrative
CSI carriers are selected with reference information and must never be
treated as a deployable subcarrier-selection method.

## Findings

### Reference harmonic ambiguity is not the dominant observed pattern

In 34/35 errors, the median reference peak-interval rate is within 2 BPM of
the original spectral reference and the interval coefficient of variation is
<=0.25. None triggers the factor-two disagreement criterion; none has the
defined strong secondary half/double reference spectral energy. These
heuristics do not rule out all reference artifacts, but they do not support
systematically halving or doubling the reference labels.

The exception is case 09, Subject04 Foetus Quiet segment002. Its spectral
reference is 19.34 BPM, peak-interval estimate 22.64 BPM, and 10-second
reference peaks vary by more than 6 BPM. The waveform shows changing cycle
spacing and amplitude. Its exact rate definition deserves review; no label
has been changed. Limited short-window resolution and filtering also
contribute uncertainty.

### Most numerical half-rate cases are not evidence of a genuine subharmonic

Of 35 errors, 23 select exactly the first allowed FFT grid position,
6.15234375 BPM. One selects exactly the upper endpoint, 41.748046875 BPM.
Nine of the 15 previously identified near-half cases are among those exact
lower-endpoint outputs.

For example, case 26 (Subject02 Yearner Interference segment002) estimates
6.15 BPM against a 12.30 BPM spectral reference; reference peak intervals give
11.84 BPM. The CSI mean spectrum rises toward the lower search endpoint,
while a smaller component remains around the reference rate. This supports
low-frequency competition/band-edge selection rather than a reference rate
that should simply be halved.

The baseline selects the largest value inside a bounded frequency range; it
does not require an interior local maximum. A rising shoulder from drift or
broadband contamination can therefore be returned as the apparent rate.
This is an estimator limitation demonstrated by the plots, not proof of the
physical source of the low-frequency energy.

### Correct-frequency energy often survives in the CSI

In 34/35 errors, CSI spectral power within ±1/30 Hz of the reference reaches
at least half the maximum CSI power within the respiratory search band.
This indicates retained energy near the reference; it does not prove a
distinct sharp respiratory peak in every case. Several spectra are broad,
flat or contain multiple competing local peaks.

Case 13 (Subject05 Log Interference segment003) illustrates a near-double
error: CSI 24.61 BPM, spectral reference 12.30 BPM, reference peak intervals
12.72 BPM. CSI has several competing peaks, including near the reference
and near twice it. There is no corresponding strong doubled reference
pattern. The current evidence favors a CSI peak-selection problem, while
motion versus respiratory harmonic origin is unresolved.

Case 01 (Subject03 Yearner Interference segment002) selects the upper search
endpoint at 41.75 BPM despite a stable reference near 12.74 BPM. CSI spectra
are broad and three short-window estimates disagree. This is not a simple
half/double error.

Thirty of the 35 errors occur in the recorded Interference condition. Taken
with same-record successes and competing spectra, interference is a plausible
contributor, but environment labels and these signals cannot isolate the
physical motion source or prove it causes every error. Quiet examples also
exhibit boundary/competition failures.

## Decisions and next controlled experiment

- Preserve all existing reference rates and model outputs; the analysis does
  not justify automatic rate doubling or reference relabeling.
- The next estimator experiment should distinguish a genuine interior peak
  from a search-boundary maximum and test a minimum local prominence.
- Compare candidate peak ranking using concentration, competing-peak evidence
  and subcarrier agreement. Select settings with the fixed inner-validation
  partitions and report coverage/good-window rejection on outer participants.
- A boundary condition should initially be an uncertainty feature, not an
  automatic rejection of every low respiratory rate. Do not raise the lower
  search bound just to hide the observed errors.
- Revisit the ambiguous reference case and count/filter sensitivity before
  treating its spectral rate as a precise target.

No production confidence rule or estimator change is enabled in this review.
The existing 100 ms gap rule and frozen breathing classifier remain unchanged.

## Artifacts

Run:

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.evaluation.review_rate_waveforms
```

Matplotlib is required for offline figures. Local ignored artifacts are in
`outputs/respiratory_rate_baseline_v1/waveform_review/`: `index.html` presents
all 35 paired figures; `review.json` contains per-case cross-checks, control
IDs and overlapping diagnostic evidence tags. Figures include selected
carriers for illustration, not proof of waveform recovery.
