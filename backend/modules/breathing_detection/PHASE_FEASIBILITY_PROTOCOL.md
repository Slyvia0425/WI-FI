# Complex-CSI phase feasibility audit

Freeze before batch phase extraction or rate scoring. This is an exploratory
audit on previously observed WiRe-Breath development data, not calibration
certification, independent validation or a production algorithm change. Preserve
all raw/reference data, original amplitude estimator, LSTM, quality gates,
replay/frontend and previous experiment outputs; no training or GitHub changes.

## Source/structure audit (all 467 files)

Inventory complex dtype/shape, finite real/imaginary values, nonzero imaginary
fraction, zero coefficients, wholly zero carriers and intermittently zero
carriers. Initial source inspection already shows genuine complex64 coefficients
and some exact zeros; angle(0) must not be treated as a measured phase.
Inventory metadata for antenna identity, physical subcarrier indices/frequencies,
hardware gain or phase calibration information. Do not infer these from 234
columns or assume columns are multiple antennas. Ordered columns alone are not
a verified uniformly spaced physical-frequency axis.

Use only raw samples bracketing each existing 30s target grid. Reject a carrier
for phase if any coefficient in this interval is nonfinite or abs(H)<=1e-12;
report whole-zero versus intermittent-invalid counts, with no label-based
carrier choice or zero-phase interpolation. Adjacent differences use only
genuinely neighboring original column pairs whose two carriers are valid.
Do not bridge removed columns. Raw amplitude/reference/gap quality remain as is.

Record circular temporal increments, fraction with magnitude >pi/2, median
across-carrier temporal-increment circular coherence, and adjacent-column
circular coherence. These are descriptive, not evidence of hardware faults,
movement or reliable unwrapping. Unwrap cannot uniquely recover changes >pi per
sample without assumptions; a small wrapped jump does not certify calibration.

## Three fixed diagnostic representations (no sweep)

1. `phase_raw`: angle of valid carriers, temporally unwrap on native samples.
2. `phase_adjacent`: angle(H[k+1]*conj(H[k])) for valid original adjacent pairs,
   temporally unwrap. Common time-varying phase rotation cancels algebraically,
   but frequency-dependent errors or useful common respiration can remain/be lost.
3. `phase_index_residual`: unwrap phase across valid ordered carrier positions
   per native timestamp, least-squares subtract intercept and linear column-index
   slope, wrap residual to (-pi,pi], then temporally unwrap. Use original column
   positions and require >=3 valid carriers. This is an **index-affine nuisance
   projection**, not verified physical-frequency sanitization/calibration. Its
   unwrap/linear assumptions can fail or remove true signal. Report this caveat.

Linearly resample these unwrapped real-valued traces to the existing 900 samples
at 30Hz, without extrapolation, extra future file tail or cross-file history.
Use unchanged unified spectral estimator for a readout, alongside the unchanged
amplitude baseline. Flat/missing representations get explicit unavailable/reason,
never silent baseline substitution or upgraded timestamp quality. No phase
scaling, high-pass redesign, PCA, learned weighting, reference-aligned signs,
antenna ratios, amplitude/phase fusion, extra lag, tuning grid or new rate bounds.

## Assessment and plots

Audit all 467 but score exactly the existing 394 quality-passing windows. Report
availability/coverage, mean and record-macro error, >5 BPM errors, harm to 365
original accurate results (including unavailable output), original large errors
corrected and per-person results. Reference-near local-peak height/rank/presence
is descriptive only, not oracle selection. Previously observed data and confirmed
within-record overlap preclude independent-test claims.

As a conservative diagnostic only, original plus the three fixed representations
may be selected by existing whole-record leave-person-out inner validations:
lowest record-macro MAE, full coverage and zero original-good harm, ties fewer
changed predictions then original. Report fixed and grouped outcomes separately;
no outer-person tuning or reference-driven representation mixture. Support
requires full coverage, zero harm, >=25% fewer large errors and no person's MAE
worsening. Baseline fallback alone is not benefit or evidence phase is calibrated.

Generate plots for the first three original large errors by descending error
then sample ID and the already known Subject04 Freefaller Interference seg002
contradictory case, with native phase increments, resampled phase traces,
spectra/reference and carrier validity. Illustrative traces use the first valid
carrier/pair only, not a reference-selected carrier. No outcome-based extra
representation is added after scoring.

Protected directory `outputs/phase_feasibility_v1/`; refuse overwrite. Save
protocol/code/source/protected artifact hashes before extraction, verify all
467 original source hashes before/after. Add synthetic tests for zero handling,
no bridging, common-rotation cancellation, affine projection, immutability,
resampling coverage and the limitations of common-signal removal. No automatic
deployment even if a diagnostic score improves.
