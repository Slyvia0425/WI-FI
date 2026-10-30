# Common-mode versus relative-carrier representation pilot

Freeze before extraction/scoring. This is a retrospective signal-representation
pilot on already observed WiRe-Breath development data, not independent final
validation, a physical interference classifier or a production algorithm change.
Keep original unified estimator, raw files, references, quality rules, LSTM,
replay/frontend and previous experiments unchanged. No spike cleaning or PCA.

## One fixed decomposition

For each original quality-passing [1,K,900] 30Hz, 30s window:

1. Linearly detrend each carrier, remove carriers with std <=1e-8, and divide
   each remaining carrier by its own std (same initial standardization as the
   existing spectral estimator). Require at least two valid carriers.
2. Let Z be these standardized traces. The common trace C(t) is their arithmetic
   mean, with equal weights and no reference-based signs, alignment or selection.
3. Relative traces R(k,t)=Z(k,t)-C(t). Preserve C and R; do not presume one is
   interference or that removing C is safe. Z=C+R and carrier-mean R=0.
4. Record common mean-square / Z mean-square and residual mean-square / Z
   mean-square. They sum to one, but neither fraction is a breathing probability.

Apply the unchanged `mean_spectrum_boundary0_v1` to C and R separately, alongside
the original input baseline. The estimator restandardizes nonflat component
traces; relative low-energy noise can consequently be amplified. Report original
energy fractions alongside spectra so a normalized residual peak is not
mistaken for newly recovered physical power. Constant/unavailable components get
explicit None/reason, not a silently assigned baseline. No filtering, rejection,
new rate bounds, component mixing, threshold search or reference-based choice.

## Diagnostics and assessment

Use all 394 existing quality-passing windows, retain 73 exclusions. Audit
reference-near genuine mean-spectrum local peaks (within existing 2 BPM
tolerance), their height and rank for each representation. This is evaluation
only, not an oracle method. Compare the 19 original >5 BPM errors and all 365
original <=2 BPM accurate windows; separately flag the previous cleanup-harmed
Yearner segment003. Generate spectra/common/relative/reference waveform plots
for all 19 errors plus that accurate case. No condition/person-specific tweaks.

Report fixed common-only and relative-only coverage, MAE, record-macro MAE,
large errors, good harm (including unavailable outputs), original large corrected,
new large errors and per-person results. As a conservative diagnostic, reuse
whole-record leave-person-out folds to choose among baseline/common/relative
using inner-validation record-macro MAE, full coverage and zero accurate harm;
ties fewer changes then baseline. No newly observed outer result tunes a rule.
All source windows overlap within recordings; folds keep those recordings whole.

Exploratory support requires full coverage, zero good harm, >=25% large-error
reduction and no person's MAE worsening. Baseline fallback alone is no benefit.
Even a pass would require a separately justified fixed deployable method and
independent validation before integration. Do not retrospectively mix components
to match references or start a follow-up parameter sweep.

Protected output: `outputs/common_mode_v1/`. Refuse overwrites; save config,
protocol/code/source and protected application hashes before real extraction;
verify all raw MAT hashes against the existing 467-file source audit. No GitHub
submission. Use synthetic reconstruction/invariance/degeneracy tests in addition
to evaluation, never synthetic results as evidence of physical identifiability.
