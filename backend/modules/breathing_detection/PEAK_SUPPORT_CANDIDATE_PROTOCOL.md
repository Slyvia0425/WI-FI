# One frozen supported-competitor candidate

Frozen after descriptive `peak_evidence_v1/audit_summary.json` and before any
new candidate predictions or evaluation outcomes. All data have been observed;
this is explicitly a retrospective development experiment, not independent
validation. No production path is changed.

## Evidence and hypothesis

In the 18 targets, original CSI has a reference-near local peak in 16 cases;
raw-Hampel-8 has one in all 18. Most reference-near candidates have weaker carrier
support than a competing incorrect peak. No reference second-peak ratio reaches
0.5; four references have >2 BPM half-window discrepancy (descriptive, with low
short-window resolution). The harmed accurate case retains a strong correct
peak after cleaning but its rank changes. These findings do not support enabling
cleaning, blindly taking a second peak, doubling rates, or declaring no signal.

Test the narrower hypothesis that relative subcarrier support can safely resolve
some competing peaks. Carrier dominance is correlated evidence, not independent
votes or physiological probability. This simple rule is expected to miss many
known failures; this is a falsifiable test, not a claim of sufficient reliability.

## Single rule: supported_competitor_v1

- Use **unmodified** original 30s amplitude input and frozen unified rate.
- Reuse CSI-only mean-spectrum genuine local candidates in 0.1–0.7 Hz, detected
  with 0.05–1 Hz context and existing normalization.
- Eligible competitors have height >=0.5 of the maximum allowed mean spectrum:
  within a factor of two, excluding clearly weak spectral ripples. This one
  threshold is fixed here, no sweep.
- Support is the fraction of individual carrier dominant frequencies in
  0.1–0.7 Hz within 2 BPM (existing accuracy tolerance) of a proposed rate.
- Choose eligible peak by greatest support, then height, then lower frequency.
- Change only if its support is **strictly greater** than the current rate's
  support and its distance from that rate is >2 BPM. Otherwise preserve the
  original rate, including ties/no peaks. No rejection, preprocessing, reference,
  identity, posture, environment, or short-window reference enters this rule.

## Selection and assessment

Exactly two choices: unchanged baseline and this rule. Reuse frozen whole-record
leave-person-out folds. Inner validation selects lower record-macro MAE among
full-coverage, zero-current-good-harm choices; ties fewer changed rates, baseline.
No assessment person or outcomes tune the rule. Report fixed-rule outcomes as
well as nested outer selection, per-person results, new large errors, accurate
harm, coverage, and individual changed cases. Good baseline denominator is 365.

Support requires full coverage, zero accurate harm, >=25% reduction of the 19
current >5 BPM errors, and no person's MAE worsening. Baseline fallback alone
cannot pass the benefit criterion. Protect `outputs/peak_support_v1/`, write
config/hashes before scoring, no overwrites or second configuration after results.
No automatic deployment even if successful. No UI, replay, LSTM or data changes.
