# Peak evidence follow-up (development only)

## Stage 1: descriptive diagnosis, before a new rule

Audit all 394 quality-passing 30-second windows, retaining the original 73
exclusions. Inspect the union of the previous fold-selected 18 large errors and
any originally accurate window harmed by that selection. Include original
unified CSI and raw-before-resampling Hampel-8 as a diagnostic contrast, not a
deployed person-specific algorithm. Reference signals are never changed.

For every genuine mean-spectrum local peak in 0.1–0.7 Hz, record height,
prominence and the fraction of individual subcarrier dominant frequencies within
2 BPM. This tolerance is the existing accuracy definition, not a tuned value.
Use the wider 0.05–1 Hz context for local-peak detection. Save all candidates,
not only a reference-matching peak. Record reference local peaks, second/first
height ratio, and first/second 15-second reference rates (short-window estimates
have poor resolution and are descriptive only).

Evaluation-only labels: a CSI local peak within 2 BPM of the reference is
`reference_near_peak_present`; otherwise `no_reference_near_mean_peak`. Neither
label proves physiological identifiability. Flag reference competing peaks when
the second peak is at least half the tallest height, and temporal discrepancy
when half-window reference rates differ by >2 BPM. These are ambiguity warnings,
not permission to discard difficult references. Describe half/double relations
with 2 BPM tolerance in the reference-frequency units.

Create per-case spectra and carrier-frequency distributions for every target
case, including the reference waveform and spectra. No physical interference
label can be inferred from these proxies alone. Output is protected at
`outputs/peak_evidence_v1/`; no production/replay/model changes.

## Stages 2–3

After this audit, document one justified CSI-only candidate and freeze it before
computing its evaluation outcomes. Do not grid-search alternatives. Keep the
current estimator as the baseline/fallback. Reuse the existing whole-record,
leave-person-out protocol; select only with inner validation, report fixed-rule
and selected outer results, and acknowledge all data are already observed.

For a correction rule, support requires full coverage, zero harm to originally
accurate estimates, >=25% fewer >5 BPM errors, and no person's MAE worsening.
For a rejection rule, report retained coverage and count rejection of accurate
outputs as harm; it cannot claim improvement just by hiding errors. No automatic
adoption, no parameter retries after evaluation, even if the experiment fails.
