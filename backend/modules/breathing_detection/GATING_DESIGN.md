# Breathing assessability gates

## Purpose

The breathing model answers whether a 10-second CSI window contains breathing
evidence. It must not be forced to make that decision when the input is poor or
another source can create a similar pattern. `inference/gating.py` wraps the
frozen model probability in an ordered, explainable decision layer.

## Ordered decision

Only the first triggered gate becomes the primary reason:

1. invalid or low-quality CSI -> `not_assessable / low_quality`
2. clear motion -> `not_assessable / motion`
3. high external Fan probability -> `not_assessable / fan`
4. calibrated stationary-periodic risk -> `not_assessable / periodic_interference`
5. unstable consecutive-window evidence -> `not_assessable / temporal_instability`
6. otherwise apply the frozen seed 123 threshold -> `breathing` or `none`

Missing optional evidence means "unavailable", not zero risk. This distinction
prevents a disconnected activity or motion-source module from silently
asserting that a window is safe.

## Implemented now

- typed and range-checked gate inputs and configuration
- fixed precedence and machine-readable reason codes
- existing dataset quality flags feeding the quality gate
- external motion and Fan probability inputs
- explicit periodic-interference and temporal-instability inputs
- conversion to the existing public `{t, status, confidence}` frame contract
- unit tests for every gate, priority, missing evidence, threshold behaviour,
  invalid scores, and frame conversion

The public confidence is a decision-margin score. It is zero at the frozen
breathing threshold and approaches one farther from the boundary. A rejected
window reports the score of the gate that rejected it.

## Risk producer status

The unified estimator's motion-related signal-pollution study uses CSI-only
transient, energy, broadband and adjacent-correlation proxies. It has no reliable
per-window physical-motion labels. Its selected grouped rules withheld six
accurate rates and intercepted none of 19 large errors, so it is not a motion
probability producer and is not connected to a deployed gate. No BPM or breathing
status is changed. See [`MOTION_POLLUTION_REPORT.md`](MOTION_POLLUTION_REPORT.md).

The temporal-instability producer is implemented as a stateful five-window
tracker. It combines decision flips with proximity to the frozen breathing
threshold and waits for at least three windows before returning a score. The
official dataset does not provide a trustworthy ordered continuous replay, so
its operating threshold has not been empirically validated yet. A subsequent
5-second sliding-window replay of 467 independent WiRe-Breath segments produced
1,234 scored windows: the diagnostic 0.8 cutoff flagged one correct prediction
and no false `none` predictions. It remains disabled. These short segments are
not long-duration validation; see
[`STREAMING_PIPELINE_REPORT.md`](STREAMING_PIPELINE_REPORT.md).

A signal-only periodic-risk producer was trained on train and calibrated on
validation. It rejected 116 correct breathing windows while removing only six
false positives and therefore did not pass the promotion criterion. It is not
enabled by default. See
[`PERIODIC_GATE_REPORT.md`](PERIODIC_GATE_REPORT.md).

The K=208 analysis still cannot be converted into a hard gate: the official
test set has already been observed, and K is confounded with environment,
user, deployment, and label.

The next experiment must report both coverage and selective performance:

- rejection rate and reason distribution
- retained breathing recall and macro-F1
- retained empty-room false-positive rate
- fraction of existing validation errors rejected
- correct predictions unnecessarily rejected

After calibration, an independent newly collected or newly declared holdout
is required for an unbiased final assessment of the updated gated system.
