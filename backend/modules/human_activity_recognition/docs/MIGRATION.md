# Migration and rollback

## Current version

- Module version: `0.3.0`
- Public replay contract: `0.3`
- Backward compatibility: additive only relative to the repository `0.1` frame fields.

## 0.2 to 0.3 migration

- The supervised label set is now the five official csi-bench HAR classes:
  `jumping`, `running`, `seated-breathing`, `walking`, `wavinghand`.
- `transition` and `unknown` remain decision states and are not training labels.
- Replay checkpoints now record the normalisation mode. Older checkpoints without
  `feature_stats.mode` are interpreted as `global_zscore`; new real-data runs use
  `window_zscore`.
- The optional `spectral_ensemble` route is additive. It emits the same required
  fields and preserves the public replay contract.

## Additive field migration

Optional evidence fields (`activity_id`, `probabilities`, `stability`, `quality`,
`model_version`, `transition`) can be ignored by older frontends. Consumers that want
them must read them defensively.

## Required field change procedure

1. Update the contract, interface documentation, frontend metadata and fusion documentation.
2. Add contract and migration tests.
3. Publish a new module version.
4. Keep the previous replay and checkpoint available for rollback.

## Rollback

- Restore the previous replay/checkpoint version and the matching model configuration.
- The previous 0.2 checkpoint remains loadable because missing normalisation mode means
  `global_zscore`.
- Dataset and model artifacts are versioned by `model_version` and configuration hash.
