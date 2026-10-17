# Migration and rollback

## Current version

- Module contract: `0.2`
- Backward compatibility: additive only relative to the repository `0.1` frame fields.

## Additive field migration

Optional evidence fields (`probabilities`, `stability`, `quality`, `model_version`, `transition`) can be ignored by older frontends. Consumers that want them must read them defensively.

## Required field change procedure

1. Open a pull request in the module branch.
2. Update `contract.py`, `docs/INTERFACE.md`, frontend `meta.js`/`index.jsx` and fusion documentation.
3. Add a contract test and a migration test.
4. Publish a new module version.
5. Keep the previous replay available for rollback.

## Rollback

- Revert the module commit or restore the previous replay/checkpoint version.
- No shared route or contract file is rewritten in place.
- Dataset and model artifacts are versioned by `model_version` and config hash.
