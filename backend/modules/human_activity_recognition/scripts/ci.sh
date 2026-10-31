#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
SCOPE="backend/modules/human_activity_recognition"
ruff format --check --config "$SCOPE/pyproject.toml" "$SCOPE"
ruff check --config "$SCOPE/pyproject.toml" "$SCOPE"
mypy --config-file "$SCOPE/pyproject.toml" "$SCOPE"
bandit -q -r "$SCOPE" -x "$SCOPE/tests"
python3 -m unittest discover -s "$SCOPE/tests" -t .
python3 -m unittest discover -s backend/tests -t .
npm ci --prefix frontend
npm --prefix frontend run build
