#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must point to the synchronized repository clone}"
case "$PROJECT_ROOT" in
  /*) ;;
  *) echo "PROJECT_ROOT must be absolute: ${PROJECT_ROOT}" >&2; exit 2 ;;
esac
if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
  echo "PROJECT_ROOT does not contain pyproject.toml: ${PROJECT_ROOT}" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m scripts.cluster.run_job
