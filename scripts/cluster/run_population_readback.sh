#!/usr/bin/env bash
set -euo pipefail

# Read-only result extraction for a run on the regional job NFS. The exact
# source run id is supplied as readback.source_run_id=<id> in
# HYDRA_OVERRIDES_JSON by scripts.cluster.submit.
global_rank="${RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${GROUP_RANK:-${LOCAL_RANK:-0}}}}}"
case "$global_rank" in
  ''|*[!0-9]*) global_rank=0 ;;
esac
if [[ "$global_rank" -ne 0 ]]; then
  echo "[population-readback] global rank ${global_rank}: coordinator owned by rank 0; exiting"
  exit 0
fi

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must point to the synchronized repository clone}"
JOB_ROOT="${JOB_ROOT:?JOB_ROOT must be an explicit absolute path}"
case "$PROJECT_ROOT" in
  /*) ;;
  *) echo "PROJECT_ROOT must be absolute: ${PROJECT_ROOT}" >&2; exit 2 ;;
esac
case "$JOB_ROOT" in
  /*) ;;
  *) echo "JOB_ROOT must be absolute: ${JOB_ROOT}" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m scripts.cluster.population_readback
