#!/usr/bin/env bash
set -euo pipefail

# ML Space's `binary` launcher still invokes the command once per visible GPU
# on this allocation. Keep exactly one coordinator and let the remaining ranks
# exit cleanly before touching shared state.
global_rank="${RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${GROUP_RANK:-${LOCAL_RANK:-0}}}}}"
case "$global_rank" in
  ''|*[!0-9]*) global_rank=0 ;;
esac
if [[ "$global_rank" -ne 0 ]]; then
  echo "[deepseek-job] global rank ${global_rank}: coordinator owned by rank 0; exiting"
  exit 0
fi

# The launcher may narrow CUDA_VISIBLE_DEVICES per MPI rank even though all
# eight devices are mounted into the worker. Rank 0 owns the TP=8 server.
export CUDA_VISIBLE_DEVICES="${DEEPSEEK_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

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
