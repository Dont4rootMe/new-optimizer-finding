#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the co_bench experiment family:
#   1. clone the (non-pip-installable) CO-Bench checkout into third_party/CO-Bench
#   2. install this project's optional [co_bench] extra (dataset + solver deps)
#   3. download the CO-Bench dataset into ${AIFS_DATA_ROOT:-./data}/co-bench
#
# Re-running is safe: the clone and the download both skip work that already
# exists. Environment overrides:
#   COBENCH_ROOT     where to clone CO-Bench (default: third_party/CO-Bench)
#   AIFS_DATA_ROOT   dataset root (default: ./data); data lands in <root>/co-bench
#   PIP             pip executable (default: python -m pip)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COBENCH_ROOT="${COBENCH_ROOT:-${REPO_ROOT}/third_party/CO-Bench}"
DATA_ROOT="${AIFS_DATA_ROOT:-${REPO_ROOT}/data}"
COBENCH_DATA_DIR="${DATA_ROOT}/co-bench"
PIP="${PIP:-python -m pip}"

echo "[bootstrap_cobench] project root : ${REPO_ROOT}"
echo "[bootstrap_cobench] checkout root : ${COBENCH_ROOT}"
echo "[bootstrap_cobench] dataset dir   : ${COBENCH_DATA_DIR}"

# 1. Clone CO-Bench (skip if it already exists).
if [ -d "${COBENCH_ROOT}/.git" ] || [ -f "${COBENCH_ROOT}/evaluation/__init__.py" ]; then
  echo "[bootstrap_cobench] CO-Bench checkout already present, skipping clone."
else
  echo "[bootstrap_cobench] cloning CO-Bench..."
  mkdir -p "$(dirname "${COBENCH_ROOT}")"
  git clone https://github.com/sunnweiwei/CO-Bench "${COBENCH_ROOT}"
fi

# 2. Install the optional extra (dataset + solver deps for the targeted tasks).
echo "[bootstrap_cobench] installing the [co_bench] optional extra..."
( cd "${REPO_ROOT}" && ${PIP} install -e ".[co_bench]" )

# 3. Download the dataset via huggingface_hub (max_workers=1 for SSL robustness).
echo "[bootstrap_cobench] downloading the CO-Bench dataset to ${COBENCH_DATA_DIR}..."
mkdir -p "${COBENCH_DATA_DIR}"
COBENCH_DATA_DIR="${COBENCH_DATA_DIR}" python - <<'PY'
import os

from huggingface_hub import snapshot_download

target = os.environ["COBENCH_DATA_DIR"]
path = snapshot_download(
    repo_id="CO-Bench/CO-Bench",
    repo_type="dataset",
    local_dir=target,
    max_workers=1,  # single worker: more robust to flaky SSL / large files
)
print(f"[bootstrap_cobench] dataset downloaded to: {path}")
PY

echo "[bootstrap_cobench] done. Set COBENCH_ROOT=${COBENCH_ROOT} if it is not the default."
