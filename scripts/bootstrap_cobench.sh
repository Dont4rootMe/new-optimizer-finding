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
#   COBENCH_REVISION pinned CO-Bench commit (default below)
#   COBENCH_DATA_REVISION pinned Hugging Face dataset revision (default below)
#   COBENCH_ALLOW_UNPINNED=1 accepts an existing checkout at another commit
#   AIFS_DATA_ROOT   dataset root (default: ./data); data lands in <root>/co-bench
#   PIP             pip executable (default: python -m pip)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COBENCH_ROOT="${COBENCH_ROOT:-${REPO_ROOT}/third_party/CO-Bench}"
DATA_ROOT="${AIFS_DATA_ROOT:-${REPO_ROOT}/data}"
COBENCH_DATA_DIR="${DATA_ROOT}/co-bench"
PIP="${PIP:-python -m pip}"
COBENCH_REVISION="${COBENCH_REVISION:-2f6620ba5cc8d80d6fefc121b5ee2c4dcec29f33}"
COBENCH_DATA_REVISION="${COBENCH_DATA_REVISION:-6e561a36254d4315b8c2d155e81a465b634e87e1}"

echo "[bootstrap_cobench] project root : ${REPO_ROOT}"
echo "[bootstrap_cobench] checkout root : ${COBENCH_ROOT}"
echo "[bootstrap_cobench] dataset dir   : ${COBENCH_DATA_DIR}"
echo "[bootstrap_cobench] code revision : ${COBENCH_REVISION}"
echo "[bootstrap_cobench] data revision : ${COBENCH_DATA_REVISION}"

# 1. Clone CO-Bench (skip if it already exists).
if [ -d "${COBENCH_ROOT}/.git" ] || [ -f "${COBENCH_ROOT}/evaluation/__init__.py" ]; then
  echo "[bootstrap_cobench] CO-Bench checkout already present, skipping clone."
  if [ -d "${COBENCH_ROOT}/.git" ]; then
    actual_revision="$(git -C "${COBENCH_ROOT}" rev-parse HEAD)"
    if [ "${actual_revision}" != "${COBENCH_REVISION}" ] && [ "${COBENCH_ALLOW_UNPINNED:-0}" != "1" ]; then
      echo "[bootstrap_cobench] ERROR: existing checkout is ${actual_revision}, expected ${COBENCH_REVISION}." >&2
      echo "Set COBENCH_ALLOW_UNPINNED=1 only for an intentional compatibility experiment." >&2
      exit 1
    fi
  fi
else
  echo "[bootstrap_cobench] cloning CO-Bench..."
  mkdir -p "$(dirname "${COBENCH_ROOT}")"
  git clone --no-checkout https://github.com/sunnweiwei/CO-Bench "${COBENCH_ROOT}"
  git -C "${COBENCH_ROOT}" checkout --detach "${COBENCH_REVISION}"
fi

# 2. Install the optional extra (dataset + solver deps for the targeted tasks).
echo "[bootstrap_cobench] installing the [co_bench] optional extra..."
( cd "${REPO_ROOT}" && ${PIP} install -e ".[co_bench]" )

# 3. Download the dataset via huggingface_hub (max_workers=1 for SSL robustness).
echo "[bootstrap_cobench] downloading the CO-Bench dataset to ${COBENCH_DATA_DIR}..."
mkdir -p "${COBENCH_DATA_DIR}"
COBENCH_DATA_DIR="${COBENCH_DATA_DIR}" COBENCH_DATA_REVISION="${COBENCH_DATA_REVISION}" python - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

target = os.environ["COBENCH_DATA_DIR"]
revision = os.environ["COBENCH_DATA_REVISION"]
path = snapshot_download(
    repo_id="CO-Bench/CO-Bench",
    repo_type="dataset",
    revision=revision,
    local_dir=target,
    max_workers=1,  # single worker: more robust to flaky SSL / large files
)
Path(target, ".source_revision.json").write_text(
    json.dumps({"repo_id": "CO-Bench/CO-Bench", "revision": revision}, indent=2) + "\n",
    encoding="utf-8",
)
print(f"[bootstrap_cobench] dataset downloaded to: {path}")
PY

echo "[bootstrap_cobench] done. Set COBENCH_ROOT=${COBENCH_ROOT} if it is not the default."
