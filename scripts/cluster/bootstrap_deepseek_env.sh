#!/usr/bin/env bash
set -euo pipefail

# Build an immutable shared-NFS runtime for the DeepSeek server. A directory
# lock prevents concurrent jobs from corrupting the same environment; failed
# builds are preserved for diagnosis rather than deleted.

SGLANG_VERSION="${SGLANG_VERSION:-0.5.16}"
DEEPSEEK_ENV_DIR="${DEEPSEEK_ENV_DIR:?DEEPSEEK_ENV_DIR must be an explicit absolute path}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"

case "$DEEPSEEK_ENV_DIR" in
  /*) ;;
  *) echo "DEEPSEEK_ENV_DIR must be absolute: ${DEEPSEEK_ENV_DIR}" >&2; exit 2 ;;
esac
if [[ "$DEEPSEEK_ENV_DIR" == "/" || "$DEEPSEEK_ENV_DIR" == "/home" || "$DEEPSEEK_ENV_DIR" == "/home/jovyan" ]]; then
  echo "Refusing unsafe DEEPSEEK_ENV_DIR: ${DEEPSEEK_ENV_DIR}" >&2
  exit 2
fi

ready_marker="${DEEPSEEK_ENV_DIR}/.runtime-ready.json"
if [[ -x "${DEEPSEEK_ENV_DIR}/bin/python" && -f "$ready_marker" ]]; then
  if "${DEEPSEEK_ENV_DIR}/bin/python" - "$SGLANG_VERSION" <<'PY'
import importlib.metadata
import sys

expected = sys.argv[1]
raise SystemExit(0 if importlib.metadata.version("sglang") == expected else 1)
PY
  then
    echo "[cluster-env] reusing ${DEEPSEEK_ENV_DIR} (sglang=${SGLANG_VERSION})"
    exit 0
  fi
fi

parent_dir="$(dirname "$DEEPSEEK_ENV_DIR")"
mkdir -p "$parent_dir"
lock_dir="${DEEPSEEK_ENV_DIR}.build.lock"

if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "[cluster-env] another process is building ${DEEPSEEK_ENV_DIR}; waiting"
  for _attempt in $(seq 1 480); do
    if [[ -x "${DEEPSEEK_ENV_DIR}/bin/python" && -f "$ready_marker" ]]; then
      echo "[cluster-env] environment became ready"
      exit 0
    fi
    sleep 15
  done
  echo "[cluster-env] timed out waiting for ${lock_dir}; inspect or move a stale lock" >&2
  exit 1
fi

build_dir="${DEEPSEEK_ENV_DIR}.building.$(date -u +%Y%m%dT%H%M%SZ).$$"
cleanup_lock() {
  rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup_lock EXIT INT TERM

echo "[cluster-env] bootstrap python: ${BOOTSTRAP_PYTHON}"
"$BOOTSTRAP_PYTHON" --version
"$BOOTSTRAP_PYTHON" -m venv "$build_dir"
"${build_dir}/bin/python" -m pip install --upgrade pip setuptools wheel uv

# SGLang owns the CUDA/PyTorch serving stack. The evolution runtime itself is
# deliberately installed as a small dependency set and imported from PROJECT_ROOT
# via PYTHONPATH, avoiding a second resolver changing SGLang's CUDA wheels.
"${build_dir}/bin/uv" pip install --python "${build_dir}/bin/python" \
  "sglang==${SGLANG_VERSION}" \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  "numpy>=1.24" \
  "matplotlib>=3.7" \
  "plotly>=5.20" \
  "huggingface_hub[hf_xet]>=0.34"

"${build_dir}/bin/python" - "$SGLANG_VERSION" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

expected = sys.argv[1]
actual = importlib.metadata.version("sglang")
if actual != expected:
    raise SystemExit(f"sglang version mismatch: expected {expected}, got {actual}")
payload = {
    "python": platform.python_version(),
    "sglang": actual,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
}
Path(sys.prefix, ".runtime-ready.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY
"${build_dir}/bin/python" -m pip freeze > "${build_dir}/.runtime-freeze.txt"

if [[ -e "$DEEPSEEK_ENV_DIR" ]]; then
  displaced_dir="${DEEPSEEK_ENV_DIR}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
  echo "[cluster-env] preserving incomplete prior environment at ${displaced_dir}"
  mv "$DEEPSEEK_ENV_DIR" "$displaced_dir"
fi
mv "$build_dir" "$DEEPSEEK_ENV_DIR"
echo "[cluster-env] ready: ${DEEPSEEK_ENV_DIR}"
