#!/usr/bin/env bash
set -euo pipefail

# Build an immutable shared-NFS runtime for the DeepSeek server. A directory
# lock prevents concurrent jobs from corrupting the same environment; failed
# builds are preserved for diagnosis rather than deleted.

SGLANG_VERSION="${SGLANG_VERSION:-0.5.16}"
SGLANG_CUDA_VARIANT="${SGLANG_CUDA_VARIANT:-cu126}"
DEEPSEEK_ENV_DIR="${DEEPSEEK_ENV_DIR:?DEEPSEEK_ENV_DIR must be an explicit absolute path}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"

if [[ "$SGLANG_VERSION" != "0.5.16" || "$SGLANG_CUDA_VARIANT" != "cu126" ]]; then
  echo "Unaudited SGLang runtime: version=${SGLANG_VERSION} cuda=${SGLANG_CUDA_VARIANT}" >&2
  exit 2
fi

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
  if "${DEEPSEEK_ENV_DIR}/bin/python" - "$SGLANG_VERSION" "$SGLANG_CUDA_VARIANT" <<'PY'
import importlib.metadata
import sys
import torch
import sgl_kernel  # noqa: F401
import sglang  # noqa: F401

expected = sys.argv[1]
variant = sys.argv[2]
valid = (
    importlib.metadata.version("sglang") == expected
    and variant == "cu126"
    and str(torch.version.cuda).startswith("12.6")
    and torch.cuda.is_available()
)
raise SystemExit(0 if valid else 1)
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

# PyPI's SGLang 0.5.16 metadata defaults to CUDA 13.  Reproduce the upstream
# Dockerfile's explicit CUDA-12 branch: install cu126 PyTorch first, install the
# SGLang wheel without dependencies, then materialize the audited CUDA-12 view
# of its dependency metadata.  This keeps current DeepSeek-V4/DSPARK support
# while matching the R560/CUDA-12.6 cluster driver.
"${build_dir}/bin/uv" pip install --python "${build_dir}/bin/python" \
  --index-url "https://download.pytorch.org/whl/cu126" \
  "torch==2.11.0" \
  "torchvision==0.26.0" \
  "torchaudio==2.11.0"
"${build_dir}/bin/uv" pip install --python "${build_dir}/bin/python" \
  --no-deps "sglang==${SGLANG_VERSION}"

requirements_file="${build_dir}/.sglang-cu126-requirements.txt"
PYTHONPATH="${PROJECT_ROOT:?PROJECT_ROOT must be set}" \
  "${build_dir}/bin/python" -m scripts.cluster.sglang_runtime --output "$requirements_file"
"${build_dir}/bin/uv" pip install --python "${build_dir}/bin/python" \
  --requirements "$requirements_file" \
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
import sgl_kernel  # noqa: F401
import sglang  # noqa: F401

expected = sys.argv[1]
actual = importlib.metadata.version("sglang")
if actual != expected:
    raise SystemExit(f"sglang version mismatch: expected {expected}, got {actual}")
payload = {
    "python": platform.python_version(),
    "sglang": actual,
    "sglang_kernel": importlib.metadata.version("sglang-kernel"),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
}
if not str(torch.version.cuda).startswith("12.6"):
    raise SystemExit(f"expected a CUDA 12.6 torch build, got {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("the CUDA 12.6 runtime cannot initialize a cluster GPU")
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
