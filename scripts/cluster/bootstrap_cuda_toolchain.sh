#!/usr/bin/env bash
set -euo pipefail

# DeepGEMM compiles its Hopper kernels at runtime.  Keep the compiler in an
# isolated, immutable prefix: the serving Python environment deliberately stays
# on cu126 for the heterogeneous SR008 driver pool, while NVIDIA's 12.9 NVCC is
# selected only for cubin generation through DG_JIT_NVCC_COMPILER.

DEEPGEMM_NVCC_VERSION="${DEEPGEMM_NVCC_VERSION:-12.9.86}"
CUDA_CURAND_VERSION="${CUDA_CURAND_VERSION:-10.3.10.19}"
DEEPGEMM_CUDA_TOOLCHAIN_DIR="${DEEPGEMM_CUDA_TOOLCHAIN_DIR:?DEEPGEMM_CUDA_TOOLCHAIN_DIR must be an explicit absolute path}"

if [[ "$DEEPGEMM_NVCC_VERSION" != "12.9.86" ]]; then
  echo "Unaudited DeepGEMM compiler version: ${DEEPGEMM_NVCC_VERSION}" >&2
  exit 2
fi
if [[ "$CUDA_CURAND_VERSION" != "10.3.10.19" ]]; then
  echo "Unaudited cuRAND development version: ${CUDA_CURAND_VERSION}" >&2
  exit 2
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "The pinned CUDA compiler toolchain supports x86_64 only" >&2
  exit 2
fi
case "$DEEPGEMM_CUDA_TOOLCHAIN_DIR" in
  /*) ;;
  *) echo "DEEPGEMM_CUDA_TOOLCHAIN_DIR must be absolute" >&2; exit 2 ;;
esac
if [[ "$DEEPGEMM_CUDA_TOOLCHAIN_DIR" == "/" || "$DEEPGEMM_CUDA_TOOLCHAIN_DIR" == "/home" || "$DEEPGEMM_CUDA_TOOLCHAIN_DIR" == "/home/jovyan" ]]; then
  echo "Refusing unsafe DEEPGEMM_CUDA_TOOLCHAIN_DIR: ${DEEPGEMM_CUDA_TOOLCHAIN_DIR}" >&2
  exit 2
fi

nvcc_path="${DEEPGEMM_CUDA_TOOLCHAIN_DIR}/bin/nvcc"
ready_marker="${DEEPGEMM_CUDA_TOOLCHAIN_DIR}/.toolchain-ready.json"
cuda_target_dir="${DEEPGEMM_CUDA_TOOLCHAIN_DIR}/targets/x86_64-linux"
curand_header="${cuda_target_dir}/include/curand.h"
curand_kernel_header="${cuda_target_dir}/include/curand_kernel.h"
validate_packages() {
  [[ -x "$nvcc_path" ]] || return 1
  [[ -f "${DEEPGEMM_CUDA_TOOLCHAIN_DIR}/targets/x86_64-linux/lib/libcudart.so" ]] || return 1
  [[ -f "$curand_header" && -f "$curand_kernel_header" ]] || return 1
  "$nvcc_path" --version | grep -F "V${DEEPGEMM_NVCC_VERSION}" >/dev/null
}
validate_toolchain() {
  validate_packages || return 1
  [[ -f "$ready_marker" ]] || return 1
  grep -F '"sm90a_curand_cubin_smoke": "passed"' "$ready_marker" >/dev/null || return 1
}

if validate_toolchain; then
  echo "[cuda-toolchain] reusing ${DEEPGEMM_CUDA_TOOLCHAIN_DIR} (nvcc=${DEEPGEMM_NVCC_VERSION})"
  exit 0
fi

parent_dir="$(dirname "$DEEPGEMM_CUDA_TOOLCHAIN_DIR")"
mkdir -p "$parent_dir"
lock_dir="${DEEPGEMM_CUDA_TOOLCHAIN_DIR}.build.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "[cuda-toolchain] another process is building ${DEEPGEMM_CUDA_TOOLCHAIN_DIR}; waiting"
  for _attempt in $(seq 1 240); do
    if validate_toolchain; then
      echo "[cuda-toolchain] toolchain became ready"
      exit 0
    fi
    sleep 15
  done
  echo "[cuda-toolchain] timed out waiting for ${lock_dir}; inspect or move a stale lock" >&2
  exit 1
fi

cleanup_lock() {
  rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup_lock EXIT INT TERM

if [[ -e "$DEEPGEMM_CUDA_TOOLCHAIN_DIR" ]] && ! validate_packages; then
  displaced_dir="${DEEPGEMM_CUDA_TOOLCHAIN_DIR}.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
  echo "[cuda-toolchain] preserving incomplete prefix at ${displaced_dir}"
  mv "$DEEPGEMM_CUDA_TOOLCHAIN_DIR" "$displaced_dir"
fi

if [[ -n "${CUDA_TOOLCHAIN_CONDA:-}" ]]; then
  conda_executable="$CUDA_TOOLCHAIN_CONDA"
elif [[ -x "/home/user/conda/bin/conda" ]]; then
  conda_executable="/home/user/conda/bin/conda"
elif command -v conda >/dev/null 2>&1; then
  conda_executable="$(command -v conda)"
else
  echo "No conda executable is available for the CUDA compiler bootstrap" >&2
  exit 1
fi

if validate_packages; then
  echo "[cuda-toolchain] finishing complete unpublished prefix ${DEEPGEMM_CUDA_TOOLCHAIN_DIR}"
else
  package_cache="${CUDA_TOOLCHAIN_CONDA_PKGS_DIR:-${parent_dir}/.conda-pkgs}"
  mkdir -p "$package_cache"
  export CONDA_PKGS_DIRS="$package_cache"
  "$conda_executable" create --yes \
    --prefix "$DEEPGEMM_CUDA_TOOLCHAIN_DIR" \
    --override-channels \
    --channel nvidia \
    --channel conda-forge \
    "cuda-nvcc=${DEEPGEMM_NVCC_VERSION}" \
    "libcurand-dev=${CUDA_CURAND_VERSION}"
fi

"$nvcc_path" --version | grep -F "V${DEEPGEMM_NVCC_VERSION}" >/dev/null
test -f "${cuda_target_dir}/lib/libcudart.so"
test -f "$curand_header"
test -f "$curand_kernel_header"
smoke_dir="$(mktemp -d /tmp/evolutionloop-nvcc-smoke.XXXXXX)"
cleanup_smoke() {
  rm -rf -- "$smoke_dir"
}
trap 'cleanup_smoke; cleanup_lock' EXIT INT TERM
python3 - "$smoke_dir/kernel.cu" <<'PY'
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    r'''#include <stdint.h>
#include <curand_kernel.h>
__device__ __forceinline__ void st128(const __int128_t* ptr, __int128_t val) {
  asm volatile("st.shared.b128 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "q"(val));
}
extern "C" __global__ void smoke(const __int128_t* input, __int128_t* output) {
  curandStatePhilox4_32_10_t random_state;
  curand_init(7, threadIdx.x, 0, &random_state);
  __shared__ __int128_t value;
  st128(&value, input[0]);
  __syncthreads();
  if (threadIdx.x == 0 && curand_uniform(&random_state) > 0.0f) output[0] = value;
}
''',
    encoding="utf-8",
)
PY
"$nvcc_path" "$smoke_dir/kernel.cu" \
  --std=c++17 --gpu-architecture=sm_90a --cubin \
  --output-file "$smoke_dir/kernel.cubin"
test -s "$smoke_dir/kernel.cubin"

"$conda_executable" list --prefix "$DEEPGEMM_CUDA_TOOLCHAIN_DIR" --explicit \
  > "${DEEPGEMM_CUDA_TOOLCHAIN_DIR}/.conda-explicit.txt"
python3 - "$ready_marker" "$DEEPGEMM_NVCC_VERSION" "$CUDA_CURAND_VERSION" "$nvcc_path" <<'PY'
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "architecture": platform.machine(),
            "compiler": sys.argv[4],
            "curand_version": sys.argv[3],
            "nvcc_version": sys.argv[2],
            "published_at": datetime.now(timezone.utc).isoformat(),
            "sm90a_int128_cubin_smoke": "passed",
            "sm90a_curand_cubin_smoke": "passed",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
echo "[cuda-toolchain] ready: ${DEEPGEMM_CUDA_TOOLCHAIN_DIR}"
