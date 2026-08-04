#!/usr/bin/env bash
set -euo pipefail

# Lightweight one-H100 acceptance job for the exact compiler/runtime pairing.
# It intentionally avoids loading the model; production repeats the same smoke
# before SGLang starts.

global_rank="${RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${GROUP_RANK:-${LOCAL_RANK:-0}}}}}"
case "$global_rank" in
  ''|*[!0-9]*) global_rank=0 ;;
esac
if [[ "$global_rank" -ne 0 ]]; then
  echo "[deepgemm-smoke] global rank ${global_rank}: coordinator owned by rank 0; exiting"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${DEEPSEEK_CUDA_VISIBLE_DEVICES:-0}"
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

export DEEPSEEK_ENV_DIR="${DEEPSEEK_ENV_DIR:-${JOB_ROOT}/runtime/sglang-0.5.16-cu126}"
export DEEPGEMM_NVCC_VERSION="${DEEPGEMM_NVCC_VERSION:-12.9.86}"
export CUDA_CURAND_VERSION="${CUDA_CURAND_VERSION:-10.3.10.19}"
export DEEPGEMM_CUDA_TOOLCHAIN_DIR="${DEEPGEMM_CUDA_TOOLCHAIN_DIR:-${JOB_ROOT}/toolchains/cuda-nvcc-${DEEPGEMM_NVCC_VERSION}-curand-${CUDA_CURAND_VERSION}}"
export SGLANG_DG_CACHE_DIR="${SGLANG_DG_CACHE_DIR:-${JOB_ROOT}/kernel_cache/deep_gemm-sm90-cuda-nvcc-${DEEPGEMM_NVCC_VERSION}}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-${JOB_ROOT}/kernel_cache/tvm-ffi-sm90-cuda-nvcc-${DEEPGEMM_NVCC_VERSION}-tvmffi-0.1.11}"
export SGLANG_VERSION="${SGLANG_VERSION:-0.5.16}"
export SGLANG_CUDA_VARIANT="${SGLANG_CUDA_VARIANT:-cu126}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "$PROJECT_ROOT"
source "${PROJECT_ROOT}/scripts/cluster/cuda_driver_env.sh"
sanitize_cuda_driver_path
bash "${PROJECT_ROOT}/scripts/cluster/bootstrap_cuda_toolchain.sh"
bash "${PROJECT_ROOT}/scripts/cluster/bootstrap_deepseek_env.sh"
mkdir -p "$SGLANG_DG_CACHE_DIR"
mkdir -p "$TVM_FFI_CACHE_DIR"
export DG_JIT_NVCC_COMPILER="${DEEPGEMM_CUDA_TOOLCHAIN_DIR}/bin/nvcc"
export DG_JIT_PRINT_COMPILER_COMMAND=1
export CUDA_HOME="$DEEPGEMM_CUDA_TOOLCHAIN_DIR"
cuda_library_dir="${CUDA_HOME}/targets/x86_64-linux/lib"
export LIBRARY_PATH="${cuda_library_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export TVM_FFI_CUDA_ARCH_LIST=9.0a
output_path="${DEEPGEMM_SMOKE_OUTPUT:-${JOB_ROOT}/toolchains/deepgemm-toolchain-smoke.json}"
"${DEEPSEEK_ENV_DIR}/bin/python" -m scripts.cluster.smoke_deepgemm_toolchain \
  --compiler "$DG_JIT_NVCC_COMPILER" \
  --cache-dir "$SGLANG_DG_CACHE_DIR" \
  --expected-version "$DEEPGEMM_NVCC_VERSION" \
  --output "$output_path"
sglang_output_path="${SGLANG_JIT_SMOKE_OUTPUT:-${JOB_ROOT}/toolchains/sglang-jit-toolchain-smoke.json}"
"${DEEPSEEK_ENV_DIR}/bin/python" -m scripts.cluster.smoke_sglang_jit_toolchain \
  --cuda-home "$CUDA_HOME" \
  --cache-dir "$TVM_FFI_CACHE_DIR" \
  --expected-nvcc-version "$DEEPGEMM_NVCC_VERSION" \
  --expected-tvm-ffi-version 0.1.11 \
  --world-size 8 \
  --output "$sglang_output_path"
echo "[deepgemm-smoke] passed: ${output_path}; ${sglang_output_path}"
