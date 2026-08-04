"""Precompile and validate SGLang's Hopper custom-all-reduce JIT modules."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
from pathlib import Path

from scripts.cluster.common import atomic_write_json, utc_now
from scripts.cluster.smoke_deepgemm_toolchain import parse_nvcc_version


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-nvcc-version", required=True)
    parser.add_argument("--expected-tvm-ffi-version", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cuda_home = args.cuda_home.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    if cache_dir in {Path("/"), Path("/home"), Path("/home/jovyan")}:
        raise SystemExit(f"refusing unsafe TVM-FFI cache path: {cache_dir}")
    if args.world_size != 8:
        raise SystemExit(
            f"production SGLang JIT smoke requires TP world size 8, got {args.world_size}"
        )

    compiler = cuda_home / "bin" / "nvcc"
    cuda_library_dir = cuda_home / "targets" / "x86_64-linux" / "lib"
    cudart_link_library = cuda_library_dir / "libcudart.so"
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise SystemExit(f"CUDA compiler is not executable: {compiler}")
    if not cudart_link_library.is_file():
        raise SystemExit(f"CUDA link library is missing: {cudart_link_library}")
    compiler_output = subprocess.run(
        [str(compiler), "--version"], check=True, capture_output=True, text=True
    ).stdout
    actual_nvcc_version = parse_nvcc_version(compiler_output)
    if actual_nvcc_version != args.expected_nvcc_version:
        raise SystemExit(
            f"NVCC mismatch: expected {args.expected_nvcc_version}, got {actual_nvcc_version}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_HOME"] = str(cuda_home)
    library_path = [str(cuda_library_dir)] + [
        component
        for component in os.environ.get("LIBRARY_PATH", "").split(os.pathsep)
        if component and component != str(cuda_library_dir)
    ]
    os.environ["LIBRARY_PATH"] = os.pathsep.join(library_path)
    os.environ["TVM_FFI_CACHE_DIR"] = str(cache_dir)
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"

    import torch
    from sglang.jit_kernel.all_reduce import (
        _init_communicator,
        _init_ipc_manager,
        get_all_reduce_module,
    )

    actual_tvm_ffi_version = importlib.metadata.version("apache-tvm-ffi")
    if actual_tvm_ffi_version != args.expected_tvm_ffi_version:
        raise SystemExit(
            "TVM-FFI mismatch: "
            f"expected {args.expected_tvm_ffi_version}, got {actual_tvm_ffi_version}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("SGLang JIT smoke requires a live CUDA device")

    # These are the three modules SGLang needs before its TP=8 communicator can
    # use the NVLink custom-all-reduce path. Compiling them before server spawn
    # avoids an eight-rank JIT race and makes later jobs reuse content-addressed
    # modules from regional NFS.
    _init_ipc_manager()
    _init_communicator()
    get_all_reduce_module(torch.bfloat16, args.world_size)
    modules = sorted(str(path) for path in cache_dir.rglob("*.so"))
    expected_module_fragments = (
        "cuda_ipc",
        "communicator",
        "custom_all_reduce_bf16_t_8",
    )
    missing = [
        fragment
        for fragment in expected_module_fragments
        if not any(fragment in path for path in modules)
    ]
    if missing:
        raise RuntimeError(f"SGLang JIT cache is missing compiled modules: {missing}")

    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "observed_at": utc_now(),
            "status": "passed",
            "cuda_home": str(cuda_home),
            "cuda_library_dir": str(cuda_library_dir),
            "nvcc_version": actual_nvcc_version,
            "tvm_ffi_version": actual_tvm_ffi_version,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "world_size": args.world_size,
            "dtype": str(torch.bfloat16),
            "cache_dir": str(cache_dir),
            "modules": modules,
        },
    )


if __name__ == "__main__":
    main()
