"""Validate the pinned DeepGEMM NVCC path with a real Hopper MHC kernel."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import subprocess
from pathlib import Path

from scripts.cluster.common import atomic_write_json, utc_now


def parse_nvcc_version(output: str) -> str:
    match = re.search(r"\bV(\d+\.\d+\.\d+)\b", output)
    if match is None:
        raise ValueError("nvcc output contains no V<major>.<minor>.<patch> version")
    return match.group(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    compiler = args.compiler.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise SystemExit(f"DeepGEMM compiler is not executable: {compiler}")
    if cache_dir in {Path("/"), Path("/home"), Path("/home/jovyan")}:
        raise SystemExit(f"refusing unsafe DeepGEMM cache path: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    compiler_output = subprocess.run(
        [str(compiler), "--version"], check=True, capture_output=True, text=True
    ).stdout
    actual_version = parse_nvcc_version(compiler_output)
    if actual_version != args.expected_version:
        raise SystemExit(
            f"DeepGEMM compiler mismatch: expected {args.expected_version}, got {actual_version}"
        )

    # DeepGEMM captures compiler/cache configuration when its native module is
    # imported, so these assignments must precede all torch/deep_gemm imports.
    os.environ["DG_JIT_NVCC_COMPILER"] = str(compiler)
    os.environ["DG_JIT_CACHE_DIR"] = str(cache_dir)

    import torch
    import deep_gemm
    from deep_gemm.testing import calc_diff

    if not torch.cuda.is_available():
        raise SystemExit("DeepGEMM smoke requires a live CUDA device")
    torch.manual_seed(0)
    m, n, k = 13, 24, 7168
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.float32, device="cuda")
    output = torch.empty((m, n), dtype=torch.float32, device="cuda")
    square_sum = torch.empty((m,), dtype=torch.float32, device="cuda")
    deep_gemm.tf32_hc_prenorm_gemm(a, b, output, square_sum, num_splits=None)
    torch.cuda.synchronize()
    difference = max(
        float(calc_diff(output, a.float() @ b.T)),
        float(calc_diff(square_sum, a.float().square().sum(-1))),
    )
    if difference >= 1e-8:
        raise RuntimeError(f"DeepGEMM MHC numerical mismatch: {difference}")

    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "observed_at": utc_now(),
            "status": "passed",
            "compiler": str(compiler),
            "nvcc_version": actual_version,
            "deep_gemm_version": importlib.metadata.version("sgl-deep-gemm"),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "kernel": "tf32_hc_prenorm_gemm",
            "shape": {"m": m, "n": n, "k": k, "num_splits": None},
            "difference": difference,
            "cache_dir": str(cache_dir),
        },
    )


if __name__ == "__main__":
    main()
