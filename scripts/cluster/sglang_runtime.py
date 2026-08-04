"""Build the CUDA 12.6 dependency view for the pinned SGLang wheel.

SGLang 0.5.16's PyPI metadata defaults to CUDA 13.  Upstream's own Dockerfile
supports CUDA 12 by rewriting the same dependency edges before installation.
This module makes that rewrite explicit, version-locked, and testable for the
ML Space H100 nodes whose R560 driver exposes CUDA Driver API 12.6.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import re
from pathlib import Path
from typing import Iterable

from scripts.cluster.common import SGLANG_VERSION

PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"
TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"
TORCHAUDIO_VERSION = "2.11.0"
TORCH_CUDA_PREFIX = "12.6"

_KERNEL_URL = (
    "https://github.com/sgl-project/whl/releases/download/v0.4.5/"
    "sglang_kernel-0.4.5+cu124-cp310-abi3-manylinux2014_x86_64.whl"
)
_DEEP_GEMM_URL = (
    "https://github.com/sgl-project/whl/releases/download/v0.1.4.post1/"
    "sgl_deep_gemm-0.1.4.post1+cu129-py3-none-manylinux2014_x86_64.whl"
)

_CUDA12_REPLACEMENTS = {
    "cuda-python": "cuda-python>=12,<13",
    "flashinfer-python": "flashinfer_python[cu12]==0.6.14",
    # Upstream removes CUDA-13 extras for its CUDA-12 Docker variants.
    "humming-kernels": "humming-kernels==0.1.10",
    "nvidia-cutlass-dsl": "nvidia-cutlass-dsl==4.6.0",
    # These are the exact CUDA-12 wheels used by upstream's Docker recipe.
    "sglang-kernel": f"sglang-kernel @ {_KERNEL_URL}",
    "sgl-deep-gemm": f"sgl-deep-gemm @ {_DEEP_GEMM_URL}",
}


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if match is None:
        raise ValueError(f"cannot parse SGLang requirement: {requirement!r}")
    return match.group(1).lower().replace("_", "-").replace(".", "-")


def cuda126_requirements(requirements: Iterable[str], *, machine: str | None = None) -> list[str]:
    """Return SGLang's base requirements rewritten like upstream CUDA-12 Docker."""

    if SGLANG_VERSION != "0.5.16":
        raise RuntimeError(f"CUDA 12.6 recipe has not been audited for SGLang {SGLANG_VERSION}")
    architecture = machine or platform.machine()
    if architecture != "x86_64":
        raise RuntimeError(f"the pinned H100 runtime supports x86_64 only, found {architecture}")

    rewritten: list[str] = []
    replaced: set[str] = set()
    for raw_requirement in requirements:
        # importlib.metadata includes optional extras.  The serving job needs
        # only the base SRT dependency set, not diffusion/test/ray extras.
        if re.search(r"\bextra\s*==", raw_requirement):
            continue
        name = _requirement_name(raw_requirement)
        replacement = _CUDA12_REPLACEMENTS.get(name)
        if replacement is not None:
            rewritten.append(replacement)
            replaced.add(name)
        else:
            rewritten.append(raw_requirement)

    missing = set(_CUDA12_REPLACEMENTS) - replaced
    if missing:
        raise RuntimeError(
            "SGLang dependency metadata drifted; missing CUDA rewrites for "
            + ", ".join(sorted(missing))
        )
    return rewritten


def write_installed_requirements(output: Path) -> Path:
    raw = importlib.metadata.requires("sglang")
    if not raw:
        raise RuntimeError("installed SGLang distribution exposes no requirements")
    lines = cuda126_requirements(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    destination = write_installed_requirements(args.output)
    print(destination)


if __name__ == "__main__":
    main()
