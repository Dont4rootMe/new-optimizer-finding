"""Validation and report helpers for unit-square circle packing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def coerce_run_output(run_output: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize `run_packing()` output to canonical numpy arrays."""

    if not isinstance(run_output, (tuple, list)) or len(run_output) != 3:
        raise ValueError("run_packing() must return a 3-tuple: (centers, radii, reported_sum).")

    centers_raw, radii_raw, reported_sum_raw = run_output
    centers = np.asarray(centers_raw, dtype=float)
    radii = np.asarray(radii_raw, dtype=float)
    reported_sum = float(reported_sum_raw)
    return centers, radii, reported_sum


def validate_circle_packing(
    *,
    centers: np.ndarray,
    radii: np.ndarray,
    reported_sum: float,
    num_circles: int,
    square_size: float,
    atol: float,
) -> None:
    """Validate one circle packing instance inside a square."""

    if centers.shape != (num_circles, 2):
        raise ValueError(f"Centers shape incorrect. Expected ({num_circles}, 2), got {centers.shape}.")
    if radii.shape != (num_circles,):
        raise ValueError(f"Radii shape incorrect. Expected ({num_circles},), got {radii.shape}.")

    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)) or not np.isfinite(reported_sum):
        raise ValueError("Centers, radii, and reported_sum must all be finite.")
    if np.any(radii < 0):
        negative_indices = np.where(radii < 0)[0].tolist()
        raise ValueError(f"Negative radii found at indices: {negative_indices}")

    radii_sum = float(np.sum(radii))
    if not np.isclose(radii_sum, reported_sum, atol=atol):
        raise ValueError(
            f"Sum of radii ({radii_sum:.12f}) does not match reported_sum ({reported_sum:.12f})."
        )

    lower = 0.0
    upper = float(square_size)
    for idx, ((x_coord, y_coord), radius) in enumerate(zip(centers, radii, strict=True)):
        is_outside = (
            x_coord - radius < lower - atol
            or x_coord + radius > upper + atol
            or y_coord - radius < lower - atol
            or y_coord + radius > upper + atol
        )
        if is_outside:
            raise ValueError(
                f"Circle {idx} lies outside the square: x={x_coord:.6f}, y={y_coord:.6f}, r={radius:.6f}."
            )

    for left in range(num_circles):
        for right in range(left + 1, num_circles):
            distance = float(np.linalg.norm(centers[left] - centers[right]))
            if distance < float(radii[left] + radii[right]) - atol:
                raise ValueError(
                    f"Circles {left} and {right} overlap. Dist={distance:.12f}, "
                    f"sum_radii={float(radii[left] + radii[right]):.12f}."
                )


def format_centers_string(centers: np.ndarray) -> str:
    """Render centers into a compact multi-line string."""

    return "\n".join(
        f"  centers[{idx}] = ({x_coord:.6f}, {y_coord:.6f})"
        for idx, (x_coord, y_coord) in enumerate(centers)
    )


def save_extra_artifact(
    *,
    organism_dir: str,
    experiment_name: str,
    centers: np.ndarray,
    radii: np.ndarray,
    reported_sum: float,
) -> str:
    """Persist packing diagnostics into an opaque `.npz` artifact."""

    out_path = Path(organism_dir) / "results" / f"{experiment_name}_extra.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, centers=centers, radii=radii, reported_sum=reported_sum)
    return str(out_path.resolve())
