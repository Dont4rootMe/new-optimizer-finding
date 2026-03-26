"""Tests for canonical baseline loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from valopt.utils.baselines import load_baseline_profile


def test_load_baseline_profile_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "stats" / "exp_a"
    path.mkdir(parents=True, exist_ok=True)
    baseline_file = path / "baseline.json"
    baseline_file.write_text(
        json.dumps(
            {
                "objective_name": "train_loss",
                "objective_direction": "min",
                "objective_last": 0.25,
                "steps": 42,
            }
        ),
        encoding="utf-8",
    )

    payload = load_baseline_profile(tmp_path / "stats", "exp_a")
    assert payload["objective_last"] == 0.25
    assert payload["steps"] == 42


def test_load_baseline_profile_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_baseline_profile(tmp_path / "stats", "exp_missing")


def test_load_baseline_profile_invalid(tmp_path: Path) -> None:
    path = tmp_path / "stats" / "exp_bad"
    path.mkdir(parents=True, exist_ok=True)
    baseline_file = path / "baseline.json"
    baseline_file.write_text(
        json.dumps(
            {
                "objective_name": "val_loss",
                "objective_direction": "min",
                "objective_last": 0.25,
                "steps": 42,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_baseline_profile(tmp_path / "stats", "exp_bad")
