"""Tests for dynamic optimizer import."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from valopt.utils.import_utils import load_optimizer_builder

ROOT = Path(__file__).resolve().parents[1]


def test_import_optimizer_factory() -> None:
    optimizer_path = ROOT / "optimizer_guesses" / "examples" / "sgd_baseline.py"
    builder, resolved_path, optimizer_name = load_optimizer_builder(str(optimizer_path))

    assert callable(builder)

    model = nn.Linear(4, 2)
    controller = builder(model, 100)

    assert callable(getattr(controller, "step", None))
    assert callable(getattr(controller, "zero_grad", None))

    assert resolved_path == optimizer_path.resolve()
    assert optimizer_name == "SGDBaselineController"
