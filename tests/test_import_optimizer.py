"""Tests for dynamic optimizer import."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn

from experiments.optimization_survey._runtime.candidate_loader import load_candidate_builder

ROOT = Path(__file__).resolve().parents[1]


def test_import_candidate_factory() -> None:
    implementation_path = ROOT / "optimizer_guesses" / "examples" / "sgd_baseline.py"
    builder, resolved_path, candidate_name = load_candidate_builder(str(implementation_path))

    assert callable(builder)

    model = nn.Linear(4, 2)
    controller = builder(model, 100)

    assert callable(getattr(controller, "step", None))
    assert callable(getattr(controller, "zero_grad", None))

    assert resolved_path == implementation_path.resolve()
    assert candidate_name == "SGDBaselineController"


def test_import_candidate_rejects_invalid_builder_signature(tmp_path: Path) -> None:
    implementation_path = tmp_path / "invalid_builder.py"
    implementation_path.write_text(
        (
            "class InvalidBuilderController:\n"
            "    def __init__(self, cfg):\n"
            "        self.cfg = cfg\n\n"
            "    def step(self, weights, grads, activations, step_fn):\n"
            "        del weights, grads, activations, step_fn\n\n"
            "    def zero_grad(self, set_to_none=True):\n"
            "        del set_to_none\n\n"
            "def build_optimizer(cfg):\n"
            "    return InvalidBuilderController(cfg)\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="build_optimizer"):
        load_candidate_builder(str(implementation_path))


def test_import_candidate_rejects_missing_step_fn(tmp_path: Path) -> None:
    implementation_path = tmp_path / "missing_step_fn.py"
    implementation_path.write_text(
        (
            "class MissingStepFnController:\n"
            "    def __init__(self, model, max_steps):\n"
            "        self.model = model\n"
            "        self.max_steps = max_steps\n\n"
            "    def step(self, weights, grads, activations):\n"
            "        del weights, grads, activations\n\n"
            "    def zero_grad(self, set_to_none=True):\n"
            "        del set_to_none\n\n"
            "def build_optimizer(model, max_steps):\n"
            "    return MissingStepFnController(model, max_steps)\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="controller.step"):
        builder, _, _ = load_candidate_builder(str(implementation_path))
        builder(nn.Linear(4, 2), 100)

def test_candidate_loader_no_outdated_builder_alias() -> None:
    import experiments.optimization_survey._runtime.candidate_loader as candidate_loader

    assert not hasattr(candidate_loader, "BuildOptimizerCallable")
