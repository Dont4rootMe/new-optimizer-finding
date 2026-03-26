"""Tests for run result schema validation."""

from __future__ import annotations

from valopt.schemas import RunResult, validate_run_result_dict


def test_result_schema_validation() -> None:
    result = RunResult(
        run_id="20260303T000000Z_deadbeef",
        timestamp="20260303T000000Z",
        experiment_name="cifar_convnet",
        optimizer_path="/tmp/sgd.py",
        optimizer_name="sgd",
        final_quality=0.91,
        target_quality=0.90,
        converged=True,
        steps=1000,
        wall_time_sec=12.5,
        best_quality=0.92,
        seed=42,
        device="cuda",
        precision="bf16",
        resolved_config_path="/tmp/resolved.yaml",
        extra_metrics={"throughput": 1200.0},
        safety_flags={"detect_nan": True},
        final_metrics={"val_acc": 0.91},
        best_metrics={"val_acc": 0.92},
        samples_or_tokens_seen=50000,
        steps_to_target=900,
        status="ok",
        smoke=False,
    )

    payload = result.to_dict()
    validate_run_result_dict(payload)
