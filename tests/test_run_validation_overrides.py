"""Tests for run-only overrides via experiments.<name>.run_validation."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from src.validate.runner import ExperimentRunner

ROOT = Path(__file__).resolve().parents[1]


def _compose(overrides: list[str]):
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        return compose(config_name="config_optimization_survey", overrides=overrides)


def test_run_validation_overrides_apply_only_in_run_mode() -> None:
    cfg_run = _compose(
        [
            "mode=run",
            "+organism_dir=/tmp/test-organism",
            "experiments.cifar_convnet.run_validation.max_steps=123",
            "experiments.cifar_convnet.run_validation.target_quality=0.77",
        ]
    )
    run_prepared = ExperimentRunner(cfg_run)._prepare_experiment_cfg(
        "cifar_convnet",
        cfg_run.experiments.cifar_convnet,
        "run",
    )
    assert int(run_prepared.compute.max_steps) == 123
    assert float(run_prepared.target.value) == 0.77

    cfg_stats = _compose(
        [
            "mode=stats",
            "experiments.cifar_convnet.run_validation.max_steps=123",
            "experiments.cifar_convnet.run_validation.target_quality=0.77",
        ]
    )
    stats_prepared = ExperimentRunner(cfg_stats)._prepare_experiment_cfg(
        "cifar_convnet",
        cfg_stats.experiments.cifar_convnet,
        "stats",
    )
    assert int(stats_prepared.compute.max_steps) == 20000
    assert float(stats_prepared.target.value) == 0.90

    cfg_smoke = _compose(
        [
            "mode=smoke",
            "experiments.cifar_convnet.run_validation.max_steps=123",
        ]
    )
    smoke_prepared = ExperimentRunner(cfg_smoke)._prepare_experiment_cfg(
        "cifar_convnet",
        cfg_smoke.experiments.cifar_convnet,
        "smoke",
    )
    assert int(smoke_prepared.compute.max_steps) == int(smoke_prepared.compute.smoke_steps)
