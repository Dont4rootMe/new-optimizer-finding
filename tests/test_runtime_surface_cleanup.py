"""Regression checks for generic runtime cleanup."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_no_valopt_env_vars_in_active_surface() -> None:
    files = [
        ROOT / "README.md",
        ROOT / "conf" / "config_optimization_survey.yaml",
        ROOT / "conf" / "config_circle_packing_shinka.yaml",
        ROOT / "conf" / "config_awtf2025_heuristic.yaml",
        ROOT / "scripts" / "seed_population.sh",
        ROOT / "scripts" / "run_evolution.sh",
    ]
    for path in files:
        assert "VALOPT_" not in path.read_text(encoding="utf-8"), str(path)


def test_validate_runtime_no_longer_merges_global_safety() -> None:
    runner_text = (ROOT / "src" / "validate" / "runner.py").read_text(encoding="utf-8")
    run_one_text = (ROOT / "src" / "validate" / "run_one.py").read_text(encoding="utf-8")

    assert "self.cfg.safety" not in runner_text
    assert "cfg.safety" not in run_one_text


def test_user_facing_entrypoints_have_no_optimizer_default_config() -> None:
    files = [
        ROOT / "src" / "main.py",
        ROOT / "src" / "evolve" / "run.py",
        ROOT / "src" / "evolve" / "seed_run.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert 'config_name="config_optimization_survey"' not in text, str(path)


def test_canonical_presets_no_longer_embed_validation_mode_surface() -> None:
    for path in [
        ROOT / "conf" / "config_optimization_survey.yaml",
        ROOT / "conf" / "config_circle_packing_shinka.yaml",
        ROOT / "conf" / "config_awtf2025_heuristic.yaml",
    ]:
        text = path.read_text(encoding="utf-8")
        assert "\norganism_dir:" not in text, str(path)


def test_validation_presets_keep_standalone_organism_surface() -> None:
    assert not (ROOT / "conf" / "config_optimization_survey_validate.yaml").exists()
    assert not (ROOT / "conf" / "config_circle_packing_shinka_validate.yaml").exists()
    assert not (ROOT / "conf" / "config_awtf2025_heuristic_validate.yaml").exists()
