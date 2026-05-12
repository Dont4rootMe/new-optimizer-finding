"""Tests for the CometRunLogger wrapper.

The wrapper must be a no-op when disabled or when comet_ml is missing, and
must dispatch the right calls to a mock experiment when enabled.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from src.evolve.comet import CometInitError, CometRunLogger
from src.evolve.visualization import RenderedSnapshot


def _make_snapshot(tmp_path: Path) -> RenderedSnapshot:
    overview_dir = tmp_path / "viz" / "overview"
    timeline_dir = tmp_path / "viz" / "timeline"
    survival_dir = tmp_path / "viz" / "survival"
    interactive_dir = tmp_path / "viz" / "interactive"
    for path in (overview_dir, timeline_dir, survival_dir, interactive_dir):
        path.mkdir(parents=True, exist_ok=True)

    composite = tmp_path / "evolution_overview.png"
    composite.write_bytes(b"\x89PNG\r\n\x1a\n")
    panel_a = overview_dir / "best_vs_evaluations.png"
    panel_a.write_bytes(b"\x89PNG\r\n\x1a\n")
    timeline_a = timeline_dir / "cumulative_evaluations_by_island.png"
    timeline_a.write_bytes(b"\x89PNG\r\n\x1a\n")
    survival_a = survival_dir / "by_runtime.png"
    survival_a.write_bytes(b"\x89PNG\r\n\x1a\n")
    interactive_a = interactive_dir / "best_vs_evaluations.html"
    interactive_a.write_text("<html><body>plotly</body></html>", encoding="utf-8")

    return RenderedSnapshot(
        population_root=tmp_path,
        generation=3,
        composite_overview_path=composite,
        overview_panels={"best_vs_evaluations": panel_a},
        timeline_panels={"cumulative_evaluations_by_island": timeline_a},
        survival_panels={"by_runtime": survival_a},
        interactive_html={"best_vs_evaluations": interactive_a},
    )


def test_from_cfg_prefers_run_name_over_legacy_experiment_name(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """When both `comet.run_name` and `comet.experiment_name` are set,
    ``run_name`` wins (it is the new explicit knob)."""

    mock_experiment = MagicMock()
    mock_experiment.get_key.return_value = "exp-key-1"
    fake_module = SimpleNamespace(
        Experiment=lambda *args, **kwargs: mock_experiment,
        ExistingExperiment=lambda *args, **kwargs: MagicMock(),
    )
    import sys

    monkeypatch.setitem(sys.modules, "comet_ml", fake_module)
    cfg = OmegaConf.create(
        {
            "comet": {
                "enabled": True,
                "api_key": "fake",
                "project_name": "proj",
                "workspace": None,
                "run_name": "my-explicit-run",
                "experiment_name": "legacy-name-should-be-ignored",
            }
        }
    )
    CometRunLogger.from_cfg(cfg, run_label="evolution", population_root=tmp_path)
    mock_experiment.set_name.assert_called_with("my-explicit-run")


def test_from_cfg_falls_back_to_legacy_experiment_name_when_run_name_missing(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Old YAMLs that only set `experiment_name` keep working."""

    mock_experiment = MagicMock()
    mock_experiment.get_key.return_value = "exp-key-2"
    fake_module = SimpleNamespace(
        Experiment=lambda *args, **kwargs: mock_experiment,
        ExistingExperiment=lambda *args, **kwargs: MagicMock(),
    )
    import sys

    monkeypatch.setitem(sys.modules, "comet_ml", fake_module)
    cfg = OmegaConf.create(
        {
            "comet": {
                "enabled": True,
                "api_key": "fake",
                "project_name": "proj",
                "workspace": None,
                "experiment_name": "legacy-only-name",
            }
        }
    )
    CometRunLogger.from_cfg(cfg, run_label="evolution", population_root=tmp_path)
    mock_experiment.set_name.assert_called_with("legacy-only-name")


def test_from_cfg_returns_disabled_logger_when_block_missing() -> None:
    cfg = OmegaConf.create({"seed": 1})
    logger = CometRunLogger.from_cfg(cfg)
    assert logger.enabled is False
    # All methods are no-ops on a disabled logger.
    logger.log_metrics({"score": 1.0})
    logger.log_parameters({"foo": "bar"})
    logger.end()


def test_from_cfg_returns_disabled_logger_when_enabled_false() -> None:
    cfg = OmegaConf.create({"comet": {"enabled": False, "api_key": None}})
    logger = CometRunLogger.from_cfg(cfg)
    assert logger.enabled is False


def test_disabled_logger_skips_snapshot(tmp_path: Path) -> None:
    cfg = OmegaConf.create({"comet": {"enabled": False}})
    logger = CometRunLogger.from_cfg(cfg)
    snapshot = _make_snapshot(tmp_path)
    # Must not raise.
    logger.log_snapshot(snapshot)


def _enabled_logger_with_mock_experiment(monkeypatch: Any, mock_experiment: Any) -> CometRunLogger:
    """Patch comet_ml.Experiment with a class that returns the mock.

    We need the wrapper to call into comet_ml.Experiment(...) successfully and
    receive our mock back, so we monkeypatch the import target in sys.modules.
    """

    fake_module = SimpleNamespace(Experiment=lambda *args, **kwargs: mock_experiment)
    import sys

    monkeypatch.setitem(sys.modules, "comet_ml", fake_module)
    return CometRunLogger(
        enabled=True,
        api_key="fake-key",
        project_name="proj",
        workspace=None,
        experiment_name=None,
        log_combined_overview=True,
        log_individual_panels=True,
        log_plotly=True,
        log_step_per_generation=True,
        run_label="test-run",
    )


def test_enabled_logger_uploads_grouped_artifacts(monkeypatch: Any, tmp_path: Path) -> None:
    mock_experiment = MagicMock()
    logger = _enabled_logger_with_mock_experiment(monkeypatch, mock_experiment)
    assert logger.enabled is True

    snapshot = _make_snapshot(tmp_path)
    logger.log_snapshot(snapshot)

    image_calls = mock_experiment.log_image.call_args_list
    image_names = {kwargs.get("name") for _, kwargs in image_calls}
    assert "overview/composite" in image_names
    assert "overview/best_vs_evaluations" in image_names
    assert "timeline/cumulative_evaluations_by_island" in image_names
    assert "survival/by_runtime" in image_names

    # Each image call uses the snapshot's generation as the Comet step.
    for _, kwargs in image_calls:
        assert kwargs.get("step") == 3

    # Plotly HTML is uploaded as a versioned asset only (log_html appends
    # into a single experiment-wide slot, which is wrong when we ship many
    # panels — see CometRunLogger._upload_html for the rationale).
    assert mock_experiment.log_asset.called
    assert not mock_experiment.log_html.called


def test_enabled_logger_raises_when_comet_module_missing(monkeypatch: Any) -> None:
    """Hard-fail policy: silent disable on a remote run would hide a config
    bug for the entire run duration. When the operator says enabled=true they
    expect telemetry; we raise so the run dies immediately and the operator
    sees the misconfiguration.
    """

    import sys

    monkeypatch.setitem(sys.modules, "comet_ml", None)  # makes import comet_ml raise
    with pytest.raises(CometInitError, match="comet_ml package is NOT installed"):
        CometRunLogger(
            enabled=True,
            api_key="fake",
            project_name="proj",
            workspace=None,
            experiment_name=None,
            log_combined_overview=True,
            log_individual_panels=True,
            log_plotly=True,
            log_step_per_generation=True,
        )


def test_enabled_logger_raises_when_api_key_missing(monkeypatch: Any) -> None:
    """Same hard-fail policy for missing API key — keep the operator honest."""

    fake_module = SimpleNamespace(Experiment=lambda *args, **kwargs: MagicMock())
    import sys

    monkeypatch.setitem(sys.modules, "comet_ml", fake_module)
    with pytest.raises(CometInitError, match="no api_key was resolved"):
        CometRunLogger(
            enabled=True,
            api_key=None,
            project_name="proj",
            workspace=None,
            experiment_name=None,
            log_combined_overview=True,
            log_individual_panels=True,
            log_plotly=True,
            log_step_per_generation=True,
        )


def test_enabled_logger_raises_when_experiment_init_fails(monkeypatch: Any) -> None:
    """Network / workspace failures propagate as CometInitError too."""

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("workspace lookup timed out")

    fake_module = SimpleNamespace(Experiment=_raise)
    import sys

    monkeypatch.setitem(sys.modules, "comet_ml", fake_module)
    with pytest.raises(CometInitError, match="experiment failed to start"):
        CometRunLogger(
            enabled=True,
            api_key="fake",
            project_name="proj",
            workspace="bad-workspace",
            experiment_name=None,
            log_combined_overview=True,
            log_individual_panels=True,
            log_plotly=True,
            log_step_per_generation=True,
        )


def test_enabled_logger_persists_experiment_key_to_population_root(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """After successful init the experiment_key is written so a follow-up
    process (e.g. evolve after seed) can resume the same Comet experiment.
    """

    mock_experiment = MagicMock()
    mock_experiment.get_key.return_value = "exp-key-abc"
    fake_module = SimpleNamespace(
        Experiment=lambda *args, **kwargs: mock_experiment,
        ExistingExperiment=lambda *args, **kwargs: MagicMock(),  # not used in this path
    )
    import sys

    monkeypatch.setitem(sys.modules, "comet_ml", fake_module)
    population_root = tmp_path / "pop"

    CometRunLogger(
        enabled=True,
        api_key="fake",
        project_name="proj",
        workspace="ws",
        experiment_name="run-1",
        log_combined_overview=True,
        log_individual_panels=True,
        log_plotly=True,
        log_step_per_generation=True,
        population_root=population_root,
    )

    cache_path = population_root / "comet_experiment.json"
    assert cache_path.exists()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["experiment_key"] == "exp-key-abc"
    assert payload["project_name"] == "proj"
    assert payload["workspace"] == "ws"


def test_enabled_logger_resumes_existing_experiment_when_cache_present(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Seed + evolve share one Comet experiment: the second process sees the
    cached key and calls ``ExistingExperiment`` instead of ``Experiment``.
    """

    existing_experiment = MagicMock()
    existing_experiment.get_key.return_value = "exp-key-abc"

    new_experiment_calls: list[dict[str, Any]] = []

    def _new_experiment(*args: Any, **kwargs: Any) -> Any:
        new_experiment_calls.append(kwargs)
        return MagicMock()

    existing_experiment_calls: list[dict[str, Any]] = []

    def _existing_experiment(*args: Any, **kwargs: Any) -> Any:
        existing_experiment_calls.append(kwargs)
        return existing_experiment

    fake_module = SimpleNamespace(
        Experiment=_new_experiment,
        ExistingExperiment=_existing_experiment,
    )
    import sys

    monkeypatch.setitem(sys.modules, "comet_ml", fake_module)
    population_root = tmp_path / "pop"
    population_root.mkdir()
    (population_root / "comet_experiment.json").write_text(
        json.dumps(
            {
                "experiment_key": "exp-key-abc",
                "project_name": "proj",
                "workspace": "ws",
            }
        ),
        encoding="utf-8",
    )

    CometRunLogger(
        enabled=True,
        api_key="fake",
        project_name="proj",
        workspace="ws",
        experiment_name=None,
        log_combined_overview=True,
        log_individual_panels=True,
        log_plotly=True,
        log_step_per_generation=True,
        population_root=population_root,
    )

    # Resume path was taken: ExistingExperiment was called once, Experiment never.
    assert len(existing_experiment_calls) == 1
    assert existing_experiment_calls[0]["previous_experiment"] == "exp-key-abc"
    assert new_experiment_calls == []


def test_resume_ignores_cache_when_project_workspace_mismatch(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """If the user changed project/workspace between runs, the cached key is
    stale — we start a fresh experiment rather than blindly resuming.
    """

    new_experiment_calls: list[dict[str, Any]] = []

    def _new_experiment(*args: Any, **kwargs: Any) -> Any:
        new_experiment_calls.append(kwargs)
        mock = MagicMock()
        mock.get_key.return_value = "new-exp"
        return mock

    existing_experiment_calls: list[dict[str, Any]] = []

    def _existing_experiment(*args: Any, **kwargs: Any) -> Any:
        existing_experiment_calls.append(kwargs)
        return MagicMock()

    fake_module = SimpleNamespace(
        Experiment=_new_experiment,
        ExistingExperiment=_existing_experiment,
    )
    import sys

    monkeypatch.setitem(sys.modules, "comet_ml", fake_module)
    population_root = tmp_path / "pop"
    population_root.mkdir()
    (population_root / "comet_experiment.json").write_text(
        json.dumps(
            {
                "experiment_key": "exp-key-abc",
                "project_name": "old-proj",
                "workspace": "ws",
            }
        ),
        encoding="utf-8",
    )

    CometRunLogger(
        enabled=True,
        api_key="fake",
        project_name="new-proj",  # different from cached
        workspace="ws",
        experiment_name=None,
        log_combined_overview=True,
        log_individual_panels=True,
        log_plotly=True,
        log_step_per_generation=True,
        population_root=population_root,
    )

    # Cache mismatch → fresh experiment, no resume.
    assert len(new_experiment_calls) == 1
    assert existing_experiment_calls == []


def test_enabled_logger_skips_plotly_when_log_plotly_disabled(monkeypatch: Any, tmp_path: Path) -> None:
    mock_experiment = MagicMock()
    fake_module = SimpleNamespace(Experiment=lambda *args, **kwargs: mock_experiment)
    import sys

    monkeypatch.setitem(sys.modules, "comet_ml", fake_module)
    logger = CometRunLogger(
        enabled=True,
        api_key="fake",
        project_name="proj",
        workspace=None,
        experiment_name=None,
        log_combined_overview=True,
        log_individual_panels=True,
        log_plotly=False,
        log_step_per_generation=True,
    )
    snapshot = _make_snapshot(tmp_path)
    logger.log_snapshot(snapshot)

    assert mock_experiment.log_image.called
    assert not mock_experiment.log_html.called
    assert not mock_experiment.log_asset.called
