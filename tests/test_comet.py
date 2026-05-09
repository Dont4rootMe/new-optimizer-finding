"""Tests for the CometRunLogger wrapper.

The wrapper must be a no-op when disabled or when comet_ml is missing, and
must dispatch the right calls to a mock experiment when enabled.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from src.evolve.comet import CometRunLogger
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

    # Plotly HTML is logged as both an asset (versioned file) and a log_html (latest only).
    assert mock_experiment.log_asset.called
    assert mock_experiment.log_html.called


def test_enabled_logger_falls_back_to_no_op_when_comet_module_missing(monkeypatch: Any) -> None:
    import sys

    # Force the import of comet_ml to fail by injecting a sentinel that raises.
    class _Raising:
        def __getattr__(self, name: str) -> Any:
            raise ImportError("comet_ml not installed in this test")

    monkeypatch.setitem(sys.modules, "comet_ml", None)  # makes import comet_ml raise
    logger = CometRunLogger(
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
    assert logger.enabled is False


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
