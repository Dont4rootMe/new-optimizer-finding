"""Optional Comet ML telemetry wrapper for the evolution loop.

The wrapper is a thin shim around ``comet_ml.Experiment`` configured via
``cfg.comet`` in the project's Hydra config. Its design goals:

- *Opt-in*: nothing happens unless ``cfg.comet.enabled`` is true.
- *Soft-fail*: if ``comet_ml`` isn't installed, the API key is missing, or
  the network is unreachable, the wrapper logs a warning and continues as a
  no-op. It must never raise into the evolution loop.
- *Idempotent*: a single ``CometRunLogger`` is created per run; per-snapshot
  uploads use a stable image-name scheme (group prefix in the asset name)
  so the Comet UI groups them automatically.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


_DEFAULT_PROJECT_NAME = "evolution-runs"


class CometRunLogger:
    """Per-run wrapper that uploads visualization snapshots to Comet.

    Only the public methods (`log_snapshot`, `log_metrics`, `end`) are stable;
    everything else is private. If ``comet_ml`` is missing or the experiment
    fails to initialize, the wrapper silently degrades to a no-op.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        api_key: str | None,
        project_name: str | None,
        workspace: str | None,
        experiment_name: str | None,
        log_combined_overview: bool,
        log_individual_panels: bool,
        log_plotly: bool,
        log_step_per_generation: bool,
        run_label: str = "evolution",
    ) -> None:
        self._enabled = enabled
        self._log_combined_overview = log_combined_overview
        self._log_individual_panels = log_individual_panels
        self._log_plotly = log_plotly
        self._log_step_per_generation = log_step_per_generation
        self._experiment: Any | None = None

        if not enabled:
            return

        try:
            import comet_ml  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("comet_ml is not available; Comet logging disabled (%s)", exc)
            self._enabled = False
            return

        try:
            self._experiment = comet_ml.Experiment(
                api_key=api_key,
                project_name=project_name or _DEFAULT_PROJECT_NAME,
                workspace=workspace,
                log_code=False,
                log_graph=False,
                auto_param_logging=False,
                auto_metric_logging=False,
                auto_output_logging="simple",
            )
            resolved_name = experiment_name or _default_experiment_name(run_label)
            try:
                self._experiment.set_name(resolved_name)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to set Comet experiment name")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to start Comet experiment; disabling logging (%s)", exc)
            self._experiment = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self._experiment is not None)

    @classmethod
    def from_cfg(cls, cfg: Any, *, run_label: str = "evolution") -> "CometRunLogger":
        """Build a logger from the ``cfg.comet`` block of a Hydra config.

        Returns a no-op logger if the block is missing or ``enabled`` is false.
        """

        comet_cfg = _get_attr(cfg, "comet", default=None)
        if comet_cfg is None:
            return cls(
                enabled=False,
                api_key=None,
                project_name=None,
                workspace=None,
                experiment_name=None,
                log_combined_overview=True,
                log_individual_panels=True,
                log_plotly=True,
                log_step_per_generation=True,
                run_label=run_label,
            )
        return cls(
            enabled=bool(_get_attr(comet_cfg, "enabled", default=False)),
            api_key=_optional_str(_get_attr(comet_cfg, "api_key", default=None)),
            project_name=_optional_str(_get_attr(comet_cfg, "project_name", default=None)),
            workspace=_optional_str(_get_attr(comet_cfg, "workspace", default=None)),
            experiment_name=_optional_str(_get_attr(comet_cfg, "experiment_name", default=None)),
            log_combined_overview=bool(_get_attr(comet_cfg, "log_combined_overview", default=True)),
            log_individual_panels=bool(_get_attr(comet_cfg, "log_individual_panels", default=True)),
            log_plotly=bool(_get_attr(comet_cfg, "log_plotly", default=True)),
            log_step_per_generation=bool(_get_attr(comet_cfg, "log_step_per_generation", default=True)),
            run_label=run_label,
        )

    def log_snapshot(self, snapshot: Any) -> None:
        """Upload the bundle produced by ``render_evolution_snapshot``.

        ``snapshot`` is a :class:`src.evolve.visualization.RenderedSnapshot`.
        We avoid importing it here to keep this module lazily-importable from
        ``evolution_loop`` even when ``visualization`` hasn't been imported yet.
        """

        if not self.enabled:
            return

        step = getattr(snapshot, "generation", None)
        if not self._log_step_per_generation:
            step = None

        try:
            composite = getattr(snapshot, "composite_overview_path", None)
            if self._log_combined_overview and composite is not None:
                self._upload_image(Path(composite), "overview/composite", step=step)

            if self._log_individual_panels:
                for name, path in dict(getattr(snapshot, "overview_panels", {}) or {}).items():
                    self._upload_image(Path(path), f"overview/{name}", step=step)
                for name, path in dict(getattr(snapshot, "timeline_panels", {}) or {}).items():
                    self._upload_image(Path(path), f"timeline/{name}", step=step)
                for name, path in dict(getattr(snapshot, "survival_panels", {}) or {}).items():
                    self._upload_image(Path(path), f"survival/{name}", step=step)

            if self._log_plotly:
                for name, path in dict(getattr(snapshot, "interactive_html", {}) or {}).items():
                    self._upload_html(Path(path), f"interactive/{name}", step=step)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to log snapshot to Comet (continuing)")

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        if not self.enabled:
            return
        try:
            self._experiment.log_metrics(dict(metrics), step=step)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to log metrics to Comet (continuing)")

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._experiment.log_parameters(dict(parameters))
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to log parameters to Comet (continuing)")

    def end(self) -> None:
        if not self.enabled or self._experiment is None:
            return
        try:
            self._experiment.end()
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to end Comet experiment cleanly")
        finally:
            self._experiment = None
            self._enabled = False

    def _upload_image(self, path: Path, asset_name: str, *, step: int | None) -> None:
        if not path.exists():
            return
        try:
            self._experiment.log_image(str(path), name=asset_name, step=step)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to log image %s as %s", path, asset_name)

    def _upload_html(self, path: Path, asset_name: str, *, step: int | None) -> None:
        if not path.exists():
            return
        try:
            html = path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to read HTML asset %s", path)
            return
        try:
            # Comet has no first-class "log_html_versioned" so we use log_asset
            # for the file (preserves history) and log_html for an inline
            # rendering. log_html only keeps the latest version per experiment,
            # which is fine for the current dashboard.
            self._experiment.log_asset(
                str(path),
                file_name=f"{asset_name.replace('/', '_')}.html",
                step=step,
            )
            self._experiment.log_html(html, clear=False)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to log HTML %s as %s", path, asset_name)


def _default_experiment_name(run_label: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{run_label}-{timestamp}"


def _get_attr(obj: Any, key: str, *, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
