"""Comet ML telemetry wrapper for the evolution loop.

Configured via ``cfg.comet`` in the project's Hydra config. Design:

- *Opt-in*: nothing happens unless ``cfg.comet.enabled`` is true.
- *Hard-fail when enabled*: when ``cfg.comet.enabled`` is true any
  initialization failure (missing ``comet_ml``, bad API key, network
  unreachable, mis-matched workspace) raises ``CometInitError`` so the
  whole run dies immediately. Silent disable would let a long remote
  run produce zero telemetry without anyone noticing.
- *One Comet experiment per population*: the Comet ``experiment_key`` is
  persisted to ``population_root/comet_experiment.json`` after the first
  successful init. Subsequent processes (e.g. ``seed_population.sh``
  followed by ``run_evolution.sh``) detect that file and resume the same
  experiment via ``ExistingExperiment``, so seed + evolve land in one
  Comet run instead of two disconnected ones.
- *Idempotent uploads*: per-snapshot assets use a stable group-prefixed
  name so the Comet UI groups them automatically.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


_DEFAULT_PROJECT_NAME = "evolution-runs"
_EXPERIMENT_KEY_FILENAME = "comet_experiment.json"


class CometInitError(RuntimeError):
    """Raised when ``cfg.comet.enabled`` is true but Comet cannot start.

    The message includes a hint about the most likely fix (install the
    package, set the env var, check workspace name) so the operator on a
    headless remote box can act without spelunking through the codebase.
    """


def _announce(message: str) -> None:
    """Mirror evolution_loop._announce — log + flushed stderr line.

    Comet init/status messages MUST surface even when Hydra reconfigures root
    logging at import time, so we always also write to stderr directly.
    """

    LOGGER.info(message)
    try:
        print(f"[comet] {message}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass


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
        population_root: Path | None = None,
    ) -> None:
        self._enabled = enabled
        self._log_combined_overview = log_combined_overview
        self._log_individual_panels = log_individual_panels
        self._log_plotly = log_plotly
        self._log_step_per_generation = log_step_per_generation
        self._experiment: Any | None = None
        self._first_snapshot_logged = False
        self._population_root = Path(population_root) if population_root is not None else None
        self._experiment_key: str | None = None

        if not enabled:
            _announce("logger disabled in config (comet.enabled=false); telemetry will not upload")
            return

        try:
            import comet_ml  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise CometInitError(
                "comet.enabled=true but the comet_ml package is NOT installed in this Python env. "
                "Install it on the remote box: `pip install comet_ml>=3.40.0` "
                f"(or the full evolve extra: `pip install -e '.[evolve]'`). "
                f"Original import error: {type(exc).__name__}: {exc}"
            ) from exc

        if not api_key:
            raise CometInitError(
                "comet.enabled=true but no api_key was resolved. Set comet.api_key in the YAML "
                "or export COMET_API_KEY in the run environment."
            )

        resolved_project = project_name or _DEFAULT_PROJECT_NAME
        cached = self._read_cached_experiment()
        try:
            if (
                cached is not None
                and cached.get("project_name") == resolved_project
                and (cached.get("workspace") or None) == (workspace or None)
                and isinstance(cached.get("experiment_key"), str)
                and cached["experiment_key"]
            ):
                # Resume: a prior process (typically seed_population.sh)
                # already created the experiment under this population_root.
                self._experiment = comet_ml.ExistingExperiment(
                    api_key=api_key,
                    previous_experiment=cached["experiment_key"],
                    log_code=False,
                    log_graph=False,
                    auto_param_logging=False,
                    auto_metric_logging=False,
                    auto_output_logging="simple",
                )
                resumed = True
            else:
                self._experiment = comet_ml.Experiment(
                    api_key=api_key,
                    project_name=resolved_project,
                    workspace=workspace,
                    log_code=False,
                    log_graph=False,
                    auto_param_logging=False,
                    auto_metric_logging=False,
                    auto_output_logging="simple",
                )
                resumed = False
        except Exception as exc:  # noqa: BLE001
            raise CometInitError(
                "comet.enabled=true but the experiment failed to start "
                f"(project={resolved_project!r}, workspace={workspace!r}). "
                "Check the api_key, workspace name, and network reachability. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        resolved_name = experiment_name or _default_experiment_name(run_label)
        if not resumed:
            try:
                self._experiment.set_name(resolved_name)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to set Comet experiment name")

        try:
            self._experiment_key = self._experiment.get_key()
        except Exception:  # noqa: BLE001
            try:
                self._experiment_key = getattr(self._experiment, "id", None)
            except Exception:  # noqa: BLE001
                self._experiment_key = None

        experiment_url = None
        for attr in ("url", "_get_experiment_url"):
            try:
                value = getattr(self._experiment, attr)
                experiment_url = value() if callable(value) else value
                if experiment_url:
                    break
            except Exception:  # noqa: BLE001
                continue
        url_suffix = f" url={experiment_url}" if experiment_url else ""
        action = "resumed existing" if resumed else "initialized"
        _announce(
            f"{action} experiment — project={resolved_project} "
            f"workspace={workspace or '(default)'} name={resolved_name} "
            f"key={self._experiment_key}{url_suffix}"
        )

        if not resumed:
            self._write_cached_experiment(
                {
                    "experiment_key": self._experiment_key,
                    "project_name": resolved_project,
                    "workspace": workspace,
                    "experiment_name": resolved_name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    def _experiment_key_path(self) -> Path | None:
        if self._population_root is None:
            return None
        return self._population_root / _EXPERIMENT_KEY_FILENAME

    def _read_cached_experiment(self) -> dict[str, Any] | None:
        path = self._experiment_key_path()
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            LOGGER.warning("comet_experiment.json at %s is unreadable; ignoring cache", path)
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _write_cached_experiment(self, payload: dict[str, Any]) -> None:
        path = self._experiment_key_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to persist comet experiment key to %s", path)

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self._experiment is not None)

    @classmethod
    def from_cfg(
        cls,
        cfg: Any,
        *,
        run_label: str = "evolution",
        population_root: Path | None = None,
    ) -> "CometRunLogger":
        """Build a logger from the ``cfg.comet`` block of a Hydra config.

        Returns a no-op logger if the block is missing or ``enabled`` is
        false. Raises :class:`CometInitError` if ``enabled`` is true but
        the experiment cannot be initialized — this is intentional: silent
        disable on a remote run produces zero telemetry without anyone
        noticing until the run is over.
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
                population_root=population_root,
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
            population_root=population_root,
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

        uploaded_images = 0
        uploaded_html = 0
        try:
            composite = getattr(snapshot, "composite_overview_path", None)
            if self._log_combined_overview and composite is not None:
                if self._upload_image(Path(composite), "overview/composite", step=step):
                    uploaded_images += 1

            if self._log_individual_panels:
                for name, path in dict(getattr(snapshot, "overview_panels", {}) or {}).items():
                    if self._upload_image(Path(path), f"overview/{name}", step=step):
                        uploaded_images += 1
                for name, path in dict(getattr(snapshot, "timeline_panels", {}) or {}).items():
                    if self._upload_image(Path(path), f"timeline/{name}", step=step):
                        uploaded_images += 1
                for name, path in dict(getattr(snapshot, "survival_panels", {}) or {}).items():
                    if self._upload_image(Path(path), f"survival/{name}", step=step):
                        uploaded_images += 1

            if self._log_plotly:
                for name, path in dict(getattr(snapshot, "interactive_html", {}) or {}).items():
                    if self._upload_html(Path(path), f"interactive/{name}", step=step):
                        uploaded_html += 1
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to log snapshot to Comet (continuing)")

        if not self._first_snapshot_logged:
            self._first_snapshot_logged = True
            _announce(
                f"first snapshot uploaded (step={step}, images={uploaded_images}, html={uploaded_html}); "
                "telemetry confirmed working"
            )
        elif uploaded_images == 0 and uploaded_html == 0:
            _announce(f"snapshot at step={step} produced zero uploads — check artifact paths")

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

    def _upload_image(self, path: Path, asset_name: str, *, step: int | None) -> bool:
        if not path.exists():
            return False
        try:
            self._experiment.log_image(str(path), name=asset_name, step=step)
            return True
        except Exception as exc:  # noqa: BLE001
            if not self._first_snapshot_logged:
                # First failure: surface loudly so the user knows uploads are broken.
                _announce(
                    f"image upload failed asset={asset_name} path={path.name} "
                    f"{type(exc).__name__}: {exc}"
                )
            LOGGER.exception("Failed to log image %s as %s", path, asset_name)
            return False

    def _upload_html(self, path: Path, asset_name: str, *, step: int | None) -> bool:
        if not path.exists():
            return False
        try:
            html = path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to read HTML asset %s", path)
            return False
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
            return True
        except Exception as exc:  # noqa: BLE001
            if not self._first_snapshot_logged:
                _announce(
                    f"html upload failed asset={asset_name} path={path.name} "
                    f"{type(exc).__name__}: {exc}"
                )
            LOGGER.exception("Failed to log HTML %s as %s", path, asset_name)
            return False


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
