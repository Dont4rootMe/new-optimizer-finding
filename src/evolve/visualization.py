"""Static population-level visualization snapshots for canonical evolution.

The legacy entrypoint :func:`render_evolution_overview` continues to render the
six-panel composite PNG into the population root. The new entrypoint
:func:`render_evolution_snapshot` additionally renders standalone PNGs per
panel (overview / timeline / survival groups) plus an interactive Plotly HTML
into ``population_root/viz/`` and returns a :class:`RenderedSnapshot` that the
evolution loop hands to telemetry sinks (e.g. CometRunLogger).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.evolve.storage import read_json, read_population_state

LOGGER = logging.getLogger(__name__)

OVERVIEW_FILENAME = "evolution_overview.png"
VIZ_SUBDIR = "viz"
VIZ_OVERVIEW_SUBDIR = "overview"
VIZ_TIMELINE_SUBDIR = "timeline"
VIZ_SURVIVAL_SUBDIR = "survival"
VIZ_INTERACTIVE_SUBDIR = "interactive"

_WITHIN_ISLAND_CROSSOVER = "within_island_crossover"
_INTER_ISLAND_CROSSOVER = "inter_island_crossover"
_MUTATION = "mutation"
_OPERATOR_PLOT_ORDER = (
    _WITHIN_ISLAND_CROSSOVER,
    _INTER_ISLAND_CROSSOVER,
    _MUTATION,
)
_OPERATOR_PLOT_LABELS = {
    _WITHIN_ISLAND_CROSSOVER: "Crossbreeding",
    _INTER_ISLAND_CROSSOVER: "Crossbreeding Across Islands",
    _MUTATION: "Mutation",
}
_OPERATOR_PLOT_COLORS = {
    _WITHIN_ISLAND_CROSSOVER: "#ff7f0e",
    _INTER_ISLAND_CROSSOVER: "#9467bd",
    _MUTATION: "#2ca02c",
}


@dataclass(frozen=True)
class OrganismVizRecord:
    organism_id: str
    island_id: str
    model_label: str
    generation_created: int
    current_generation_active: int
    operator: str
    status: str
    pipeline_state: str
    mother_id: str | None
    father_id: str | None
    simple_score: float | None
    hard_score: float | None
    created_at: datetime | None
    simple_eval_finished_at: datetime | None
    active: bool


@dataclass(frozen=True)
class RenderedSnapshot:
    """Bundle of paths produced by :func:`render_evolution_snapshot`.

    ``composite_overview_path`` is the legacy six-panel PNG written into the
    population root. ``overview_panels`` / ``timeline_panels`` /
    ``survival_panels`` map a stable panel id to its standalone PNG inside
    ``viz/<group>/``. ``interactive_html`` maps a stable id to the Plotly HTML
    inside ``viz/interactive/``. Telemetry sinks consume these maps to upload
    artifacts under deterministic names.
    """

    population_root: Path
    generation: int
    composite_overview_path: Path | None
    overview_panels: dict[str, Path] = field(default_factory=dict)
    timeline_panels: dict[str, Path] = field(default_factory=dict)
    survival_panels: dict[str, Path] = field(default_factory=dict)
    interactive_html: dict[str, Path] = field(default_factory=dict)


def render_evolution_overview(population_root: str | Path) -> Path | None:
    """Render the legacy six-panel composite PNG.

    Kept for callers that only need the dashboard image. New callers should
    prefer :func:`render_evolution_snapshot`.
    """

    snapshot = render_evolution_snapshot(population_root)
    return snapshot.composite_overview_path if snapshot is not None else None


def render_evolution_overview_sampled(
    population_root: str | Path,
    *,
    max_evaluated_points: int | None = None,
    output_filename: str = OVERVIEW_FILENAME,
) -> Path | None:
    """Compatibility shim: render only the composite PNG with optional sampling."""

    snapshot = render_evolution_snapshot(
        population_root,
        max_evaluated_points=max_evaluated_points,
        composite_filename=output_filename,
        render_extras=False,
    )
    return snapshot.composite_overview_path if snapshot is not None else None


def render_evolution_snapshot(
    population_root: str | Path,
    *,
    max_evaluated_points: int | None = None,
    composite_filename: str = OVERVIEW_FILENAME,
    render_extras: bool = True,
) -> RenderedSnapshot | None:
    """Render the composite dashboard plus per-panel and interactive artifacts.

    ``render_extras=False`` skips the per-panel PNGs and the Plotly HTML and is
    used by the legacy compatibility shims. The default rendering produces all
    artifacts inside ``population_root/viz/{overview,timeline,survival,interactive}``.
    """

    root = Path(population_root).expanduser().resolve()
    all_records, current_generation, _ = _load_records(root)
    if not all_records:
        return None

    records = _sample_records_for_render(
        all_records,
        max_evaluated_points=max_evaluated_points,
    )

    composite_path = _render_composite_overview(
        root,
        records,
        all_records=all_records,
        current_generation=current_generation,
        max_evaluated_points=max_evaluated_points,
        output_filename=composite_filename,
    )

    snapshot = RenderedSnapshot(
        population_root=root,
        generation=current_generation,
        composite_overview_path=composite_path,
    )

    if not render_extras:
        return snapshot

    overview_dir = root / VIZ_SUBDIR / VIZ_OVERVIEW_SUBDIR
    timeline_dir = root / VIZ_SUBDIR / VIZ_TIMELINE_SUBDIR
    survival_dir = root / VIZ_SUBDIR / VIZ_SURVIVAL_SUBDIR
    interactive_dir = root / VIZ_SUBDIR / VIZ_INTERACTIVE_SUBDIR
    for path in (overview_dir, timeline_dir, survival_dir, interactive_dir):
        path.mkdir(parents=True, exist_ok=True)

    overview_panels: dict[str, Path] = {}
    overview_panels["best_vs_evaluations"] = _save_panel(
        overview_dir / "best_vs_evaluations.png",
        lambda ax: _plot_best_vs_evaluations(ax, records),
    )
    overview_panels["evaluations_per_generation"] = _save_panel(
        overview_dir / "evaluations_per_generation.png",
        lambda ax: _plot_evaluations_per_generation(ax, records),
    )
    overview_panels["operators_total"] = _save_panel(
        overview_dir / "operators_total.png",
        lambda ax: _plot_operator_mix_by_generation(ax, records, context_records=all_records),
    )
    overview_panels["best_vs_runtime"] = _save_panel(
        overview_dir / "best_vs_runtime.png",
        lambda ax: _plot_best_vs_runtime(ax, records),
    )
    overview_panels["score_by_island"] = _save_panel(
        overview_dir / "score_by_island.png",
        lambda ax: _plot_score_by_island(ax, records),
    )
    overview_panels["score_by_model"] = _save_panel(
        overview_dir / "score_by_model.png",
        lambda ax: _plot_score_by_model(ax, records),
    )
    overview_panels["score_by_generation"] = _save_panel(
        overview_dir / "score_by_generation.png",
        lambda ax: _plot_score_by_generation(ax, records),
    )
    overview_panels["best_vs_evaluations_with_dead"] = _save_panel(
        overview_dir / "best_vs_evaluations_with_dead.png",
        lambda ax: _plot_best_vs_evaluations_with_dead(ax, records),
    )
    overview_panels["best_vs_runtime_with_dead"] = _save_panel(
        overview_dir / "best_vs_runtime_with_dead.png",
        lambda ax: _plot_best_vs_runtime_with_dead(ax, records),
    )

    survival_panels: dict[str, Path] = {}
    survival_panels["by_evaluations"] = _save_panel(
        survival_dir / "by_evaluations.png",
        lambda ax: _plot_survival_by_evaluations(ax, records),
    )
    survival_panels["by_runtime"] = _save_panel(
        survival_dir / "by_runtime.png",
        lambda ax: _plot_survival_by_runtime(ax, records),
    )
    survival_panels["by_generation"] = _save_panel(
        survival_dir / "by_generation.png",
        lambda ax: _plot_survival_by_generation(ax, records),
    )

    timeline_panels: dict[str, Path] = {}
    timeline_panels["cumulative_evaluations_by_island"] = _save_panel(
        timeline_dir / "cumulative_evaluations_by_island.png",
        lambda ax: _plot_cumulative_evaluations_by_island_over_generation(ax, records),
    )
    timeline_panels["cumulative_creations_by_operator"] = _save_panel(
        timeline_dir / "cumulative_creations_by_operator.png",
        lambda ax: _plot_cumulative_creations_by_operator_over_generation(
            ax, records, context_records=all_records
        ),
    )
    timeline_panels["cumulative_creations_by_island"] = _save_panel(
        timeline_dir / "cumulative_creations_by_island.png",
        lambda ax: _plot_cumulative_creations_by_island_over_generation(ax, records),
    )
    timeline_panels["cumulative_max_score_by_model"] = _save_panel(
        timeline_dir / "cumulative_max_score_by_model.png",
        lambda ax: _plot_cumulative_max_score_by_model_over_generation(ax, records),
    )

    # Every matplotlib panel also gets a Plotly HTML sibling. PNGs give
    # instant thumbnails in Comet; HTMLs let the operator hover for organism
    # metadata, zoom into a regime shift, and toggle traces. Each renderer
    # soft-skips when plotly isn't installed or when there's no data yet.
    interactive_html: dict[str, Path] = {}
    plotly_renderers = _build_plotly_renderers(records, all_records=all_records, interactive_dir=interactive_dir)
    for name, renderer in plotly_renderers.items():
        try:
            path = renderer()
        except Exception:  # noqa: BLE001
            LOGGER.exception("Plotly renderer %s failed", name)
            continue
        if path is not None:
            interactive_html[name] = path

    return RenderedSnapshot(
        population_root=root,
        generation=current_generation,
        composite_overview_path=composite_path,
        overview_panels=overview_panels,
        timeline_panels=timeline_panels,
        survival_panels=survival_panels,
        interactive_html=interactive_html,
    )


def _render_composite_overview(
    root: Path,
    records: list[OrganismVizRecord],
    *,
    all_records: list[OrganismVizRecord],
    current_generation: int,
    max_evaluated_points: int | None,
    output_filename: str,
) -> Path | None:
    active_count = sum(1 for record in records if record.active)
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.patch.set_facecolor("white")

    title = "Evolution Results"
    if max_evaluated_points is None:
        subtitle = (
            f"Generation {current_generation} | active organisms: {active_count} | "
            f"tracked organisms: {len(records)} | best active score: "
            f"{_format_score(_best_active_simple_score(records))}"
        )
    else:
        subtitle = (
            f"Generation {current_generation} | active organisms shown: {active_count} | "
            f"best active score: {_format_score(_best_active_simple_score(records))}"
        )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=20, fontweight="bold", y=0.98)

    _plot_best_vs_evaluations(axes[0, 0], records)
    _plot_evaluations_per_generation(axes[0, 1], records)
    _plot_operator_mix_by_generation(axes[0, 2], records, context_records=all_records)
    _plot_best_vs_runtime(axes[1, 0], records)
    _plot_score_by_island(axes[1, 1], records)
    _plot_score_by_model(axes[1, 2], records)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = root / output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_panel(path: Path, render_fn: Any) -> Path:
    """Render a single panel into its own PNG and return the path."""

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    try:
        render_fn(ax)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failed to render panel %s", path.name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _maybe_render_plotly_best_vs_evaluations(
    out_path: Path,
    records: list[OrganismVizRecord],
) -> Path | None:
    try:
        from src.evolve.visualization_plotly import render_best_vs_evaluations_plotly
    except ImportError as exc:  # pragma: no cover - optional dep
        LOGGER.warning("Plotly not installed; skipping interactive HTML (%s)", exc)
        return None
    try:
        return render_best_vs_evaluations_plotly(records, out_path=out_path)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Plotly render failed for %s", out_path)
        return None


def _build_plotly_renderers(
    records: list[OrganismVizRecord],
    *,
    all_records: list[OrganismVizRecord],
    interactive_dir: Path,
) -> dict[str, Any]:
    """Map panel-id → zero-arg callable that returns the rendered Path or None.

    Centralising the registry here keeps ``render_evolution_snapshot`` short
    and means a new plotly panel needs exactly two edits (one renderer + one
    line below). The matching matplotlib PNG names are deliberately kept
    one-to-one so the Comet UI can group them by id.
    """

    try:
        from src.evolve.visualization_plotly import (
            render_best_vs_evaluations_plotly,
            render_best_vs_runtime_plotly,
            render_cumulative_creations_by_island_plotly,
            render_cumulative_creations_by_operator_plotly,
            render_cumulative_evaluations_by_island_plotly,
            render_cumulative_max_score_by_model_plotly,
            render_evaluations_per_generation_plotly,
            render_operators_total_plotly,
            render_score_by_generation_plotly,
            render_score_by_island_plotly,
            render_score_by_model_plotly,
            render_survival_by_evaluations_plotly,
            render_survival_by_generation_plotly,
            render_survival_by_runtime_plotly,
        )
    except ImportError as exc:  # pragma: no cover - optional dep
        LOGGER.warning("Plotly not installed; skipping interactive HTML (%s)", exc)
        return {}

    def _bind(fn: Any, name: str, **kwargs: Any) -> Any:
        return lambda: fn(records, out_path=interactive_dir / name, **kwargs)

    return {
        # overview — same ids as overview_panels for clean Comet grouping
        "best_vs_evaluations": _bind(render_best_vs_evaluations_plotly, "best_vs_evaluations.html"),
        "best_vs_runtime": _bind(render_best_vs_runtime_plotly, "best_vs_runtime.html"),
        "score_by_island": _bind(render_score_by_island_plotly, "score_by_island.html"),
        "score_by_model": _bind(render_score_by_model_plotly, "score_by_model.html"),
        "score_by_generation": _bind(render_score_by_generation_plotly, "score_by_generation.html"),
        "operators_total": (
            lambda: render_operators_total_plotly(
                records, out_path=interactive_dir / "operators_total.html", context_records=all_records,
            )
        ),
        "evaluations_per_generation": _bind(
            render_evaluations_per_generation_plotly, "evaluations_per_generation.html"
        ),
        # timeline — cumulative-over-generation views
        "cumulative_evaluations_by_island": _bind(
            render_cumulative_evaluations_by_island_plotly,
            "cumulative_evaluations_by_island.html",
        ),
        "cumulative_creations_by_operator": (
            lambda: render_cumulative_creations_by_operator_plotly(
                records,
                out_path=interactive_dir / "cumulative_creations_by_operator.html",
                context_records=all_records,
            )
        ),
        "cumulative_creations_by_island": _bind(
            render_cumulative_creations_by_island_plotly, "cumulative_creations_by_island.html"
        ),
        "cumulative_max_score_by_model": _bind(
            render_cumulative_max_score_by_model_plotly, "cumulative_max_score_by_model.html"
        ),
        # survival
        "survival_by_evaluations": _bind(
            render_survival_by_evaluations_plotly, "survival_by_evaluations.html"
        ),
        "survival_by_runtime": _bind(render_survival_by_runtime_plotly, "survival_by_runtime.html"),
        "survival_by_generation": _bind(
            render_survival_by_generation_plotly, "survival_by_generation.html"
        ),
    }


def _load_records(population_root: Path) -> tuple[list[OrganismVizRecord], int, int]:
    state = read_population_state(population_root)
    active_ids: set[str] = set()
    current_generation = 0
    if isinstance(state, dict):
        current_generation = int(state.get("current_generation", 0))
        active_ids = {
            str(entry.get("organism_id"))
            for entry in state.get("active_organisms", [])
            if isinstance(entry, dict) and entry.get("organism_id")
        }

    records: list[OrganismVizRecord] = []
    meta_files = sorted(
        population_root.glob("gen_*/island_*/org_*/organism.json"),
        key=lambda path: str(path),
    )
    for meta_path in meta_files:
        try:
            meta = read_json(meta_path)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed reading organism meta for visualization: %s", meta_path)
            continue
        if not isinstance(meta, dict):
            continue

        org_dir = meta_path.parent
        summary = {}
        summary_path = org_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = read_json(summary_path)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed reading organism summary for visualization: %s", summary_path)
                summary = {}

        phase_results = summary.get("phase_results", {}) if isinstance(summary, dict) else {}
        simple_phase = phase_results.get("simple", {}) if isinstance(phase_results, dict) else {}

        organism_id = str(meta.get("organism_id", org_dir.name))
        records.append(
            OrganismVizRecord(
                organism_id=organism_id,
                island_id=str(meta.get("island_id", "unknown")),
                model_label=_model_label(meta),
                generation_created=_safe_int(meta.get("generation_created"), default=0),
                current_generation_active=_safe_int(meta.get("current_generation_active"), default=0),
                operator=str(meta.get("operator", "unknown")),
                status=str(meta.get("status", "unknown")),
                pipeline_state=str(meta.get("pipeline_state", "")),
                mother_id=_optional_str(meta.get("mother_id")),
                father_id=_optional_str(meta.get("father_id")),
                simple_score=_safe_float(meta.get("simple_score")),
                hard_score=_safe_float(meta.get("hard_score")),
                created_at=_parse_timestamp(meta.get("timestamp")),
                simple_eval_finished_at=_parse_timestamp(simple_phase.get("eval_finished_at")),
                active=organism_id in active_ids,
            )
        )

    return records, current_generation, len(active_ids)


# ---------------------------------------------------------------------------
# Overview panels (legacy six + new variants for tasks 4 and 5)
# ---------------------------------------------------------------------------


def _plot_best_vs_evaluations(ax: Any, records: list[OrganismVizRecord]) -> None:
    evaluated = _evaluated_records(records)
    if not evaluated:
        return _empty_panel(ax, "No evaluated organisms yet.")

    xs = list(range(1, len(evaluated) + 1))
    ys = [record.simple_score for record in evaluated]
    best_scores = _cumulative_best(ys)
    lineage = _maternal_lineage_points_by_evaluations(records, evaluated)

    ax.scatter(xs, ys, color="black", s=18, alpha=0.85, label="Individual Evals")
    ax.plot(xs, best_scores, color="#d62728", linewidth=2.0, label="Best Score")
    if lineage:
        ax.plot(
            [point[0] for point in lineage],
            [point[1] for point in lineage],
            color="#1f77b4",
            linestyle="--",
            linewidth=2.0,
            alpha=0.8,
            marker="o",
            markersize=5,
            label="Current Best Maternal Line",
        )
    ax.set_title("Best Score vs Evaluations", fontsize=16, fontweight="bold")
    ax.set_xlabel("# Evaluated Organisms", fontsize=12, fontweight="bold")
    ax.set_ylabel("Simple Score", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)


def _plot_evaluations_per_generation(ax: Any, records: list[OrganismVizRecord]) -> None:
    evaluated = _evaluated_records(records)
    if not evaluated:
        return _empty_panel(ax, "No evaluated organisms yet.")

    generation_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    islands = sorted({record.island_id for record in evaluated})
    for record in evaluated:
        generation_counts[record.generation_created][record.island_id] += 1

    generations = sorted(generation_counts)
    bottoms = [0] * len(generations)
    cmap = plt.get_cmap("tab10")
    for idx, island_id in enumerate(islands):
        heights = [generation_counts[generation].get(island_id, 0) for generation in generations]
        ax.bar(
            generations,
            heights,
            bottom=bottoms,
            label=island_id,
            color=cmap(idx % 10),
            alpha=0.9,
        )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights, strict=True)]

    ax.set_title("Evaluations per Generation", fontsize=16, fontweight="bold")
    ax.set_xlabel("Generation Created", fontsize=12, fontweight="bold")
    ax.set_ylabel("# Evaluated Organisms", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)


def _plot_operator_mix_by_generation(
    ax: Any,
    records: list[OrganismVizRecord],
    *,
    context_records: list[OrganismVizRecord] | None = None,
) -> None:
    operator_totals = _offspring_operator_totals(records, context_records=context_records)
    categories = [
        category
        for category in _OPERATOR_PLOT_ORDER
        if operator_totals.get(category, 0) > 0
    ]
    if not categories:
        return _empty_panel(ax, "No offspring records yet.")

    x_positions = list(range(len(categories)))
    bar_handles: list[Any] = []
    bar_labels: list[str] = []
    for x_position, category in zip(x_positions, categories, strict=True):
        container = ax.bar(
            [x_position],
            [operator_totals[category]],
            width=0.68,
            color=_OPERATOR_PLOT_COLORS[category],
            alpha=0.9,
            label=_OPERATOR_PLOT_LABELS[category],
        )
        bar_handles.append(container[0])
        bar_labels.append(_OPERATOR_PLOT_LABELS[category])

    ax.set_title("Created Organisms by Operator", fontsize=16, fontweight="bold")
    ax.set_xlabel("Operator", fontsize=12, fontweight="bold")
    ax.set_ylabel("# Created Organisms", fontsize=12, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_OPERATOR_PLOT_LABELS[category] for category in categories], rotation=12, ha="right")
    ax.grid(True, axis="y", alpha=0.25)

    if bar_handles:
        ax.legend(bar_handles, bar_labels, loc="upper left", fontsize=9)


def _plot_best_vs_runtime(ax: Any, records: list[OrganismVizRecord]) -> None:
    evaluated = [record for record in _evaluated_records(records) if record.simple_eval_finished_at is not None]
    if not evaluated:
        return _empty_panel(ax, "No evaluation timestamps yet.")

    evaluated.sort(key=lambda record: (record.simple_eval_finished_at, record.organism_id))
    t0 = evaluated[0].simple_eval_finished_at
    if t0 is None:
        return _empty_panel(ax, "No evaluation timestamps yet.")

    xs = [(record.simple_eval_finished_at - t0).total_seconds() for record in evaluated if record.simple_eval_finished_at]
    ys = [record.simple_score for record in evaluated]
    best_scores = _cumulative_best(ys)
    lineage = _maternal_lineage_points_by_runtime(records, evaluated, t0)

    ax.scatter(xs, ys, color="black", s=18, alpha=0.85, label="Individual Evals")
    ax.plot(xs, best_scores, color="#d62728", linewidth=2.0, label="Best Score")
    if lineage:
        ax.plot(
            [point[0] for point in lineage],
            [point[1] for point in lineage],
            color="#1f77b4",
            linestyle="--",
            linewidth=2.0,
            alpha=0.8,
            marker="o",
            markersize=5,
            label="Current Best Maternal Line",
        )
    ax.xaxis.set_major_formatter(FuncFormatter(_format_elapsed_seconds))
    ax.set_title("Best Score vs Runtime", fontsize=16, fontweight="bold")
    ax.set_xlabel("Elapsed Runtime", fontsize=12, fontweight="bold")
    ax.set_ylabel("Simple Score", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)


def _plot_score_by_island(ax: Any, records: list[OrganismVizRecord]) -> None:
    evaluated = _evaluated_records(records)
    if not evaluated:
        return _empty_panel(ax, "No evaluated organisms yet.")

    islands = sorted({record.island_id for record in evaluated})
    x_lookup = {island_id: idx for idx, island_id in enumerate(islands)}

    xs = [x_lookup[record.island_id] + _stable_jitter(record.organism_id) for record in evaluated]
    ys = [record.simple_score for record in evaluated]
    ax.scatter(xs, ys, color="black", s=20, alpha=0.75, label="Evaluated")

    active_records = [record for record in evaluated if record.active]
    if active_records:
        ax.scatter(
            [x_lookup[record.island_id] + _stable_jitter(record.organism_id) for record in active_records],
            [record.simple_score for record in active_records],
            facecolors="none",
            edgecolors="#2ca02c",
            linewidths=2.0,
            s=90,
            label="Active",
        )

    for island_id in islands:
        island_scores = [record.simple_score for record in evaluated if record.island_id == island_id]
        if not island_scores:
            continue
        x_center = x_lookup[island_id]
        ax.hlines(
            max(island_scores),
            x_center - 0.25,
            x_center + 0.25,
            colors="#d62728",
            linewidth=2.0,
        )

    ax.set_xticks(list(x_lookup.values()))
    ax.set_xticklabels(islands, rotation=12, ha="right")
    ax.set_title("Score by Island", fontsize=16, fontweight="bold")
    ax.set_xlabel("Island", fontsize=12, fontweight="bold")
    ax.set_ylabel("Simple Score", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=10)


def _plot_score_by_model(ax: Any, records: list[OrganismVizRecord]) -> None:
    evaluated = _evaluated_records(records)
    if not evaluated:
        return _empty_panel(ax, "No evaluated organisms yet.")

    models = sorted({record.model_label for record in evaluated})
    x_lookup = {model_label: idx for idx, model_label in enumerate(models)}

    xs = [x_lookup[record.model_label] + _stable_jitter(record.organism_id) for record in evaluated]
    ys = [record.simple_score for record in evaluated]
    ax.scatter(xs, ys, color="black", s=20, alpha=0.75, label="Evaluated")

    active_records = [record for record in evaluated if record.active]
    if active_records:
        ax.scatter(
            [x_lookup[record.model_label] + _stable_jitter(record.organism_id) for record in active_records],
            [record.simple_score for record in active_records],
            facecolors="none",
            edgecolors="#2ca02c",
            linewidths=2.0,
            s=90,
            label="Active",
        )

    for model_label in models:
        model_scores = [record.simple_score for record in evaluated if record.model_label == model_label]
        if not model_scores:
            continue
        x_center = x_lookup[model_label]
        ax.hlines(
            max(model_scores),
            x_center - 0.25,
            x_center + 0.25,
            colors="#d62728",
            linewidth=2.0,
        )

    ax.set_xticks(list(x_lookup.values()))
    ax.set_xticklabels(models, rotation=14, ha="right")
    ax.set_title("Score by Model", fontsize=16, fontweight="bold")
    ax.set_xlabel("LLM Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Simple Score", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=10)


def _plot_score_by_generation(ax: Any, records: list[OrganismVizRecord]) -> None:
    """Task 4: column-of-points scatter where Ox is the generation index.

    Multiple organisms in the same generation stack as a vertical column with a
    small horizontal jitter for separation.
    """

    evaluated = _evaluated_records(records)
    if not evaluated:
        return _empty_panel(ax, "No evaluated organisms yet.")

    xs = [record.generation_created + _stable_jitter(record.organism_id) for record in evaluated]
    ys = [record.simple_score for record in evaluated]
    ax.scatter(xs, ys, color="black", s=20, alpha=0.7, label="Evaluated")

    active_records = [record for record in evaluated if record.active]
    if active_records:
        ax.scatter(
            [record.generation_created + _stable_jitter(record.organism_id) for record in active_records],
            [record.simple_score for record in active_records],
            facecolors="none",
            edgecolors="#2ca02c",
            linewidths=2.0,
            s=90,
            label="Active",
        )

    generations = sorted({record.generation_created for record in evaluated})
    for generation in generations:
        per_gen_scores = [record.simple_score for record in evaluated if record.generation_created == generation]
        if not per_gen_scores:
            continue
        ax.hlines(
            max(per_gen_scores),
            generation - 0.32,
            generation + 0.32,
            colors="#d62728",
            linewidth=2.0,
        )

    ax.set_title("Score by Generation", fontsize=16, fontweight="bold")
    ax.set_xlabel("Generation Created", fontsize=12, fontweight="bold")
    ax.set_ylabel("Simple Score", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=10)


def _plot_best_vs_evaluations_with_dead(ax: Any, records: list[OrganismVizRecord]) -> None:
    """Task 5: like Best vs Evaluations but the Ox axis includes dead organisms.

    Each tracked organism (sorted by creation time) consumes one Ox column.
    If the organism never reached scoring its column is empty (no marker),
    leaving a visible gap. The cumulative-best line still tracks scored ones
    only.
    """

    ordered = sorted(
        records,
        key=lambda record: (
            record.created_at.timestamp() if record.created_at else float("inf"),
            record.generation_created,
            record.organism_id,
        ),
    )
    if not ordered:
        return _empty_panel(ax, "No tracked organisms yet.")

    xs_scored: list[int] = []
    ys_scored: list[float] = []
    best_running = -float("inf")
    best_xs: list[int] = []
    best_ys: list[float] = []
    for index, record in enumerate(ordered, start=1):
        if record.simple_score is None:
            continue
        xs_scored.append(index)
        ys_scored.append(record.simple_score)
        best_running = max(best_running, record.simple_score)
        best_xs.append(index)
        best_ys.append(best_running)

    if not xs_scored:
        return _empty_panel(ax, "No evaluated organisms yet.")

    ax.scatter(xs_scored, ys_scored, color="black", s=18, alpha=0.85, label="Scored Evals")
    ax.plot(best_xs, best_ys, color="#d62728", linewidth=2.0, label="Best Score (scored only)")
    ax.set_xlim(0.5, len(ordered) + 0.5)
    ax.set_title("Best Score vs Evaluations (incl. dead organisms)", fontsize=15, fontweight="bold")
    ax.set_xlabel("# Tracked Organisms (in creation order)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Simple Score", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)


def _plot_best_vs_runtime_with_dead(ax: Any, records: list[OrganismVizRecord]) -> None:
    """Task 5 runtime variant: dead organisms still consume the Ox axis.

    The Ox axis is the elapsed runtime since the first observed event
    (creation timestamp or simple-eval-finished timestamp, whichever exists).
    Dead organisms appear as faint gray ticks so the gap is visible without
    misreading them as scored points.
    """

    timed_records = [
        record
        for record in records
        if record.created_at is not None or record.simple_eval_finished_at is not None
    ]
    if not timed_records:
        return _empty_panel(ax, "No timestamped organisms yet.")

    def _record_time(record: OrganismVizRecord) -> datetime:
        return record.simple_eval_finished_at or record.created_at  # type: ignore[return-value]

    timed_records.sort(key=lambda record: (_record_time(record), record.organism_id))
    t0 = _record_time(timed_records[0])

    elapsed_xs: list[float] = []
    scored_xs: list[float] = []
    scored_ys: list[float] = []
    best_running = -float("inf")
    best_xs: list[float] = []
    best_ys: list[float] = []
    dead_xs: list[float] = []

    for record in timed_records:
        elapsed = (_record_time(record) - t0).total_seconds()
        elapsed_xs.append(elapsed)
        if record.simple_score is None:
            dead_xs.append(elapsed)
            continue
        scored_xs.append(elapsed)
        scored_ys.append(record.simple_score)
        best_running = max(best_running, record.simple_score)
        best_xs.append(elapsed)
        best_ys.append(best_running)

    if not scored_xs:
        return _empty_panel(ax, "No evaluated organisms yet.")

    if dead_xs:
        ax.scatter(
            dead_xs,
            [float("nan")] * len(dead_xs),
            marker="|",
            color="#999999",
            alpha=0.7,
            label="Dead Organisms",
        )
        # render dead ticks at the bottom of the panel via secondary y-fill
        ymin, ymax = min(scored_ys), max(scored_ys)
        margin = 0.05 * (ymax - ymin) if ymax != ymin else 1.0
        ax.scatter(
            dead_xs,
            [ymin - margin] * len(dead_xs),
            marker="|",
            color="#999999",
            alpha=0.7,
        )

    ax.scatter(scored_xs, scored_ys, color="black", s=18, alpha=0.85, label="Scored Evals")
    ax.plot(best_xs, best_ys, color="#d62728", linewidth=2.0, label="Best Score (scored only)")

    ax.xaxis.set_major_formatter(FuncFormatter(_format_elapsed_seconds))
    ax.set_title("Best Score vs Runtime (incl. dead organisms)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Elapsed Runtime (since first event)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Simple Score", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)


# ---------------------------------------------------------------------------
# Survival panels (task 6)
# ---------------------------------------------------------------------------


def _plot_survival_by_evaluations(ax: Any, records: list[OrganismVizRecord]) -> None:
    """Cumulative scored / total / ratio over creation index."""

    ordered = sorted(
        records,
        key=lambda record: (
            record.created_at.timestamp() if record.created_at else float("inf"),
            record.generation_created,
            record.organism_id,
        ),
    )
    if not ordered:
        return _empty_panel(ax, "No tracked organisms yet.")

    xs = list(range(1, len(ordered) + 1))
    cumulative_scored = 0
    cumulative_dead = 0
    scored_series: list[int] = []
    dead_series: list[int] = []
    ratio_series: list[float] = []
    for record in ordered:
        if record.simple_score is None:
            cumulative_dead += 1
        else:
            cumulative_scored += 1
        scored_series.append(cumulative_scored)
        dead_series.append(cumulative_dead)
        total = cumulative_scored + cumulative_dead
        ratio_series.append(cumulative_scored / total if total else 0.0)

    ax.plot(xs, scored_series, color="#2ca02c", linewidth=2.0, label="Scored (cumulative)")
    ax.plot(xs, dead_series, color="#d62728", linewidth=2.0, label="Dead (cumulative)")
    ax.set_ylabel("# Organisms (cumulative)", fontsize=12, fontweight="bold")

    ax_ratio = ax.twinx()
    ax_ratio.plot(xs, ratio_series, color="#1f77b4", linestyle="--", linewidth=1.8, label="Survival Ratio")
    ax_ratio.set_ylim(0.0, 1.0)
    ax_ratio.set_ylabel("scored / (scored + dead)", fontsize=11, color="#1f77b4")
    ax_ratio.tick_params(axis="y", labelcolor="#1f77b4")

    ax.set_title("Survival by Creation Index", fontsize=15, fontweight="bold")
    ax.set_xlabel("# Tracked Organisms (in creation order)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax_ratio.legend(loc="lower right", fontsize=9)


def _plot_survival_by_runtime(ax: Any, records: list[OrganismVizRecord]) -> None:
    timed_records = [
        record
        for record in records
        if record.created_at is not None or record.simple_eval_finished_at is not None
    ]
    if not timed_records:
        return _empty_panel(ax, "No timestamped organisms yet.")

    def _record_time(record: OrganismVizRecord) -> datetime:
        return record.simple_eval_finished_at or record.created_at  # type: ignore[return-value]

    timed_records.sort(key=lambda record: (_record_time(record), record.organism_id))
    t0 = _record_time(timed_records[0])

    xs: list[float] = []
    scored_series: list[int] = []
    dead_series: list[int] = []
    ratio_series: list[float] = []
    cumulative_scored = 0
    cumulative_dead = 0
    for record in timed_records:
        elapsed = (_record_time(record) - t0).total_seconds()
        if record.simple_score is None:
            cumulative_dead += 1
        else:
            cumulative_scored += 1
        xs.append(elapsed)
        scored_series.append(cumulative_scored)
        dead_series.append(cumulative_dead)
        total = cumulative_scored + cumulative_dead
        ratio_series.append(cumulative_scored / total if total else 0.0)

    ax.plot(xs, scored_series, color="#2ca02c", linewidth=2.0, label="Scored (cumulative)")
    ax.plot(xs, dead_series, color="#d62728", linewidth=2.0, label="Dead (cumulative)")
    ax.set_ylabel("# Organisms (cumulative)", fontsize=12, fontweight="bold")

    ax_ratio = ax.twinx()
    ax_ratio.plot(xs, ratio_series, color="#1f77b4", linestyle="--", linewidth=1.8, label="Survival Ratio")
    ax_ratio.set_ylim(0.0, 1.0)
    ax_ratio.set_ylabel("scored / (scored + dead)", fontsize=11, color="#1f77b4")
    ax_ratio.tick_params(axis="y", labelcolor="#1f77b4")

    ax.xaxis.set_major_formatter(FuncFormatter(_format_elapsed_seconds))
    ax.set_title("Survival by Runtime", fontsize=15, fontweight="bold")
    ax.set_xlabel("Elapsed Runtime", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax_ratio.legend(loc="lower right", fontsize=9)


def _plot_survival_by_generation(ax: Any, records: list[OrganismVizRecord]) -> None:
    if not records:
        return _empty_panel(ax, "No tracked organisms yet.")

    per_generation: dict[int, dict[str, int]] = defaultdict(lambda: {"scored": 0, "dead": 0})
    for record in records:
        bucket = per_generation[record.generation_created]
        if record.simple_score is None:
            bucket["dead"] += 1
        else:
            bucket["scored"] += 1

    generations = sorted(per_generation)
    cumulative_scored = 0
    cumulative_dead = 0
    scored_series: list[int] = []
    dead_series: list[int] = []
    ratio_series: list[float] = []
    for generation in generations:
        cumulative_scored += per_generation[generation]["scored"]
        cumulative_dead += per_generation[generation]["dead"]
        scored_series.append(cumulative_scored)
        dead_series.append(cumulative_dead)
        total = cumulative_scored + cumulative_dead
        ratio_series.append(cumulative_scored / total if total else 0.0)

    ax.plot(generations, scored_series, color="#2ca02c", linewidth=2.0, label="Scored (cumulative)")
    ax.plot(generations, dead_series, color="#d62728", linewidth=2.0, label="Dead (cumulative)")
    ax.set_ylabel("# Organisms (cumulative)", fontsize=12, fontweight="bold")

    ax_ratio = ax.twinx()
    ax_ratio.plot(generations, ratio_series, color="#1f77b4", linestyle="--", linewidth=1.8, label="Survival Ratio")
    ax_ratio.set_ylim(0.0, 1.0)
    ax_ratio.set_ylabel("scored / (scored + dead)", fontsize=11, color="#1f77b4")
    ax_ratio.tick_params(axis="y", labelcolor="#1f77b4")

    ax.set_title("Survival by Generation", fontsize=15, fontweight="bold")
    ax.set_xlabel("Generation Created", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax_ratio.legend(loc="lower right", fontsize=9)


# ---------------------------------------------------------------------------
# Cumulative timeline panels (task 7)
# ---------------------------------------------------------------------------


def _plot_cumulative_evaluations_by_island_over_generation(
    ax: Any,
    records: list[OrganismVizRecord],
) -> None:
    evaluated = _evaluated_records(records)
    if not evaluated:
        return _empty_panel(ax, "No evaluated organisms yet.")

    islands = sorted({record.island_id for record in evaluated})
    generations = sorted({record.generation_created for record in evaluated})
    cumulative: dict[str, list[int]] = {island_id: [] for island_id in islands}
    running: dict[str, int] = {island_id: 0 for island_id in islands}
    per_gen_per_island: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in evaluated:
        per_gen_per_island[record.generation_created][record.island_id] += 1

    for generation in generations:
        for island_id in islands:
            running[island_id] += per_gen_per_island[generation].get(island_id, 0)
            cumulative[island_id].append(running[island_id])

    cmap = plt.get_cmap("tab10")
    bottoms = [0.0] * len(generations)
    for idx, island_id in enumerate(islands):
        heights = cumulative[island_id]
        ax.fill_between(
            generations,
            bottoms,
            [bottom + height for bottom, height in zip(bottoms, heights, strict=True)],
            color=cmap(idx % 10),
            alpha=0.7,
            label=island_id,
        )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights, strict=True)]

    ax.set_title("Cumulative Evaluations by Island (over generations)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Generation Created", fontsize=12, fontweight="bold")
    ax.set_ylabel("# Evaluated Organisms (cumulative)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)


def _plot_cumulative_creations_by_operator_over_generation(
    ax: Any,
    records: list[OrganismVizRecord],
    *,
    context_records: list[OrganismVizRecord] | None = None,
) -> None:
    counts_by_generation = _offspring_operator_counts_by_generation(
        records,
        context_records=context_records,
    )
    if not any(per_gen for per_gen in counts_by_generation.values()):
        return _empty_panel(ax, "No offspring records yet.")

    all_generations = sorted(
        {generation for per_gen in counts_by_generation.values() for generation in per_gen.keys()}
    )

    cumulative: dict[str, list[int]] = {category: [] for category in _OPERATOR_PLOT_ORDER}
    for category in _OPERATOR_PLOT_ORDER:
        running = 0
        per_generation = counts_by_generation.get(category, {})
        for generation in all_generations:
            running += per_generation.get(generation, 0)
            cumulative[category].append(running)

    for category in _OPERATOR_PLOT_ORDER:
        if not cumulative[category] or cumulative[category][-1] == 0:
            continue
        ax.plot(
            all_generations,
            cumulative[category],
            color=_OPERATOR_PLOT_COLORS[category],
            linewidth=2.0,
            label=_OPERATOR_PLOT_LABELS[category],
        )

    ax.set_title("Cumulative Creations by Operator (over generations)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Generation Created", fontsize=12, fontweight="bold")
    ax.set_ylabel("# Created Organisms (cumulative)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)


def _plot_cumulative_creations_by_island_over_generation(
    ax: Any,
    records: list[OrganismVizRecord],
) -> None:
    if not records:
        return _empty_panel(ax, "No tracked organisms yet.")

    islands = sorted({record.island_id for record in records})
    generations = sorted({record.generation_created for record in records})

    per_gen_per_island: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        per_gen_per_island[record.generation_created][record.island_id] += 1

    cumulative: dict[str, list[int]] = {island_id: [] for island_id in islands}
    running: dict[str, int] = {island_id: 0 for island_id in islands}
    for generation in generations:
        for island_id in islands:
            running[island_id] += per_gen_per_island[generation].get(island_id, 0)
            cumulative[island_id].append(running[island_id])

    cmap = plt.get_cmap("tab10")
    for idx, island_id in enumerate(islands):
        ax.plot(
            generations,
            cumulative[island_id],
            color=cmap(idx % 10),
            linewidth=2.0,
            label=island_id,
        )

    ax.set_title("Cumulative Creations by Island (over generations)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Generation Created", fontsize=12, fontweight="bold")
    ax.set_ylabel("# Created Organisms (cumulative)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)


def _plot_cumulative_max_score_by_model_over_generation(
    ax: Any,
    records: list[OrganismVizRecord],
) -> None:
    evaluated = _evaluated_records(records)
    if not evaluated:
        return _empty_panel(ax, "No evaluated organisms yet.")

    models = sorted({record.model_label for record in evaluated})
    generations = sorted({record.generation_created for record in evaluated})

    per_gen_per_model_max: dict[str, dict[int, float]] = {model_label: {} for model_label in models}
    for record in evaluated:
        bucket = per_gen_per_model_max[record.model_label]
        previous = bucket.get(record.generation_created)
        if previous is None or record.simple_score > previous:
            bucket[record.generation_created] = float(record.simple_score)

    cumulative_max: dict[str, list[float | None]] = {model_label: [] for model_label in models}
    for model_label in models:
        running_max: float | None = None
        for generation in generations:
            sample = per_gen_per_model_max[model_label].get(generation)
            if sample is not None and (running_max is None or sample > running_max):
                running_max = sample
            cumulative_max[model_label].append(running_max)

    cmap = plt.get_cmap("tab10")
    for idx, model_label in enumerate(models):
        ys = cumulative_max[model_label]
        if not any(value is not None for value in ys):
            continue
        ax.plot(
            generations,
            [value if value is not None else float("nan") for value in ys],
            color=cmap(idx % 10),
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=model_label,
        )

    ax.set_title("Cumulative Max Score by Model (over generations)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Generation Created", fontsize=12, fontweight="bold")
    ax.set_ylabel("Best Simple Score so far", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)


# ---------------------------------------------------------------------------
# Helper accessors
# ---------------------------------------------------------------------------


def _evaluated_records(records: list[OrganismVizRecord]) -> list[OrganismVizRecord]:
    evaluated = [record for record in records if record.simple_score is not None]
    evaluated.sort(key=lambda record: (_time_sort_key(record), record.generation_created, record.organism_id))
    return evaluated


def _best_active_simple_score(records: list[OrganismVizRecord]) -> float | None:
    active_scores = [
        record.simple_score
        for record in records
        if record.active and record.simple_score is not None
    ]
    if active_scores:
        return max(active_scores)
    all_scores = [record.simple_score for record in records if record.simple_score is not None]
    if not all_scores:
        return None
    return max(all_scores)


def _current_best_record(records: list[OrganismVizRecord]) -> OrganismVizRecord | None:
    active_scored = [
        record
        for record in records
        if record.active and record.simple_score is not None
    ]
    pool = active_scored or [record for record in records if record.simple_score is not None]
    if not pool:
        return None
    return max(
        pool,
        key=lambda record: (
            record.simple_score,
            record.current_generation_active,
            record.generation_created,
            _time_sort_key(record),
        ),
    )


def _maternal_lineage(records: list[OrganismVizRecord]) -> list[OrganismVizRecord]:
    current_best = _current_best_record(records)
    if current_best is None:
        return []

    records_by_id = {record.organism_id: record for record in records}
    lineage: list[OrganismVizRecord] = []
    seen: set[str] = set()
    cursor: OrganismVizRecord | None = current_best
    while cursor is not None and cursor.organism_id not in seen:
        lineage.append(cursor)
        seen.add(cursor.organism_id)
        if cursor.mother_id is None:
            break
        cursor = records_by_id.get(cursor.mother_id)
    lineage.reverse()
    return lineage


def _maternal_lineage_points_by_evaluations(
    records: list[OrganismVizRecord],
    evaluated: list[OrganismVizRecord],
) -> list[tuple[int, float]]:
    eval_index_by_id = {
        record.organism_id: idx + 1
        for idx, record in enumerate(evaluated)
    }
    return [
        (eval_index_by_id[record.organism_id], float(record.simple_score))
        for record in _maternal_lineage(records)
        if record.simple_score is not None and record.organism_id in eval_index_by_id
    ]


def _maternal_lineage_points_by_runtime(
    records: list[OrganismVizRecord],
    evaluated: list[OrganismVizRecord],
    t0: datetime,
) -> list[tuple[float, float]]:
    elapsed_by_id = {
        record.organism_id: float((record.simple_eval_finished_at - t0).total_seconds())
        for record in evaluated
        if record.simple_eval_finished_at is not None
    }
    return [
        (elapsed_by_id[record.organism_id], float(record.simple_score))
        for record in _maternal_lineage(records)
        if record.simple_score is not None and record.organism_id in elapsed_by_id
    ]


def _offspring_operator_counts_by_generation(
    records: list[OrganismVizRecord],
    *,
    context_records: list[OrganismVizRecord] | None = None,
) -> dict[str, dict[int, int]]:
    counts = {category: defaultdict(int) for category in _OPERATOR_PLOT_ORDER}
    lookup_source = context_records or records
    records_by_id = {record.organism_id: record for record in lookup_source}

    for record in records:
        category = _offspring_operator_category(record, records_by_id)
        if category is None:
            continue
        counts[category][record.generation_created] += 1

    return {category: dict(per_generation) for category, per_generation in counts.items()}


def _offspring_operator_totals(
    records: list[OrganismVizRecord],
    *,
    context_records: list[OrganismVizRecord] | None = None,
) -> dict[str, int]:
    counts_by_generation = _offspring_operator_counts_by_generation(
        records,
        context_records=context_records,
    )
    return {
        category: sum(per_generation.values())
        for category, per_generation in counts_by_generation.items()
    }


def _offspring_operator_category(
    record: OrganismVizRecord,
    records_by_id: dict[str, OrganismVizRecord],
) -> str | None:
    if record.operator == _MUTATION:
        return _MUTATION
    if record.operator != "crossover":
        return None
    if record.father_id:
        father = records_by_id.get(record.father_id)
        if father is not None and father.island_id != record.island_id:
            return _INTER_ISLAND_CROSSOVER
        return _WITHIN_ISLAND_CROSSOVER
    return _WITHIN_ISLAND_CROSSOVER


def build_ancestor_chains(records: list[OrganismVizRecord]) -> dict[str, list[str]]:
    """Return a maternal-ancestor chain (root → self) for every record.

    Used by the interactive Plotly graph to power the per-point hover trace.
    The chain is ordered oldest-first and includes the organism itself at the
    tail. Cycles (corrupted lineage data) are guarded against with a seen-set.
    """

    records_by_id = {record.organism_id: record for record in records}
    chains: dict[str, list[str]] = {}
    for record in records:
        chain: list[str] = []
        cursor: OrganismVizRecord | None = record
        seen: set[str] = set()
        while cursor is not None and cursor.organism_id not in seen:
            chain.append(cursor.organism_id)
            seen.add(cursor.organism_id)
            if cursor.mother_id is None:
                break
            cursor = records_by_id.get(cursor.mother_id)
        chain.reverse()
        chains[record.organism_id] = chain
    return chains


def _cumulative_best(values: list[float]) -> list[float]:
    best_values: list[float] = []
    running_best = -float("inf")
    for value in values:
        running_best = max(running_best, value)
        best_values.append(running_best)
    return best_values


def _sample_records_for_render(
    records: list[OrganismVizRecord],
    *,
    max_evaluated_points: int | None,
) -> list[OrganismVizRecord]:
    if max_evaluated_points is None:
        return list(records)

    evaluated = _evaluated_records(records)
    if len(evaluated) <= max_evaluated_points:
        return evaluated

    priority_ids: list[str] = []
    _extend_unique_ids(priority_ids, [evaluated[0].organism_id, evaluated[-1].organism_id])
    _extend_unique_ids(
        priority_ids,
        [
            record.organism_id
            for record in _maternal_lineage(records)
            if record.simple_score is not None
        ],
    )
    _extend_unique_ids(
        priority_ids,
        [record.organism_id for record in evaluated if record.active],
    )

    selected_ids = priority_ids[:max_evaluated_points]
    selected_lookup = set(selected_ids)
    remaining_slots = max_evaluated_points - len(selected_ids)
    if remaining_slots > 0:
        remaining_records = [
            record for record in evaluated if record.organism_id not in selected_lookup
        ]
        sampled_records = _evenly_sample_records(remaining_records, remaining_slots)
        _extend_unique_ids(selected_ids, [record.organism_id for record in sampled_records])
        selected_lookup = set(selected_ids)

    return [
        record
        for record in evaluated
        if record.organism_id in selected_lookup
    ]


def _extend_unique_ids(target: list[str], organism_ids: list[str]) -> None:
    seen = set(target)
    for organism_id in organism_ids:
        if organism_id in seen:
            continue
        target.append(organism_id)
        seen.add(organism_id)


def _evenly_sample_records(
    records: list[OrganismVizRecord],
    sample_size: int,
) -> list[OrganismVizRecord]:
    if sample_size <= 0 or not records:
        return []
    if sample_size >= len(records):
        return list(records)

    indices = [
        ((2 * slot + 1) * len(records)) // (2 * sample_size)
        for slot in range(sample_size)
    ]
    return [records[index] for index in indices]


def _stable_jitter(identifier: str) -> float:
    digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()
    normalized = int(digest[:8], 16) / 0xFFFFFFFF
    return (normalized - 0.5) * 0.42


def _time_sort_key(record: OrganismVizRecord) -> tuple[float, str]:
    reference = record.simple_eval_finished_at or record.created_at
    if reference is None:
        return (float("inf"), record.organism_id)
    return (reference.timestamp(), record.organism_id)


def _format_elapsed_seconds(seconds: float, _: int) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _format_score(score: float | None) -> str:
    if score is None:
        return "n/a"
    return f"{score:.4f}"


def _empty_panel(ax: Any, message: str) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#666666", wrap=True)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _model_label(meta: dict[str, Any]) -> str:
    for key in ("provider_model_id", "model_name", "llm_route_id", "llm_provider"):
        value = _optional_str(meta.get(key))
        if value:
            return value
    return "(unknown model)"
