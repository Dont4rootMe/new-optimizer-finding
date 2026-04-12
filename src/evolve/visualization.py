"""Static population-level visualization snapshots for canonical evolution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from src.evolve.storage import read_json, read_population_state

LOGGER = logging.getLogger(__name__)

OVERVIEW_FILENAME = "evolution_overview.png"

_OPERATOR_MARKERS = {
    "seed": "^",
    "mutation": "o",
    "crossover": "s",
}


@dataclass(frozen=True)
class OrganismVizRecord:
    organism_id: str
    island_id: str
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


def render_evolution_overview(population_root: str | Path) -> Path | None:
    """Render the latest static overview image into `population_root`."""

    root = Path(population_root).expanduser().resolve()
    records, current_generation, active_count = _load_records(root)
    if not records:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.patch.set_facecolor("white")

    title = "Evolution Results"
    subtitle = (
        f"Generation {current_generation} | active organisms: {active_count} | "
        f"tracked organisms: {len(records)}"
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=20, fontweight="bold", y=0.98)

    _plot_best_vs_evaluations(axes[0, 0], records)
    _plot_evaluations_per_generation(axes[0, 1], records)
    _plot_best_vs_generation(axes[0, 2], records)
    _plot_best_vs_runtime(axes[1, 0], records)
    _plot_score_by_island(axes[1, 1], records)
    _plot_lineage_graph(fig, axes[1, 2], records)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = root / OVERVIEW_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


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


def _plot_best_vs_evaluations(ax: Any, records: list[OrganismVizRecord]) -> None:
    evaluated = _evaluated_records(records)
    if not evaluated:
        return _empty_panel(ax, "No evaluated organisms yet.")

    xs = list(range(1, len(evaluated) + 1))
    ys = [record.simple_score for record in evaluated]
    best_scores = _cumulative_best(ys)

    ax.scatter(xs, ys, color="black", s=18, alpha=0.85, label="Individual Evals")
    ax.plot(xs, best_scores, color="#d62728", linewidth=2.0, label="Best Score")
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


def _plot_best_vs_generation(ax: Any, records: list[OrganismVizRecord]) -> None:
    evaluated = _evaluated_records(records)
    if not evaluated:
        return _empty_panel(ax, "No evaluated organisms yet.")

    evaluated.sort(key=lambda record: (record.generation_created, _time_sort_key(record), record.organism_id))
    xs = [record.generation_created for record in evaluated]
    ys = [record.simple_score for record in evaluated]

    generation_best: list[tuple[int, float]] = []
    running_best = -float("inf")
    for generation in sorted({record.generation_created for record in evaluated}):
        generation_scores = [record.simple_score for record in evaluated if record.generation_created == generation]
        if not generation_scores:
            continue
        running_best = max(running_best, max(generation_scores))
        generation_best.append((generation, running_best))

    ax.scatter(xs, ys, color="black", s=18, alpha=0.85, label="Individual Evals")
    if generation_best:
        ax.step(
            [entry[0] for entry in generation_best],
            [entry[1] for entry in generation_best],
            where="post",
            color="#d62728",
            linewidth=2.0,
            label="Best Score",
        )
    ax.set_title("Best Score vs Generation", fontsize=16, fontweight="bold")
    ax.set_xlabel("Generation Created", fontsize=12, fontweight="bold")
    ax.set_ylabel("Simple Score", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)


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

    ax.scatter(xs, ys, color="black", s=18, alpha=0.85, label="Individual Evals")
    ax.plot(xs, best_scores, color="#d62728", linewidth=2.0, label="Best Score")
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


def _plot_lineage_graph(fig: Any, ax: Any, records: list[OrganismVizRecord]) -> None:
    if not records:
        return _empty_panel(ax, "No organisms tracked yet.")

    positions = _lineage_positions(records)
    if not positions:
        return _empty_panel(ax, "No lineage coordinates available.")

    record_by_id = {record.organism_id: record for record in records}
    for record in records:
        child_pos = positions.get(record.organism_id)
        if child_pos is None:
            continue
        for parent_id, linestyle in ((record.mother_id, "-"), (record.father_id, "--")):
            if not parent_id:
                continue
            parent_pos = positions.get(parent_id)
            if parent_pos is None:
                continue
            ax.plot(
                [parent_pos[0], child_pos[0]],
                [parent_pos[1], child_pos[1]],
                color="#b0b0b0",
                linewidth=0.9,
                linestyle=linestyle,
                zorder=1,
                alpha=0.85,
            )

    scored_records = [record for record in records if record.simple_score is not None]
    score_norm = None
    if scored_records:
        score_min = min(record.simple_score for record in scored_records)
        score_max = max(record.simple_score for record in scored_records)
        if math.isclose(score_min, score_max):
            score_max = score_min + 1.0
        score_norm = Normalize(vmin=score_min, vmax=score_max)

    cmap = plt.get_cmap("viridis")
    for record in records:
        x_coord, y_coord = positions[record.organism_id]
        marker = _OPERATOR_MARKERS.get(record.operator, "X")
        facecolor = "#d9d9d9"
        if score_norm is not None and record.simple_score is not None:
            facecolor = cmap(score_norm(record.simple_score))
        ax.scatter(
            [x_coord],
            [y_coord],
            marker=marker,
            s=120 if record.active else 65,
            facecolors=facecolor,
            edgecolors="#2ca02c" if record.active else "black",
            linewidths=2.0 if record.active else 0.6,
            zorder=3,
        )

    best_active = max(
        (record for record in records if record.active and record.simple_score is not None),
        key=lambda record: record.simple_score,
        default=None,
    )
    if best_active is not None:
        x_coord, y_coord = positions[best_active.organism_id]
        ax.annotate(
            "best",
            xy=(x_coord, y_coord),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#d62728",
        )

    ax.set_title("Program Lineage Graph", fontsize=16, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=12, fontweight="bold")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.2)

    generations = sorted({record.generation_created for record in records})
    ax.set_xticks(generations)
    ax.set_xlim(min(generations) - 0.5, max(generations) + 0.5)

    legend_handles = [
        Line2D([0], [0], marker="^", color="w", label="Seed", markerfacecolor="#bdbdbd", markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Mutation", markerfacecolor="#bdbdbd", markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="s", color="w", label="Crossover", markerfacecolor="#bdbdbd", markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Active", markerfacecolor="none", markeredgecolor="#2ca02c", markersize=9, markeredgewidth=2.0),
        Line2D([0], [0], color="#b0b0b0", lw=1.0, linestyle="-", label="Mother edge"),
        Line2D([0], [0], color="#b0b0b0", lw=1.0, linestyle="--", label="Father edge"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, ncol=2)

    if score_norm is not None:
        sm = ScalarMappable(norm=score_norm, cmap=cmap)
        sm.set_array([])
        colorbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Simple Score", fontsize=10, fontweight="bold")


def _lineage_positions(records: list[OrganismVizRecord]) -> dict[str, tuple[float, float]]:
    grouped: dict[int, list[OrganismVizRecord]] = defaultdict(list)
    for record in records:
        grouped[record.generation_created].append(record)

    positions: dict[str, tuple[float, float]] = {}
    for generation, generation_records in sorted(grouped.items()):
        ordered = sorted(
            generation_records,
            key=lambda record: (
                record.island_id,
                -(record.simple_score if record.simple_score is not None else -float("inf")),
                record.organism_id,
            ),
        )
        center = (len(ordered) - 1) / 2.0
        for idx, record in enumerate(ordered):
            positions[record.organism_id] = (float(generation), center - idx)
    return positions


def _evaluated_records(records: list[OrganismVizRecord]) -> list[OrganismVizRecord]:
    evaluated = [record for record in records if record.simple_score is not None]
    evaluated.sort(key=lambda record: (_time_sort_key(record), record.generation_created, record.organism_id))
    return evaluated


def _cumulative_best(values: list[float]) -> list[float]:
    best_values: list[float] = []
    running_best = -float("inf")
    for value in values:
        running_best = max(running_best, value)
        best_values.append(running_best)
    return best_values


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
