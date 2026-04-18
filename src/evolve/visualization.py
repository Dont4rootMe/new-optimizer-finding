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
from matplotlib.ticker import FuncFormatter

from src.evolve.storage import read_json, read_population_state

LOGGER = logging.getLogger(__name__)

OVERVIEW_FILENAME = "evolution_overview.png"
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


def render_evolution_overview(population_root: str | Path) -> Path | None:
    """Render the latest static overview image into `population_root`."""

    return render_evolution_overview_sampled(population_root)


def render_evolution_overview_sampled(
    population_root: str | Path,
    *,
    max_evaluated_points: int | None = None,
    output_filename: str = OVERVIEW_FILENAME,
) -> Path | None:
    """Render the latest static overview image into `population_root`.

    When `max_evaluated_points` is set, render a deterministic evaluation sample
    that preserves the current best maternal line and active evaluated records.
    """

    root = Path(population_root).expanduser().resolve()
    all_records, current_generation, _ = _load_records(root)
    if not all_records:
        return None

    records = _sample_records_for_render(
        all_records,
        max_evaluated_points=max_evaluated_points,
    )
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
