"""Filesystem-backed storage helpers for evolve runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return RFC3339-like UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    """Create directory recursively and return the resolved path."""

    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON payload with stable formatting."""

    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def read_json(path: str | Path) -> dict[str, Any]:
    """Read JSON payload from path."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    """Append one JSON record as a single line."""

    out = Path(path)
    ensure_dir(out.parent)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return out


def sha1_text(text: str) -> str:
    """Compute sha1 hex digest for text."""

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def generation_dir(population_root: str | Path, generation: int) -> Path:
    """Path for one generation directory."""

    return ensure_dir(Path(population_root) / f"gen_{generation}")


def candidate_dir(gen_dir: str | Path, candidate_id: str) -> Path:
    """Path for one candidate directory with required subfolders."""

    base = ensure_dir(Path(gen_dir) / f"cand_{candidate_id}")
    ensure_dir(base / "results")
    ensure_dir(base / "logs")
    return base


def list_candidate_dirs(gen_dir: str | Path) -> list[Path]:
    """Return sorted candidate directories for a generation."""

    root = Path(gen_dir)
    if not root.exists():
        return []
    dirs = [item for item in root.iterdir() if item.is_dir() and item.name.startswith("cand_")]
    return sorted(dirs, key=lambda item: item.name)


def meta_path(cand_dir: str | Path) -> Path:
    return Path(cand_dir) / "meta.json"


def optimizer_path(cand_dir: str | Path) -> Path:
    return Path(cand_dir) / "optimizer.py"


def summary_path(cand_dir: str | Path) -> Path:
    return Path(cand_dir) / "summary.json"


def selection_path(cand_dir: str | Path) -> Path:
    return Path(cand_dir) / "selection.json"


def result_path(cand_dir: str | Path, experiment_name: str) -> Path:
    return Path(cand_dir) / "results" / f"{experiment_name}.json"


def stdout_path(cand_dir: str | Path, experiment_name: str) -> Path:
    return Path(cand_dir) / "logs" / f"{experiment_name}.out"


def stderr_path(cand_dir: str | Path, experiment_name: str) -> Path:
    return Path(cand_dir) / "logs" / f"{experiment_name}.err"


def has_complete_results(cand_dir: str | Path, experiments: list[str]) -> bool:
    """Check whether summary and all experiment result files are present."""

    cand = Path(cand_dir)
    if not summary_path(cand).exists():
        return False
    for exp_name in experiments:
        if not result_path(cand, exp_name).exists():
            return False
    return True


def missing_experiments(cand_dir: str | Path, experiments: list[str]) -> list[str]:
    """Return experiments without saved result JSON."""

    return [exp_name for exp_name in experiments if not result_path(cand_dir, exp_name).exists()]


def organism_dir(gen_dir: Path, organism_id: str) -> Path:
    """Create and return organism directory."""
    d = gen_dir / f"org_{organism_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def idea_dna_path(org_dir: Path) -> Path:
    return org_dir / "idea_dna.txt"


def evolution_log_path(org_dir: Path) -> Path:
    return org_dir / "evolution_log.json"


def organism_meta_path(org_dir: Path) -> Path:
    return org_dir / "organism.json"


def load_population(population_root: str | Path, generation: int) -> list[dict[str, Any]]:
    """Load all organism summaries for a given generation."""
    root = Path(population_root)
    gen_dir = root / f"gen_{generation}"
    if not gen_dir.exists():
        return []

    organisms = []
    for org_path in sorted(gen_dir.glob("org_*")):
        meta_file = organism_meta_path(org_path)
        if meta_file.exists():
            organisms.append(read_json(meta_file))
    return organisms


def load_top_organisms(
    population_root: str | Path,
    limit: int = 5,
    score_key: str = "score",
) -> list[dict[str, Any]]:
    """Load top-scoring organisms across all generations."""
    root = Path(population_root)
    all_organisms: list[dict[str, Any]] = []

    for gen_path in sorted(root.glob("gen_*")):
        for org_path in gen_path.glob("org_*"):
            meta_file = organism_meta_path(org_path)
            if meta_file.exists():
                meta = read_json(meta_file)
                if meta.get(score_key) is not None:
                    all_organisms.append(meta)

    all_organisms.sort(key=lambda x: x.get(score_key, -float("inf")), reverse=True)
    return all_organisms[:limit]


def load_best_context(population_root: str | Path, limit: int = 3) -> list[dict[str, Any]]:
    """Load top candidate summaries from all generations for prompt context."""

    root = Path(population_root)
    if not root.exists():
        return []

    summaries: list[dict[str, Any]] = []
    for summary_file in root.glob("gen_*/cand_*/summary.json"):
        try:
            payload = read_json(summary_file)
        except Exception:
            continue
        if payload.get("status") == "ok" and payload.get("aggregate_score") is not None:
            summaries.append(payload)

    summaries.sort(key=lambda item: float(item.get("aggregate_score", float("-inf"))), reverse=True)
    return summaries[:limit]


def load_recent_experiment_scores(
    population_root: str | Path,
    experiments: list[str],
    history_window: int,
) -> dict[str, list[float]]:
    """Load recent per-experiment exp_score history from candidate summaries."""

    history_window = max(1, int(history_window))
    output: dict[str, list[float]] = {exp_name: [] for exp_name in experiments}

    root = Path(population_root)
    if not root.exists():
        return output

    summary_files = list(root.glob("gen_*/cand_*/summary.json"))

    def _sort_key(path: Path) -> tuple[int, str]:
        generation_name = path.parents[1].name
        generation = -1
        if generation_name.startswith("gen_"):
            try:
                generation = int(generation_name.split("_", 1)[1])
            except ValueError:
                generation = -1
        return generation, path.parent.name

    summary_files.sort(key=_sort_key)

    for summary_file in summary_files:
        try:
            payload = read_json(summary_file)
        except Exception:
            continue

        experiments_payload = payload.get("experiments")
        if not isinstance(experiments_payload, dict):
            continue

        for exp_name in experiments:
            exp_payload = experiments_payload.get(exp_name)
            if not isinstance(exp_payload, dict):
                continue
            if str(exp_payload.get("status", "failed")) != "ok":
                continue
            try:
                score = float(exp_payload.get("exp_score"))
            except (TypeError, ValueError):
                continue
            if score != score:
                continue
            output[exp_name].append(score)
            if len(output[exp_name]) > history_window:
                output[exp_name] = output[exp_name][-history_window:]

    return output
