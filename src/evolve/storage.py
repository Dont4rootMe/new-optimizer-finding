"""Filesystem-backed storage helpers for canonical organism evolution."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evolve.types import ManifestEntry, OrganismMeta

_GENERATION_RE = re.compile(r"^gen_(\d+)$")
_SECTION_RE = re.compile(
    r"^##\s+(CORE_GENES|INTERACTION_NOTES|COMPUTE_NOTES)\s*$",
    re.MULTILINE,
)


def utc_now_iso() -> str:
    """Return RFC3339-like UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    """Create directory recursively and return the resolved path."""

    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON payload with stable formatting."""

    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def read_json(path: str | Path) -> Any:
    """Read JSON payload from path."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha1_text(text: str) -> str:
    """Compute sha1 hex digest for text."""

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _generation_dir_name(generation: int) -> str:
    return f"gen_{int(generation):04d}"


def generation_dir(population_root: str | Path, generation: int) -> Path:
    """Path for one generation directory."""

    return ensure_dir(Path(population_root) / _generation_dir_name(generation))


def _generation_sort_key(path: Path) -> tuple[int, str]:
    match = _GENERATION_RE.match(path.name)
    if match is None:
        return (-1, path.name)
    try:
        return (int(match.group(1)), path.name)
    except ValueError:
        return (-1, path.name)


def organism_dir(
    gen_dir: str | Path,
    organism_id: str,
    *,
    island_id: str,
) -> Path:
    """Create and return organism directory with island nesting."""

    base = ensure_dir(Path(gen_dir) / f"island_{island_id}" / f"org_{organism_id}")
    ensure_dir(base / "results" / "simple")
    ensure_dir(base / "results" / "hard")
    ensure_dir(base / "logs")
    return base


def implementation_path(org_dir: str | Path) -> Path:
    return Path(org_dir) / "implementation.py"


def genetic_code_path(org_dir: str | Path) -> Path:
    return Path(org_dir) / "genetic_code.md"


def lineage_path(org_dir: str | Path) -> Path:
    return Path(org_dir) / "lineage.json"


def organism_meta_path(org_dir: str | Path) -> Path:
    return Path(org_dir) / "organism.json"


def organism_summary_path(org_dir: str | Path) -> Path:
    return Path(org_dir) / "summary.json"


def phase_result_path(org_dir: str | Path, phase: str, experiment_name: str) -> Path:
    return Path(org_dir) / "results" / phase / f"{experiment_name}.json"


def phase_stdout_path(org_dir: str | Path, phase: str, experiment_name: str) -> Path:
    return Path(org_dir) / "logs" / f"{phase}_{experiment_name}.out"


def phase_stderr_path(org_dir: str | Path, phase: str, experiment_name: str) -> Path:
    return Path(org_dir) / "logs" / f"{phase}_{experiment_name}.err"


def population_state_path(population_root: str | Path) -> Path:
    return Path(population_root) / "population_state.json"


def _coerce_manifest_entry(entry: OrganismMeta | dict[str, Any]) -> ManifestEntry:
    if isinstance(entry, OrganismMeta):
        return {
            "organism_id": entry.organism_id,
            "island_id": entry.island_id,
            "organism_dir": entry.organism_dir,
            "generation_created": entry.generation_created,
            "current_generation_active": entry.current_generation_active,
            "simple_score": entry.simple_score,
            "hard_score": entry.hard_score,
        }

    if not isinstance(entry, dict):
        raise TypeError("Population state entries must be OrganismMeta or dict payloads.")

    return {
        "organism_id": str(entry["organism_id"]),
        "island_id": str(entry["island_id"]),
        "organism_dir": str(entry["organism_dir"]),
        "generation_created": int(entry["generation_created"]),
        "current_generation_active": int(entry["current_generation_active"]),
        "simple_score": entry.get("simple_score"),
        "hard_score": entry.get("hard_score"),
    }


def _build_relationship_history(population_root: str | Path) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for meta_path in sorted(
        Path(population_root).glob("gen_*/island_*/org_*/organism.json"),
        key=lambda path: (_generation_sort_key(path.parents[2])[0], str(path)),
    ):
        try:
            payload = read_json(meta_path)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(payload, dict):
            continue
        history.append(
            {
                "organism_id": payload.get("organism_id"),
                "mother_id": payload.get("mother_id"),
                "father_id": payload.get("father_id"),
                "island_id": payload.get("island_id"),
                "generation_created": payload.get("generation_created"),
                "operator": payload.get("operator"),
                "organism_dir": str(meta_path.parent.resolve()),
            }
        )
    return history


def write_population_state(
    population_root: str | Path,
    generation: int,
    organisms: list[OrganismMeta | dict[str, Any]],
    *,
    best_organism_id: str | None = None,
    best_simple_score: float | None = None,
    inflight_generation: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "current_generation": int(generation),
        "active_organisms": [_coerce_manifest_entry(entry) for entry in organisms],
        "best_organism_id": best_organism_id,
        "best_simple_score": best_simple_score,
        "timestamp": utc_now_iso(),
        "relationship_history": _build_relationship_history(population_root),
        "inflight_generation": inflight_generation,
    }
    return write_json(population_state_path(population_root), payload)


def _coerce_int_field(value: Any, field_name: str, *, path: Path) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"JSON payload at {path} contains non-int {field_name}: {value!r}") from exc


def _validate_manifest_entry(
    entry: Any,
    *,
    idx: int,
    path: Path,
    seen_ids: set[str],
    seen_dirs: set[str],
) -> ManifestEntry:
    if not isinstance(entry, dict):
        raise ValueError(f"Population state at {path} entry #{idx} must be a JSON object.")

    organism_id = str(entry.get("organism_id", "")).strip()
    if not organism_id:
        raise ValueError(f"Population state at {path} entry #{idx} is missing organism_id.")
    if organism_id in seen_ids:
        raise ValueError(f"Population state at {path} contains duplicate organism_id '{organism_id}'.")

    island_id = entry.get("island_id")
    if not isinstance(island_id, str) or not island_id.strip():
        raise ValueError(f"Population state at {path} entry #{idx} is missing island_id.")

    organism_dir_value = entry.get("organism_dir")
    if not isinstance(organism_dir_value, str) or not organism_dir_value.strip():
        raise ValueError(f"Population state at {path} entry #{idx} is missing organism_dir.")
    organism_dir_path = Path(organism_dir_value).expanduser()
    if not organism_dir_path.is_absolute():
        organism_dir_path = (path.parent / organism_dir_path).resolve()
    else:
        organism_dir_path = organism_dir_path.resolve()
    organism_dir = str(organism_dir_path)
    if organism_dir in seen_dirs:
        raise ValueError(f"Population state at {path} contains duplicate organism_dir '{organism_dir}'.")

    generation_created = _coerce_int_field(entry.get("generation_created"), "generation_created", path=path)
    current_generation_active = _coerce_int_field(
        entry.get("current_generation_active"),
        "current_generation_active",
        path=path,
    )

    seen_ids.add(organism_id)
    seen_dirs.add(organism_dir)
    return {
        "organism_id": organism_id,
        "island_id": island_id,
        "organism_dir": organism_dir,
        "generation_created": generation_created,
        "current_generation_active": current_generation_active,
        "simple_score": entry.get("simple_score"),
        "hard_score": entry.get("hard_score"),
    }


def read_population_state(population_root: str | Path) -> dict[str, Any] | None:
    path = population_state_path(population_root)
    if not path.exists():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Population state at {path} must be a JSON object.")
    generation = _coerce_int_field(payload.get("current_generation"), "current_generation", path=path)
    active = payload.get("active_organisms")
    if not isinstance(active, list):
        raise ValueError(f"Population state at {path} must contain active_organisms list.")
    seen_ids: set[str] = set()
    seen_dirs: set[str] = set()
    validated = [
        _validate_manifest_entry(entry, idx=idx, path=path, seen_ids=seen_ids, seen_dirs=seen_dirs)
        for idx, entry in enumerate(active)
    ]
    return {
        "current_generation": generation,
        "active_organisms": validated,
        "best_organism_id": payload.get("best_organism_id"),
        "best_simple_score": payload.get("best_simple_score"),
        "timestamp": payload.get("timestamp"),
        "relationship_history": payload.get("relationship_history", []),
        "inflight_generation": payload.get("inflight_generation"),
    }


def _render_genetic_code(genetic_code: dict[str, Any]) -> str:
    core_genes = genetic_code.get("core_genes", [])
    if not isinstance(core_genes, list):
        core_genes = []
    interaction_notes = str(genetic_code.get("interaction_notes", "")).strip()
    compute_notes = str(genetic_code.get("compute_notes", "")).strip()
    gene_lines = "\n".join(f"- {str(gene).strip()}" for gene in core_genes if str(gene).strip())
    return (
        "## CORE_GENES\n"
        f"{gene_lines}\n\n"
        "## INTERACTION_NOTES\n"
        f"{interaction_notes}\n\n"
        "## COMPUTE_NOTES\n"
        f"{compute_notes}\n"
    )


def _parse_genetic_code_sections(text: str) -> dict[str, str]:
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[key] = text[start:end].strip()
    return sections


def write_genetic_code(path: str | Path, genetic_code: dict[str, Any]) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(_render_genetic_code(genetic_code), encoding="utf-8")
    return out


def read_genetic_code(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Canonical genetic code is missing at {target}")

    sections = _parse_genetic_code_sections(target.read_text(encoding="utf-8"))
    required_sections = ("CORE_GENES", "INTERACTION_NOTES", "COMPUTE_NOTES")
    missing_sections = [section for section in required_sections if section not in sections]
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(f"Canonical genetic code at {target} is malformed; missing required sections: {missing}")

    core_genes_raw = sections.get("CORE_GENES", "")
    core_genes: list[str] = []
    for line in core_genes_raw.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("- "):
            cleaned = cleaned[2:].strip()
        if cleaned:
            core_genes.append(cleaned)
    if not core_genes:
        raise ValueError(f"Canonical genetic code at {target} must contain at least one CORE_GENES bullet.")

    return {
        "core_genes": core_genes,
        "interaction_notes": sections.get("INTERACTION_NOTES", ""),
        "compute_notes": sections.get("COMPUTE_NOTES", ""),
    }


def write_lineage(path: str | Path, lineage: list[dict[str, Any]]) -> Path:
    return write_json(path, lineage)


def read_lineage(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Canonical lineage is missing at {target}")

    payload = read_json(target)
    if not isinstance(payload, list):
        raise ValueError(f"Lineage payload at {target} must be a JSON list.")
    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"Lineage payload at {target} entry #{idx} must be a JSON object.")
    return payload


def _require_str_field(
    payload: dict[str, Any],
    field_name: str,
    *,
    meta_file: Path,
) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Canonical organism meta at {meta_file} is missing required field '{field_name}'.")
    return value


def _require_path_field(
    payload: dict[str, Any],
    field_name: str,
    *,
    org_dir: Path,
    meta_file: Path,
) -> str:
    value = _require_str_field(payload, field_name, meta_file=meta_file)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (org_dir / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def _optional_str_field(payload: dict[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_organism_meta_payload(
    payload: dict[str, Any],
    *,
    org_dir: Path,
    meta_file: Path,
) -> dict[str, Any]:
    return {
        "organism_id": _require_str_field(payload, "organism_id", meta_file=meta_file),
        "island_id": _require_str_field(payload, "island_id", meta_file=meta_file),
        "generation_created": _coerce_int_field(payload.get("generation_created"), "generation_created", path=meta_file),
        "current_generation_active": _coerce_int_field(
            payload.get("current_generation_active"),
            "current_generation_active",
            path=meta_file,
        ),
        "timestamp": _require_str_field(payload, "timestamp", meta_file=meta_file),
        "mother_id": _optional_str_field(payload, "mother_id"),
        "father_id": _optional_str_field(payload, "father_id"),
        "operator": _require_str_field(payload, "operator", meta_file=meta_file),
        "genetic_code_path": _require_path_field(payload, "genetic_code_path", org_dir=org_dir, meta_file=meta_file),
        "implementation_path": _require_path_field(
            payload,
            "implementation_path",
            org_dir=org_dir,
            meta_file=meta_file,
        ),
        "lineage_path": _require_path_field(payload, "lineage_path", org_dir=org_dir, meta_file=meta_file),
        "organism_dir": _require_path_field(payload, "organism_dir", org_dir=org_dir, meta_file=meta_file),
        "ancestor_ids": list(payload.get("ancestor_ids", [])),
        "experiment_report_index": dict(payload.get("experiment_report_index", {})),
        "simple_score": payload.get("simple_score"),
        "hard_score": payload.get("hard_score"),
        "status": _require_str_field(payload, "status", meta_file=meta_file),
        "llm_route_id": str(payload.get("llm_route_id", "")),
        "llm_provider": str(payload.get("llm_provider", payload.get("provider", ""))),
        "provider_model_id": str(payload.get("provider_model_id", payload.get("model_name", ""))),
        "model_name": str(payload.get("model_name", payload.get("provider_model_id", ""))),
        "prompt_hash": str(payload.get("prompt_hash", "")),
        "seed": int(payload.get("seed", 0)),
        "pipeline_state": str(payload.get("pipeline_state", "")),
        "planned_phase_evaluations": dict(payload.get("planned_phase_evaluations", {})),
    }


def write_organism_meta(org: OrganismMeta) -> Path:
    return write_json(organism_meta_path(org.organism_dir), org.to_dict())


def read_organism_meta(path: str | Path) -> OrganismMeta:
    target = Path(path)
    meta_file = target if target.name.endswith(".json") else organism_meta_path(target)
    payload = read_json(meta_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Organism meta at {meta_file} must be a JSON object.")

    org_dir = meta_file.parent
    canonical = _coerce_organism_meta_payload(payload, org_dir=org_dir, meta_file=meta_file)
    if not Path(canonical["implementation_path"]).exists():
        raise FileNotFoundError(
            f"Canonical implementation is missing at {canonical['implementation_path']}"
        )
    if not Path(canonical["genetic_code_path"]).exists():
        raise FileNotFoundError(
            f"Canonical genetic code is missing at {canonical['genetic_code_path']}"
        )
    if not Path(canonical["lineage_path"]).exists():
        raise FileNotFoundError(
            f"Canonical lineage is missing at {canonical['lineage_path']}"
        )
    return OrganismMeta(
        organism_id=canonical["organism_id"],
        island_id=canonical["island_id"],
        generation_created=canonical["generation_created"],
        current_generation_active=canonical["current_generation_active"],
        timestamp=canonical["timestamp"],
        mother_id=canonical["mother_id"],
        father_id=canonical["father_id"],
        operator=canonical["operator"],
        genetic_code_path=canonical["genetic_code_path"],
        implementation_path=canonical["implementation_path"],
        lineage_path=canonical["lineage_path"],
        organism_dir=canonical["organism_dir"],
        ancestor_ids=canonical["ancestor_ids"],
        experiment_report_index=canonical["experiment_report_index"],
        simple_score=canonical["simple_score"],
        hard_score=canonical["hard_score"],
        status=canonical["status"],
        llm_route_id=canonical["llm_route_id"],
        llm_provider=canonical["llm_provider"],
        provider_model_id=canonical["provider_model_id"],
        model_name=canonical["model_name"],
        prompt_hash=canonical["prompt_hash"],
        seed=canonical["seed"],
        pipeline_state=canonical["pipeline_state"],
        planned_phase_evaluations=canonical["planned_phase_evaluations"],
    )


def write_organism_summary(org_dir: str | Path, payload: dict[str, Any]) -> Path:
    return write_json(organism_summary_path(org_dir), payload)


def read_organism_summary(org_dir: str | Path) -> dict[str, Any] | None:
    path = organism_summary_path(org_dir)
    if not path.exists():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Organism summary at {path} must be a JSON object.")
    return payload


def load_recent_organism_experiment_scores(
    population_root: str | Path,
    experiments: list[str],
    history_window: int,
) -> dict[str, list[float]]:
    """Load recent experiment score history from organism summaries only."""

    history_window = max(1, int(history_window))
    output: dict[str, list[float]] = {exp_name: [] for exp_name in experiments}

    root = Path(population_root)
    if not root.exists():
        return output

    summary_files = sorted(
        root.glob("gen_*/island_*/org_*/summary.json"),
        key=lambda path: (_generation_sort_key(path.parents[2])[0], str(path)),
    )

    for summary_file in summary_files:
        try:
            payload = read_json(summary_file)
        except Exception:  # noqa: BLE001
            continue

        phase_results = payload.get("phase_results")
        if not isinstance(phase_results, dict):
            continue
        for phase_payload in phase_results.values():
            if not isinstance(phase_payload, dict):
                continue
            nested_experiments = phase_payload.get("experiments")
            if not isinstance(nested_experiments, dict):
                continue
            for exp_name in experiments:
                exp_payload = nested_experiments.get(exp_name)
                if isinstance(exp_payload, dict):
                    _append_experiment_score(output, exp_name, exp_payload, history_window)

    return output


def _append_experiment_score(
    output: dict[str, list[float]],
    exp_name: str,
    exp_payload: dict[str, Any],
    history_window: int,
) -> None:
    if str(exp_payload.get("status", "failed")) != "ok":
        return
    try:
        score = float(exp_payload.get("score"))
    except (TypeError, ValueError):
        return
    if score != score:
        return
    output[exp_name].append(score)
    if len(output[exp_name]) > history_window:
        output[exp_name] = output[exp_name][-history_window:]
