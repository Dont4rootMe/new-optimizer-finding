"""Helpers for canonical organism validation, persistence, and lineage tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.evolve.prompt_utils import PromptBundle, compose_system_prompt
from src.evolve.storage import (
    genetic_code_path,
    lineage_path,
    organism_meta_path,
    read_genetic_code,
    write_genetic_code,
    write_json,
    write_lineage,
)
from src.evolve.template_parser import (
    extract_editable_sections,
    validate_rendered_code,
)
from src.evolve.types import LineageEntry, OrganismMeta

LOGGER = logging.getLogger(__name__)

_MAX_LINEAGE_IN_PROMPT = 5
_REQUIRED_RESPONSE_SECTIONS = (
    "CORE_GENES",
    "INTERACTION_NOTES",
    "COMPUTE_NOTES",
    "CHANGE_DESCRIPTION",
)


def load_organism_code_sections(org: OrganismMeta) -> dict[str, str]:
    """Load editable code sections from an organism's optimizer.py."""

    code = Path(org.optimizer_path).read_text(encoding="utf-8")
    return extract_editable_sections(code)


def read_organism_genetic_code(org: OrganismMeta) -> dict[str, Any]:
    """Return canonical genetic code, loading it from disk when needed."""

    if org.genetic_code:
        return dict(org.genetic_code)
    genetic_code = read_genetic_code(org.genetic_code_path)
    org.genetic_code = dict(genetic_code)
    return genetic_code


def format_genetic_code(genetic_code: dict[str, Any]) -> str:
    """Format canonical genetic code for inclusion in prompts."""

    core_genes = genetic_code.get("core_genes", [])
    if not isinstance(core_genes, list) or not core_genes:
        core_gene_block = "(none)"
    else:
        core_gene_block = "\n".join(f"- {str(gene).strip()}" for gene in core_genes if str(gene).strip())
        if not core_gene_block.strip():
            core_gene_block = "(none)"

    interaction_notes = str(genetic_code.get("interaction_notes", "")).strip() or "(none)"
    compute_notes = str(genetic_code.get("compute_notes", "")).strip() or "(none)"

    return (
        "CORE_GENES:\n"
        f"{core_gene_block}\n\n"
        "INTERACTION_NOTES:\n"
        f"{interaction_notes}\n\n"
        "COMPUTE_NOTES:\n"
        f"{compute_notes}"
    )


def format_lineage_summary(
    entries: list[dict[str, Any]],
    limit: int = _MAX_LINEAGE_IN_PROMPT,
) -> str:
    """Format recent lineage entries for inclusion in prompts."""

    recent = entries[-limit:] if len(entries) > limit else entries
    if not recent:
        return "No prior lineage history."

    lines: list[str] = []
    for entry in recent:
        generation = entry.get("generation", "?")
        operator = entry.get("operator", "?")
        change_description = str(entry.get("change_description", "")).strip() or "(none)"
        simple_score = entry.get("simple_score")
        hard_score = entry.get("hard_score")
        cross_island = bool(entry.get("cross_island", False))
        father_island_id = entry.get("father_island_id")

        parts = [
            f"gen={generation}",
            f"op={operator}",
            f"change={change_description}",
            f"simple={simple_score}",
            f"hard={hard_score}",
        ]
        if cross_island:
            parts.append(f"cross_island=true:{father_island_id}")
        lines.append(" | ".join(parts))

    return "\n".join(lines)


def save_organism_artifacts(org: OrganismMeta) -> None:
    """Persist organism metadata, genetic code, and lineage to disk."""

    write_json(organism_meta_path(org.organism_dir), org.to_dict())
    write_genetic_code(genetic_code_path(org.organism_dir), org.genetic_code)
    write_lineage(lineage_path(org.organism_dir), org.lineage)


def update_latest_lineage_entry(
    org: OrganismMeta,
    *,
    phase: str,
    phase_score: float | None,
    selected_experiments: list[str],
) -> None:
    """Backfill the latest lineage entry after evaluation."""

    if not org.lineage:
        return

    entry = dict(org.lineage[-1])
    if phase == "simple":
        entry["selected_simple_experiments"] = list(selected_experiments)
        entry["simple_score"] = phase_score
    elif phase == "hard":
        entry["selected_hard_experiments"] = list(selected_experiments)
        entry["hard_score"] = phase_score
    else:
        raise ValueError(f"Unsupported lineage phase '{phase}'")

    org.lineage[-1] = entry
    write_lineage(org.lineage_path, org.lineage)


def _normalize_gene_lines(core_genes_raw: str) -> list[str]:
    genes: list[str] = []
    for line in core_genes_raw.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("- "):
            cleaned = cleaned[2:].strip()
        if cleaned:
            genes.append(cleaned)
    return genes


def _validate_core_genes(genes: list[str]) -> None:
    if len(genes) < 3:
        raise ValueError("Canonical organism response must contain at least 3 non-empty CORE_GENES bullets.")
    for gene in genes:
        if len(gene.split()) < 2:
            raise ValueError(f"Canonical organism gene is too thin: {gene!r}")


def _require_section(parsed: dict[str, str], key: str) -> str:
    value = parsed.get(key)
    if value is None:
        raise ValueError(f"Canonical organism response is missing required section {key}.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"Canonical organism response section {key} must not be empty.")
    return stripped


def _build_genetic_code(parsed: dict[str, str]) -> dict[str, Any]:
    genes = _normalize_gene_lines(_require_section(parsed, "CORE_GENES"))
    _validate_core_genes(genes)

    interaction_notes = _require_section(parsed, "INTERACTION_NOTES")
    compute_notes = _require_section(parsed, "COMPUTE_NOTES")
    return {
        "core_genes": genes,
        "interaction_notes": interaction_notes,
        "compute_notes": compute_notes,
    }


def build_implementation_prompt_from_design(
    parsed: dict[str, str],
    prompts: PromptBundle,
) -> tuple[str, str]:
    """Build the shared implementation-stage prompt from a design-stage response."""

    genetic_code = _build_genetic_code(parsed)
    change_description = _require_section(parsed, "CHANGE_DESCRIPTION")
    system = compose_system_prompt(prompts.project_context, prompts.implementation_system)
    user = prompts.implementation_user.format(
        organism_genetic_code=format_genetic_code(genetic_code),
        change_description=change_description,
        implementation_template=prompts.implementation_template,
    )
    return system, user


def build_organism_from_response(
    parsed: dict[str, str],
    optimizer_code: str,
    implementation_template: str,
    organism_id: str,
    island_id: str,
    generation: int,
    mother_id: str | None,
    father_id: str | None,
    operator: str,
    org_dir: Path,
    model_name: str,
    prompt_hash: str,
    seed: int,
    timestamp: str,
    parent_lineage: list[dict[str, Any]] | None = None,
    cross_island: bool = False,
    father_island_id: str | None = None,
) -> OrganismMeta:
    """Build `OrganismMeta` from a canonical design response and full optimizer source."""

    for key in _REQUIRED_RESPONSE_SECTIONS:
        if key not in parsed:
            raise ValueError(f"Canonical organism response is missing required section {key}.")

    genetic_code = _build_genetic_code(parsed)
    change_description = _require_section(parsed, "CHANGE_DESCRIPTION")

    is_valid, error = validate_rendered_code(optimizer_code, implementation_template)
    if not is_valid:
        raise ValueError(f"Generated code failed validation: {error}")

    optimizer_file = Path(org_dir) / "optimizer.py"
    optimizer_file.write_text(optimizer_code, encoding="utf-8")

    lineage = list(parent_lineage or [])
    lineage.append(
        LineageEntry(
            generation=generation,
            operator=operator,
            mother_id=mother_id,
            father_id=father_id,
            change_description=change_description,
            cross_island=cross_island,
            father_island_id=father_island_id,
        ).to_dict()
    )

    org = OrganismMeta(
        organism_id=organism_id,
        island_id=island_id,
        generation_created=generation,
        current_generation_active=generation,
        timestamp=timestamp,
        mother_id=mother_id,
        father_id=father_id,
        operator=operator,
        genetic_code_path=str(genetic_code_path(org_dir)),
        optimizer_path=str(optimizer_file),
        lineage_path=str(lineage_path(org_dir)),
        organism_dir=str(org_dir),
        model_name=model_name,
        prompt_hash=prompt_hash,
        seed=seed,
        genetic_code=genetic_code,
        lineage=lineage,
    )

    save_organism_artifacts(org)
    return org
