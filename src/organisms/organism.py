"""Helpers for canonical organism validation, persistence, and lineage tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
    render_template,
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
    "IMPORTS",
    "INIT_BODY",
    "STEP_BODY",
    "ZERO_GRAD_BODY",
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
        gene_diff_summary = str(entry.get("gene_diff_summary", "")).strip() or "(none)"
        simple_score = entry.get("simple_score")
        hard_score = entry.get("hard_score")
        cross_island = bool(entry.get("cross_island", False))
        father_island_id = entry.get("father_island_id")

        parts = [
            f"gen={generation}",
            f"op={operator}",
            f"change={change_description}",
            f"genes={gene_diff_summary}",
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


def _lower_to_original(genes: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for gene in genes:
        normalized = gene.strip().lower()
        if normalized and normalized not in mapping:
            mapping[normalized] = gene.strip()
    return mapping


def summarize_mutation_gene_diff(
    parent_genes: list[str],
    child_genes: list[str],
) -> str:
    """Describe actual conceptual mutation deltas using parent-vs-child genes."""

    parent_map = _lower_to_original(parent_genes)
    child_map = _lower_to_original(child_genes)

    removed = [parent_map[key] for key in parent_map.keys() - child_map.keys()]
    added_or_rewritten = [child_map[key] for key in child_map.keys() - parent_map.keys()]
    preserved = [child_map[key] for key in child_map.keys() & parent_map.keys()]

    parts = [
        f"Preserved genes: {', '.join(preserved) if preserved else '(none)'}",
        f"Removed genes: {', '.join(removed) if removed else '(none)'}",
        f"Added or rewritten genes: {', '.join(added_or_rewritten) if added_or_rewritten else '(none)'}",
    ]
    return ". ".join(parts) + "."


def summarize_crossover_gene_diff(
    mother_genes: list[str],
    father_genes: list[str],
    child_genes: list[str],
) -> str:
    """Describe actual conceptual crossover deltas using maternal/paternal provenance."""

    mother_map = _lower_to_original(mother_genes)
    father_map = _lower_to_original(father_genes)
    child_map = _lower_to_original(child_genes)

    maternal_preserved = [child_map[key] for key in child_map.keys() & mother_map.keys()]
    paternal_introduced = [
        child_map[key]
        for key in child_map.keys() & father_map.keys()
        if key not in mother_map
    ]
    rewrites = [
        child_map[key]
        for key in child_map.keys()
        if key not in mother_map and key not in father_map
    ]

    parts = [
        f"Maternal genes preserved: {', '.join(maternal_preserved) if maternal_preserved else '(none)'}",
        f"Paternal genes introduced: {', '.join(paternal_introduced) if paternal_introduced else '(none)'}",
        f"Major rewrites: {', '.join(rewrites) if rewrites else '(none)'}",
    ]
    return ". ".join(parts) + "."


def build_organism_from_response(
    parsed: dict[str, str],
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
    gene_diff_summary: str | None = None,
    cross_island: bool = False,
    father_island_id: str | None = None,
) -> OrganismMeta:
    """Build `OrganismMeta` from a canonical structured LLM response."""

    for key in _REQUIRED_RESPONSE_SECTIONS:
        if key not in parsed:
            raise ValueError(f"Canonical organism response is missing required section {key}.")

    genetic_code = _build_genetic_code(parsed)
    change_description = _require_section(parsed, "CHANGE_DESCRIPTION")
    diff_summary = (gene_diff_summary or "").strip()
    if not diff_summary:
        diff_summary = "No semantic gene diff summary recorded."

    class_name = f"Optimizer_{organism_id[:8]}"
    optimizer_name = class_name

    sections = {
        "IMPORTS": _require_section(parsed, "IMPORTS"),
        "INIT_BODY": parsed.get("INIT_BODY", ""),
        "STEP_BODY": parsed.get("STEP_BODY", ""),
        "ZERO_GRAD_BODY": parsed.get("ZERO_GRAD_BODY", ""),
    }
    for key in ("INIT_BODY", "STEP_BODY", "ZERO_GRAD_BODY"):
        sections[key] = _require_section(parsed, key)

    for key in ("INIT_BODY", "STEP_BODY", "ZERO_GRAD_BODY"):
        lines = sections[key].split("\n")
        indented = []
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                indented.append("")
            elif not stripped.startswith("        "):
                indented.append("        " + stripped.lstrip())
            else:
                indented.append(stripped)
        sections[key] = "\n".join(indented)

    code = render_template(sections, optimizer_name=optimizer_name, class_name=class_name)
    is_valid, error = validate_rendered_code(code)
    if not is_valid:
        raise ValueError(f"Generated code failed validation: {error}")

    optimizer_file = Path(org_dir) / "optimizer.py"
    optimizer_file.write_text(code, encoding="utf-8")

    lineage = list(parent_lineage or [])
    lineage.append(
        LineageEntry(
            generation=generation,
            operator=operator,
            mother_id=mother_id,
            father_id=father_id,
            change_description=change_description,
            gene_diff_summary=diff_summary,
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
