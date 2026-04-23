"""Helpers for generic organism validation, persistence, and lineage tracking."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.evolve.prompt_utils import REPO_ROOT, PromptBundle, compose_system_prompt
from src.evolve.storage import (
    genetic_code_path,
    implementation_path,
    lineage_path,
    organism_meta_path,
    parse_genetic_code_text,
    read_genetic_code,
    read_lineage,
    write_genetic_code,
    write_json,
    write_lineage,
)
from src.evolve.types import LineageEntry, OrganismMeta
from src.organisms.genetic_code_format import load_genome_schema

LOGGER = logging.getLogger(__name__)

_MAX_LINEAGE_IN_PROMPT = 5
_REQUIRED_RESPONSE_SECTIONS = (
    "CORE_GENES",
    "INTERACTION_NOTES",
    "COMPUTE_NOTES",
    "CHANGE_DESCRIPTION",
)


def read_organism_implementation(org: OrganismMeta) -> str:
    """Return the raw implementation.py text for one organism."""

    return Path(org.implementation_path).read_text(encoding="utf-8")


def format_implementation_code(code: str) -> str:
    """Format raw implementation code for prompt inclusion."""

    rendered = str(code).rstrip()
    return rendered or "(empty)"


def read_organism_genetic_code(org: OrganismMeta) -> dict[str, Any]:
    """Return canonical genetic code for one organism."""

    return read_genetic_code(org.genetic_code_path)


def read_organism_lineage(org: OrganismMeta) -> list[dict[str, Any]]:
    """Return canonical lineage entries for one organism."""

    return read_lineage(org.lineage_path)


def format_genetic_code(genetic_code: dict[str, Any]) -> str:
    """Format canonical genetic code for inclusion in prompts."""

    core_gene_block = _format_core_gene_block(genetic_code)

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


def _format_gene_entry(text: object) -> str:
    lines = str(text).strip().splitlines()
    if not lines:
        return ""
    rendered = [f"- {lines[0].strip()}"]
    rendered.extend(f"  {line.rstrip()}" for line in lines[1:])
    return "\n".join(rendered)


def _format_core_gene_block(genetic_code: dict[str, Any]) -> str:
    sectioned = genetic_code.get("core_gene_sections")
    if isinstance(sectioned, list) and sectioned:
        blocks: list[str] = []
        for section in sectioned:
            if not isinstance(section, dict):
                continue
            name = str(section.get("name", "")).strip()
            entries = section.get("entries", [])
            if not name:
                continue
            blocks.append(f"### {name}")
            if isinstance(entries, list):
                blocks.extend(_format_gene_entry(entry) for entry in entries if str(entry).strip())
        rendered = "\n".join(block for block in blocks if block.strip()).strip()
        if rendered:
            return rendered

    core_genes = genetic_code.get("core_genes", [])
    if not isinstance(core_genes, list) or not core_genes:
        return "(none)"
    rendered = "\n".join(_format_gene_entry(gene) for gene in core_genes if str(gene).strip())
    return rendered.strip() or "(none)"


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


def format_error_history(errors: list[dict[str, Any]]) -> str:
    """Render evaluator error history for repair prompts."""

    if not errors:
        return "No prior evaluator errors."

    expected_shape_re = re.compile(
        r"expected(?:\s+(?:center|centers)?\s*shape)?\s*[:=]?\s*\(\s*\d+\s*,\s*2\s*\)",
        re.IGNORECASE,
    )
    expected_vector_re = re.compile(
        r"expected(?:\s+(?:radius|radii)?\s*shape)?\s*[:=]?\s*\(\s*\d+\s*,\s*\)",
        re.IGNORECASE,
    )
    required_shape_re = re.compile(
        r"(?:required|target)\s+(?:center|centers)?\s*shape\s*[:=]?\s*\(\s*\d+\s*,\s*2\s*\)",
        re.IGNORECASE,
    )
    required_vector_re = re.compile(
        r"(?:required|target)\s+(?:radius|radii)?\s*shape\s*[:=]?\s*\(\s*\d+\s*,\s*\)",
        re.IGNORECASE,
    )
    lines: list[str] = []
    for entry in errors:
        attempt = entry.get("attempt", "?")
        status = entry.get("status", "unknown")
        error_msg = str(entry.get("error_msg", "")).strip() or "(none)"
        error_msg = expected_shape_re.sub("Expected required center shape", error_msg)
        error_msg = expected_vector_re.sub("Expected required radius shape", error_msg)
        error_msg = required_shape_re.sub("required center shape", error_msg)
        error_msg = required_vector_re.sub("required radius shape", error_msg)
        timestamp = str(entry.get("timestamp", "")).strip() or "(unknown time)"
        lines.append(f"- attempt={attempt} status={status} timestamp={timestamp} error={error_msg}")
    return "\n".join(lines)


def save_organism_artifacts(
    org: OrganismMeta,
    *,
    genetic_code: dict[str, Any],
    lineage: list[dict[str, Any]],
) -> None:
    """Persist organism metadata, genetic code, and lineage to disk."""

    write_json(organism_meta_path(org.organism_dir), org.to_dict())
    write_genetic_code(genetic_code_path(org.organism_dir), genetic_code)
    write_lineage(lineage_path(org.organism_dir), lineage)


def update_latest_lineage_entry(
    org: OrganismMeta,
    *,
    phase: str,
    phase_score: float | None,
    selected_experiments: list[str],
) -> None:
    """Backfill the latest lineage entry after evaluation."""

    lineage = read_organism_lineage(org)
    if not lineage:
        return

    entry = dict(lineage[-1])
    if phase == "simple":
        entry["selected_simple_experiments"] = list(selected_experiments)
        entry["simple_score"] = phase_score
    elif phase == "hard":
        entry["selected_hard_experiments"] = list(selected_experiments)
        entry["hard_score"] = phase_score
    else:
        raise ValueError(f"Unsupported lineage phase '{phase}'")

    lineage[-1] = entry
    write_lineage(org.lineage_path, lineage)


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


def _validate_exact_design_response_sections(parsed: dict[str, str]) -> None:
    actual = tuple(parsed.keys())
    if actual == _REQUIRED_RESPONSE_SECTIONS:
        return
    missing = [name for name in _REQUIRED_RESPONSE_SECTIONS if name not in parsed]
    unexpected = [name for name in actual if name not in _REQUIRED_RESPONSE_SECTIONS]
    details: list[str] = []
    if missing:
        details.append(f"missing required section(s): {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected section(s): {', '.join(unexpected)}")
    if not details:
        details.append(
            "sections are out of order; expected "
            + ", ".join(_REQUIRED_RESPONSE_SECTIONS)
            + "; got "
            + ", ".join(actual)
        )
    raise ValueError("Canonical organism response must contain exactly the required top-level sections: " + "; ".join(details))


def _design_response_genetic_code_text(parsed: dict[str, str]) -> str:
    _validate_exact_design_response_sections(parsed)
    return "\n\n".join(f"## {key}\n{_require_section(parsed, key)}" for key in _REQUIRED_RESPONSE_SECTIONS)


def _build_genetic_code(
    parsed: dict[str, str],
    *,
    expected_core_gene_sections: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    genetic_code = parse_genetic_code_text(
        _design_response_genetic_code_text(parsed),
        expected_section_names=expected_core_gene_sections,
    )
    if genetic_code.get("format_kind") == "legacy_flat":
        genes = list(genetic_code.get("core_genes", []))
        _validate_core_genes(genes)

    return genetic_code


def require_response_section(parsed: dict[str, str], key: str) -> str:
    """Public wrapper for required structured-response sections."""

    return _require_section(parsed, key)


def build_genetic_code_from_design_response(
    parsed: dict[str, str],
    *,
    expected_core_gene_sections: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Public wrapper for canonical genetic-code extraction from design responses."""

    return _build_genetic_code(parsed, expected_core_gene_sections=expected_core_gene_sections)


def load_expected_core_gene_sections_from_config(cfg: Any) -> tuple[str, ...] | None:
    """Return schema-derived CORE_GENES subsection names when config declares a schema."""

    try:
        schema_path_value = cfg.evolver.prompts.get("genome_schema")
    except Exception:  # noqa: BLE001
        return None
    if not schema_path_value:
        return None

    candidate = Path(str(schema_path_value)).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    sections = load_genome_schema(str(candidate))
    return tuple(section.name for section in sections)


def build_implementation_prompt_from_design(
    parsed: dict[str, str],
    prompts: PromptBundle,
    *,
    expected_core_gene_sections: tuple[str, ...] | None = None,
) -> tuple[str, str]:
    """Build the shared implementation-stage prompt from a design-stage response."""

    genetic_code = build_genetic_code_from_design_response(
        parsed,
        expected_core_gene_sections=expected_core_gene_sections,
    )
    change_description = require_response_section(parsed, "CHANGE_DESCRIPTION")
    return build_implementation_prompt(
        genetic_code=genetic_code,
        change_description=change_description,
        prompts=prompts,
    )


def build_implementation_prompt(
    *,
    genetic_code: dict[str, Any],
    change_description: str,
    prompts: PromptBundle,
    compilation_mode: str = "FULL",
    changed_sections: str = "NONE",
    base_parent_genetic_code: str = "NONE",
    base_parent_implementation: str = "NONE",
) -> tuple[str, str]:
    """Build the shared implementation prompt from canonical genetic-code artifacts."""

    normalized_change_description = str(change_description).strip()
    if not normalized_change_description:
        raise ValueError("Implementation prompt requires a non-empty CHANGE_DESCRIPTION.")

    system = compose_system_prompt("", prompts.implementation_system)
    user = prompts.implementation_user.format(
        organism_genetic_code=format_genetic_code(dict(genetic_code)),
        change_description=normalized_change_description,
        implementation_template=prompts.implementation_template,
        compilation_mode=compilation_mode,
        changed_sections=changed_sections,
        base_parent_genetic_code=base_parent_genetic_code,
        base_parent_implementation=base_parent_implementation,
    )
    return system, user


def build_repair_prompt(
    organism: OrganismMeta,
    prompts: PromptBundle,
    *,
    phase: str,
    experiment_name: str,
    errors: list[dict[str, Any]],
) -> tuple[str, str]:
    """Build the repair-stage prompt from organism artifacts and evaluator errors."""

    genetic_code = read_organism_genetic_code(organism)
    lineage = read_organism_lineage(organism)
    current_implementation = read_organism_implementation(organism)
    change_description = "No recorded novelty summary."
    if lineage:
        latest = lineage[-1]
        candidate = str(latest.get("change_description", "")).strip()
        if candidate:
            change_description = candidate

    system = compose_system_prompt("", prompts.repair_system)
    user = prompts.repair_user.format(
        organism_genetic_code=format_genetic_code(genetic_code),
        change_description=change_description,
        current_implementation=format_implementation_code(current_implementation),
        implementation_template=prompts.implementation_template,
        phase=str(phase),
        experiment_name=str(experiment_name),
        error_history=format_error_history(errors),
    )
    return system, user


def build_organism_from_response(
    parsed: dict[str, str],
    implementation_code: str,
    organism_id: str,
    island_id: str,
    generation: int,
    mother_id: str | None,
    father_id: str | None,
    operator: str,
    org_dir: Path,
    llm_route_id: str,
    llm_provider: str,
    provider_model_id: str,
    prompt_hash: str,
    seed: int,
    timestamp: str,
    parent_lineage: list[dict[str, Any]] | None = None,
    ancestor_ids: list[str] | None = None,
    cross_island: bool = False,
    father_island_id: str | None = None,
    expected_core_gene_sections: tuple[str, ...] | None = None,
) -> OrganismMeta:
    """Build `OrganismMeta` from a canonical design response and raw implementation text."""

    _validate_exact_design_response_sections(parsed)

    genetic_code = build_genetic_code_from_design_response(
        parsed,
        expected_core_gene_sections=expected_core_gene_sections,
    )
    change_description = require_response_section(parsed, "CHANGE_DESCRIPTION")
    if not str(implementation_code).strip():
        raise ValueError("Implementation stage must return non-empty implementation.py text.")

    implementation_file = implementation_path(org_dir)
    implementation_file.write_text(implementation_code, encoding="utf-8")

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

    parent_ancestors = list(ancestor_ids or [])
    for parent_id in (mother_id, father_id):
        if parent_id and parent_id not in parent_ancestors:
            parent_ancestors.append(parent_id)

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
        implementation_path=str(implementation_file),
        lineage_path=str(lineage_path(org_dir)),
        organism_dir=str(org_dir),
        ancestor_ids=parent_ancestors,
        experiment_report_index={},
        llm_route_id=llm_route_id,
        llm_provider=llm_provider,
        provider_model_id=provider_model_id,
        prompt_hash=prompt_hash,
        seed=seed,
    )

    save_organism_artifacts(org, genetic_code=genetic_code, lineage=lineage)
    return org
