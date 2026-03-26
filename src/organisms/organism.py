"""Helpers for loading, saving, and building organisms."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.evolve.storage import (
    evolution_log_path,
    idea_dna_path,
    organism_meta_path,
    write_json,
)
from src.evolve.template_parser import (
    extract_editable_sections,
    render_template,
    validate_rendered_code,
)
from src.evolve.types import EvolutionEntry, OrganismMeta

LOGGER = logging.getLogger(__name__)

_MAX_EVOLUTION_LOG_IN_PROMPT = 5


def load_organism_code_sections(org: OrganismMeta) -> dict[str, str]:
    """Load editable code sections from an organism's optimizer.py."""
    code = Path(org.optimizer_path).read_text(encoding="utf-8")
    return extract_editable_sections(code)


def format_evolution_log(
    entries: list[dict[str, Any]],
    limit: int = _MAX_EVOLUTION_LOG_IN_PROMPT,
) -> str:
    """Format evolution log entries for inclusion in LLM prompts."""
    recent = entries[-limit:] if len(entries) > limit else entries
    if not recent:
        return "No prior evolution history."
    lines = []
    for entry in recent:
        gen = entry.get("generation", "?")
        desc = entry.get("change_description", "")
        score = entry.get("score")
        lines.append(f"  gen={gen}: {desc} (score={score})")
    return "\n".join(lines)


def save_organism_artifacts(
    org: OrganismMeta,
    org_dir: Path,
    idea_dna: list[str],
    evolution_log: list[dict[str, Any]],
) -> None:
    """Persist organism metadata, idea DNA, and evolution log to disk."""
    write_json(organism_meta_path(org_dir), org.to_dict())
    idea_dna_path(org_dir).write_text("; ".join(idea_dna), encoding="utf-8")
    write_json(evolution_log_path(org_dir), evolution_log)


def build_organism_from_response(
    parsed: dict[str, str],
    organism_id: str,
    generation: int,
    parent_ids: list[str],
    operator: str,
    org_dir: Path,
    model_name: str,
    prompt_hash: str,
    seed: int,
    timestamp: str,
    parent_evolution_log: list[dict[str, Any]] | None = None,
    idea_dna_override: list[str] | None = None,
) -> OrganismMeta:
    """Build OrganismMeta from parsed LLM response and render optimizer.py.

    Parameters
    ----------
    idea_dna_override:
        If provided, use this idea_dna instead of parsing from ``parsed``.
        Used by crossbreeding/mutation where the DNA is determined
        before the LLM call.
    """
    if idea_dna_override is not None:
        idea_dna = list(idea_dna_override)
    else:
        idea_dna_str = parsed.get("IDEA_DNA", "")
        idea_dna = [s.strip() for s in idea_dna_str.split(";") if s.strip()]

    change_description = parsed.get("CHANGE_DESCRIPTION", "Initial creation")

    class_name = f"Optimizer_{organism_id[:8]}"
    optimizer_name = class_name

    sections = {
        "IMPORTS": parsed.get("IMPORTS", "import math"),
        "INIT_BODY": parsed.get("INIT_BODY", "        pass"),
        "STEP_BODY": parsed.get("STEP_BODY", "        pass"),
        "ZERO_GRAD_BODY": parsed.get("ZERO_GRAD_BODY", "        pass"),
    }

    # Ensure proper indentation for body sections
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

    optimizer_path = org_dir / "optimizer.py"
    optimizer_path.write_text(code, encoding="utf-8")

    new_entry = EvolutionEntry(
        generation=generation,
        change_description=change_description,
        score=None,
        parent_ids=parent_ids,
    )

    evolution_log = list(parent_evolution_log or [])
    evolution_log.append(new_entry.to_dict())

    org = OrganismMeta(
        organism_id=organism_id,
        generation=generation,
        timestamp=timestamp,
        parent_ids=parent_ids,
        operator=operator,
        idea_dna=idea_dna,
        evolution_log=evolution_log,
        model_name=model_name,
        prompt_hash=prompt_hash,
        seed=seed,
        organism_dir=str(org_dir),
        optimizer_path=str(optimizer_path),
    )

    save_organism_artifacts(org, org_dir, idea_dna, evolution_log)
    return org
