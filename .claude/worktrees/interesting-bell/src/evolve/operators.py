"""Genetic operators: seed, mutation, crossover."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.evolve.storage import (
    evolution_log_path,
    idea_dna_path,
    organism_meta_path,
    read_json,
    write_json,
)
from src.evolve.template_parser import (
    extract_editable_sections,
    parse_llm_response,
    render_template,
    validate_rendered_code,
)
from src.evolve.types import EvolutionEntry, OrganismMeta

LOGGER = logging.getLogger(__name__)

_MAX_EVOLUTION_LOG_IN_PROMPT = 5


def _load_organism_code_sections(org: OrganismMeta) -> dict[str, str]:
    """Load editable sections from an organism's optimizer.py."""
    code = Path(org.optimizer_path).read_text(encoding="utf-8")
    return extract_editable_sections(code)


def _format_evolution_log(entries: list[dict[str, Any]], limit: int = _MAX_EVOLUTION_LOG_IN_PROMPT) -> str:
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


def _save_organism_artifacts(
    org: OrganismMeta,
    org_dir: Path,
    idea_dna: list[str],
    evolution_log: list[dict[str, Any]],
) -> None:
    """Save organism metadata, idea DNA, and evolution log to disk."""
    write_json(organism_meta_path(org_dir), org.to_dict())
    idea_dna_path(org_dir).write_text("; ".join(idea_dna), encoding="utf-8")
    write_json(evolution_log_path(org_dir), evolution_log)


def _build_organism_from_response(
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
) -> OrganismMeta:
    """Build OrganismMeta from parsed LLM response and render optimizer.py."""
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

    _save_organism_artifacts(org, org_dir, idea_dna, evolution_log)
    return org


class GeneticOperator(ABC):
    """Base class for genetic operators."""

    @abstractmethod
    def build_prompts(
        self,
        parents: list[OrganismMeta],
        prompts_dir: Path,
    ) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for LLM call."""

    @property
    @abstractmethod
    def operator_name(self) -> str: ...

    def parent_ids(self, parents: list[OrganismMeta]) -> list[str]:
        return [p.organism_id for p in parents]


class SeedOperator(GeneticOperator):
    """Create a new organism from scratch (generation 0)."""

    operator_name = "seed"

    def build_prompts(self, parents: list[OrganismMeta], prompts_dir: Path) -> tuple[str, str]:
        system = (prompts_dir / "seed_system.txt").read_text(encoding="utf-8")
        user = (prompts_dir / "seed_user.txt").read_text(encoding="utf-8")
        return system, user


class MutationOperator(GeneticOperator):
    """Mutate a single parent organism."""

    operator_name = "mutation"

    def build_prompts(self, parents: list[OrganismMeta], prompts_dir: Path) -> tuple[str, str]:
        parent = parents[0]
        sections = _load_organism_code_sections(parent)

        system = (prompts_dir / "mutation_system.txt").read_text(encoding="utf-8")
        user_template = (prompts_dir / "mutation_user.txt").read_text(encoding="utf-8")

        user = user_template.format(
            parent_idea_dna="; ".join(parent.idea_dna),
            parent_score=parent.score,
            parent_evolution_log=_format_evolution_log(parent.evolution_log),
            parent_imports=sections.get("IMPORTS", ""),
            parent_init_body=sections.get("INIT_BODY", ""),
            parent_step_body=sections.get("STEP_BODY", ""),
            parent_zero_grad_body=sections.get("ZERO_GRAD_BODY", ""),
        )
        return system, user


class CrossoverOperator(GeneticOperator):
    """Crossover between two parent organisms."""

    operator_name = "crossover"

    def build_prompts(self, parents: list[OrganismMeta], prompts_dir: Path) -> tuple[str, str]:
        parent_a, parent_b = parents[0], parents[1]
        sections_a = _load_organism_code_sections(parent_a)
        sections_b = _load_organism_code_sections(parent_b)

        system = (prompts_dir / "crossover_system.txt").read_text(encoding="utf-8")
        user_template = (prompts_dir / "crossover_user.txt").read_text(encoding="utf-8")

        user = user_template.format(
            parent_a_idea_dna="; ".join(parent_a.idea_dna),
            parent_a_score=parent_a.score,
            parent_a_evolution_log=_format_evolution_log(parent_a.evolution_log),
            parent_a_imports=sections_a.get("IMPORTS", ""),
            parent_a_init_body=sections_a.get("INIT_BODY", ""),
            parent_a_step_body=sections_a.get("STEP_BODY", ""),
            parent_a_zero_grad_body=sections_a.get("ZERO_GRAD_BODY", ""),
            parent_b_idea_dna="; ".join(parent_b.idea_dna),
            parent_b_score=parent_b.score,
            parent_b_evolution_log=_format_evolution_log(parent_b.evolution_log),
            parent_b_imports=sections_b.get("IMPORTS", ""),
            parent_b_init_body=sections_b.get("INIT_BODY", ""),
            parent_b_step_body=sections_b.get("STEP_BODY", ""),
            parent_b_zero_grad_body=sections_b.get("ZERO_GRAD_BODY", ""),
        )
        return system, user


OPERATORS: dict[str, type[GeneticOperator]] = {
    "seed": SeedOperator,
    "mutation": MutationOperator,
    "crossover": CrossoverOperator,
}
