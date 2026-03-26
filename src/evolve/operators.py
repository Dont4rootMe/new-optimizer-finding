"""Genetic operators: seed, mutation, crossover.

The ``SeedOperator`` creates organisms from scratch (generation 0).
``MutationOperator`` and ``CrossoverOperator`` are legacy prompt-only
operators kept for backward compatibility with ``generate_organism()``.

For probabilistic genetic operations on idea_dna, see:
- ``src.organisms.crossbreeding.CrossbreedingOperator``
- ``src.organisms.mutation.MutationOperator``
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.evolve.types import OrganismMeta
from src.organisms.organism import (
    format_evolution_log,
    load_organism_code_sections,
)

LOGGER = logging.getLogger(__name__)

# Re-export build_organism_from_response for generator.py backward compat
from src.organisms.organism import build_organism_from_response as _build_organism_from_response  # noqa: F401, E402


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
    """Legacy prompt-only mutation (delegates entirely to LLM).

    For probabilistic trait deletion, use
    ``src.organisms.mutation.MutationOperator`` instead.
    """

    operator_name = "mutation"

    def build_prompts(self, parents: list[OrganismMeta], prompts_dir: Path) -> tuple[str, str]:
        parent = parents[0]
        sections = load_organism_code_sections(parent)

        system = (prompts_dir / "mutation_system.txt").read_text(encoding="utf-8")
        user_template = (prompts_dir / "mutation_user.txt").read_text(encoding="utf-8")

        user = user_template.format(
            parent_idea_dna="; ".join(parent.idea_dna),
            parent_score=parent.score,
            parent_evolution_log=format_evolution_log(parent.evolution_log),
            parent_imports=sections.get("IMPORTS", ""),
            parent_init_body=sections.get("INIT_BODY", ""),
            parent_step_body=sections.get("STEP_BODY", ""),
            parent_zero_grad_body=sections.get("ZERO_GRAD_BODY", ""),
        )
        return system, user


class CrossoverOperator(GeneticOperator):
    """Legacy prompt-only crossover (delegates entirely to LLM).

    For probabilistic trait recombination, use
    ``src.organisms.crossbreeding.CrossbreedingOperator`` instead.
    """

    operator_name = "crossover"

    def build_prompts(self, parents: list[OrganismMeta], prompts_dir: Path) -> tuple[str, str]:
        parent_a, parent_b = parents[0], parents[1]
        sections_a = load_organism_code_sections(parent_a)
        sections_b = load_organism_code_sections(parent_b)

        system = (prompts_dir / "crossover_system.txt").read_text(encoding="utf-8")
        user_template = (prompts_dir / "crossover_user.txt").read_text(encoding="utf-8")

        user = user_template.format(
            parent_a_idea_dna="; ".join(parent_a.idea_dna),
            parent_a_score=parent_a.score,
            parent_a_evolution_log=format_evolution_log(parent_a.evolution_log),
            parent_a_imports=sections_a.get("IMPORTS", ""),
            parent_a_init_body=sections_a.get("INIT_BODY", ""),
            parent_a_step_body=sections_a.get("STEP_BODY", ""),
            parent_a_zero_grad_body=sections_a.get("ZERO_GRAD_BODY", ""),
            parent_b_idea_dna="; ".join(parent_b.idea_dna),
            parent_b_score=parent_b.score,
            parent_b_evolution_log=format_evolution_log(parent_b.evolution_log),
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
