"""Mutation operator: probabilistic deletion of idea_dna traits.

Phase 1: Each trait is deleted with probability q.
Phase 2: LLM regenerates optimizer code for the reduced idea_dna,
         optionally adding one new idea to replace deleted ones.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from src.evolve.storage import sha1_text, utc_now_iso
from src.evolve.template_parser import extract_editable_sections, parse_llm_response
from src.evolve.types import OrganismMeta
from src.organisms.organism import (
    build_organism_from_response,
    format_evolution_log,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: probabilistic trait deletion
# ---------------------------------------------------------------------------


def mutate_idea_dna(
    dna: list[str],
    q: float = 0.2,
    rng: random.Random | None = None,
) -> tuple[list[str], list[str]]:
    """Delete traits from idea_dna with probability q.

    Returns ``(surviving_traits, removed_traits)``.
    If all traits would be removed, one random trait is kept.
    """
    if rng is None:
        rng = random.Random()

    surviving: list[str] = []
    removed: list[str] = []

    for trait in dna:
        if rng.random() < q:
            removed.append(trait.strip())
        else:
            surviving.append(trait.strip())

    # Guarantee at least one trait survives
    if not surviving and dna:
        rescued = rng.choice(dna).strip()
        surviving.append(rescued)
        if rescued in removed:
            removed.remove(rescued)

    return surviving, removed


# ---------------------------------------------------------------------------
# Phase 2: LLM code generation for the mutated DNA
# ---------------------------------------------------------------------------


def _build_mutate_prompt(
    child_dna: list[str],
    removed_traits: list[str],
    parent: OrganismMeta,
    prompts_dir: Path,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for mutation LLM call."""
    system = (prompts_dir / "mutation_system.txt").read_text(encoding="utf-8")
    user_template = (prompts_dir / "mutate_user.txt").read_text(encoding="utf-8")

    parent_sections = extract_editable_sections(
        Path(parent.optimizer_path).read_text(encoding="utf-8")
    )

    removed_str = "; ".join(removed_traits) if removed_traits else "(none)"

    user = user_template.format(
        child_idea_dna="; ".join(child_dna),
        removed_traits=removed_str,
        parent_idea_dna="; ".join(parent.idea_dna),
        parent_score=parent.score,
        parent_evolution_log=format_evolution_log(parent.evolution_log),
        parent_imports=parent_sections.get("IMPORTS", ""),
        parent_init_body=parent_sections.get("INIT_BODY", ""),
        parent_step_body=parent_sections.get("STEP_BODY", ""),
        parent_zero_grad_body=parent_sections.get("ZERO_GRAD_BODY", ""),
    )
    return system, user


class MutationOperator:
    """Two-phase mutation: probabilistic trait deletion + LLM code gen."""

    def __init__(self, q: float = 0.2, seed: int | None = None) -> None:
        self.q = q
        self.rng = random.Random(seed)

    def produce(
        self,
        parent: OrganismMeta,
        organism_id: str,
        generation: int,
        org_dir: Path,
        generator: Any,
        prompts_dir: Path,
    ) -> OrganismMeta:
        """Create a child organism via mutation.

        Parameters
        ----------
        generator:
            ``OptimizerGenerator`` instance (used for LLM calls).
        prompts_dir:
            Directory containing prompt template files.
        """
        # Phase 1: delete traits
        child_dna, removed = mutate_idea_dna(parent.idea_dna, self.q, self.rng)
        LOGGER.info(
            "Mutate %s: kept %d/%d traits, removed: %s",
            parent.organism_id[:8],
            len(child_dna),
            len(parent.idea_dna),
            removed or "(none)",
        )

        # Phase 2: LLM generates code for child_dna
        system_prompt, user_prompt = _build_mutate_prompt(
            child_dna, removed, parent, prompts_dir
        )
        prompt_hash = sha1_text(system_prompt + user_prompt)

        raw_response = generator.call_llm(system_prompt, user_prompt, org_dir)
        parsed = parse_llm_response(raw_response)

        parent_ids = [parent.organism_id]

        return build_organism_from_response(
            parsed=parsed,
            organism_id=organism_id,
            generation=generation,
            parent_ids=parent_ids,
            operator="mutation",
            org_dir=org_dir,
            model_name=generator.model_name,
            prompt_hash=prompt_hash,
            seed=generator.seed,
            timestamp=utc_now_iso(),
            parent_evolution_log=parent.evolution_log,
            idea_dna_override=child_dna,
        )
