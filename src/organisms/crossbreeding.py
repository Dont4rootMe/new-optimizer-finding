"""Crossbreeding operator: probabilistic recombination of idea_dna traits.

Phase 1: Each trait from the dominant parent is included with probability p,
          each trait from the non-dominant parent with probability 1-p.
Phase 2: LLM generates optimizer code implementing the resulting idea_dna.
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
# Phase 1: deterministic / probabilistic DNA recombination
# ---------------------------------------------------------------------------


def crossbreed_idea_dna(
    dominant_dna: list[str],
    non_dominant_dna: list[str],
    p: float = 0.7,
    rng: random.Random | None = None,
) -> list[str]:
    """Recombine idea_dna from two parents.

    For each trait in ``dominant_dna``:  include with probability ``p``.
    For each trait in ``non_dominant_dna``: include with probability ``1 - p``.
    Duplicate traits (case-insensitive) are skipped.
    If the result would be empty, one random trait from the dominant parent
    is kept.

    Returns a new list of trait strings.
    """
    if rng is None:
        rng = random.Random()

    seen_lower: set[str] = set()
    child: list[str] = []

    for trait in dominant_dna:
        if rng.random() < p:
            key = trait.strip().lower()
            if key not in seen_lower:
                seen_lower.add(key)
                child.append(trait.strip())

    for trait in non_dominant_dna:
        if rng.random() < (1.0 - p):
            key = trait.strip().lower()
            if key not in seen_lower:
                seen_lower.add(key)
                child.append(trait.strip())

    # Guarantee at least one trait
    if not child:
        fallback = dominant_dna if dominant_dna else non_dominant_dna
        if fallback:
            child.append(rng.choice(fallback).strip())

    return child


# ---------------------------------------------------------------------------
# Phase 2: LLM code generation for the child DNA
# ---------------------------------------------------------------------------


def _build_crossbreed_prompt(
    child_dna: list[str],
    dominant: OrganismMeta,
    non_dominant: OrganismMeta,
    prompts_dir: Path,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for crossbreeding LLM call."""
    system = (prompts_dir / "crossover_system.txt").read_text(encoding="utf-8")
    user_template = (prompts_dir / "crossbreed_user.txt").read_text(encoding="utf-8")

    dom_sections = extract_editable_sections(
        Path(dominant.optimizer_path).read_text(encoding="utf-8")
    )
    non_dom_sections = extract_editable_sections(
        Path(non_dominant.optimizer_path).read_text(encoding="utf-8")
    )

    user = user_template.format(
        child_idea_dna="; ".join(child_dna),
        parent_a_idea_dna="; ".join(dominant.idea_dna),
        parent_a_score=dominant.score,
        parent_a_evolution_log=format_evolution_log(dominant.evolution_log),
        parent_a_imports=dom_sections.get("IMPORTS", ""),
        parent_a_init_body=dom_sections.get("INIT_BODY", ""),
        parent_a_step_body=dom_sections.get("STEP_BODY", ""),
        parent_a_zero_grad_body=dom_sections.get("ZERO_GRAD_BODY", ""),
        parent_b_idea_dna="; ".join(non_dominant.idea_dna),
        parent_b_score=non_dominant.score,
        parent_b_evolution_log=format_evolution_log(non_dominant.evolution_log),
        parent_b_imports=non_dom_sections.get("IMPORTS", ""),
        parent_b_init_body=non_dom_sections.get("INIT_BODY", ""),
        parent_b_step_body=non_dom_sections.get("STEP_BODY", ""),
        parent_b_zero_grad_body=non_dom_sections.get("ZERO_GRAD_BODY", ""),
    )
    return system, user


class CrossbreedingOperator:
    """Two-phase crossbreeding: probabilistic DNA recombination + LLM code gen."""

    def __init__(self, p: float = 0.7, seed: int | None = None) -> None:
        self.p = p
        self.rng = random.Random(seed)

    def produce(
        self,
        dominant: OrganismMeta,
        non_dominant: OrganismMeta,
        organism_id: str,
        generation: int,
        org_dir: Path,
        generator: Any,
        prompts_dir: Path,
    ) -> OrganismMeta:
        """Create a child organism via crossbreeding.

        Parameters
        ----------
        generator:
            ``OptimizerGenerator`` instance (used for LLM calls).
        prompts_dir:
            Directory containing prompt template files.
        """
        # Phase 1: recombine idea_dna
        child_dna = crossbreed_idea_dna(
            dominant.idea_dna, non_dominant.idea_dna, self.p, self.rng
        )
        LOGGER.info(
            "Crossbreed %s x %s -> %d traits (from %d + %d)",
            dominant.organism_id[:8],
            non_dominant.organism_id[:8],
            len(child_dna),
            len(dominant.idea_dna),
            len(non_dominant.idea_dna),
        )

        # Phase 2: LLM generates code for child_dna
        system_prompt, user_prompt = _build_crossbreed_prompt(
            child_dna, dominant, non_dominant, prompts_dir
        )
        prompt_hash = sha1_text(system_prompt + user_prompt)

        raw_response = generator.call_llm(system_prompt, user_prompt, org_dir)
        parsed = parse_llm_response(raw_response)

        # Merge evolution logs from both parents
        merged_log = list(dominant.evolution_log) + list(non_dominant.evolution_log)
        # Sort by generation, keep unique
        merged_log.sort(key=lambda e: e.get("generation", 0))

        parent_ids = [dominant.organism_id, non_dominant.organism_id]

        return build_organism_from_response(
            parsed=parsed,
            organism_id=organism_id,
            generation=generation,
            parent_ids=parent_ids,
            operator="crossover",
            org_dir=org_dir,
            model_name=generator.model_name,
            prompt_hash=prompt_hash,
            seed=generator.seed,
            timestamp=utc_now_iso(),
            parent_evolution_log=merged_log,
            idea_dna_override=child_dna,
        )
