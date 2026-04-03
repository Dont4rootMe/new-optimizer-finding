"""Crossbreeding operator for canonical organism genetic code."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from src.evolve.prompt_utils import PromptBundle, compose_system_prompt
from src.evolve.storage import sha1_text, utc_now_iso
from src.evolve.template_parser import parse_llm_response
from src.evolve.types import OrganismMeta
from src.organisms.organism import (
    build_organism_from_response,
    format_genetic_code,
    format_lineage_summary,
    load_organism_code_sections,
    read_organism_genetic_code,
    summarize_crossover_gene_diff,
)

LOGGER = logging.getLogger(__name__)


def crossbreed_idea_dna(
    mother_dna: list[str],
    father_dna: list[str],
    p: float = 0.7,
    rng: random.Random | None = None,
) -> list[str]:
    """Recombine gene pools with maternal inheritance probability `p`."""

    if rng is None:
        rng = random.Random()

    seen_lower: set[str] = set()
    child: list[str] = []

    for trait in mother_dna:
        if rng.random() < p:
            key = trait.strip().lower()
            if key not in seen_lower:
                seen_lower.add(key)
                child.append(trait.strip())

    for trait in father_dna:
        if rng.random() < (1.0 - p):
            key = trait.strip().lower()
            if key not in seen_lower:
                seen_lower.add(key)
                child.append(trait.strip())

    if not child:
        fallback = mother_dna if mother_dna else father_dna
        if fallback:
            child.append(rng.choice(fallback).strip())

    return child


def _build_crossbreed_prompt(
    inherited_genes: list[str],
    mother: OrganismMeta,
    father: OrganismMeta,
    prompts: PromptBundle,
) -> tuple[str, str]:
    """Build `(system_prompt, user_prompt)` for crossover LLM call."""

    mother_sections = load_organism_code_sections(mother)
    father_sections = load_organism_code_sections(father)

    system = compose_system_prompt(prompts.project_context, prompts.crossover_system)
    user = prompts.crossover_user.format(
        inherited_gene_pool="\n".join(f"- {gene}" for gene in inherited_genes) or "(none)",
        mother_genetic_code=format_genetic_code(read_organism_genetic_code(mother)),
        mother_selection_reward=mother.selection_reward,
        mother_simple_reward=mother.simple_reward,
        mother_hard_reward=mother.hard_reward,
        mother_lineage_summary=format_lineage_summary(mother.lineage),
        mother_imports=mother_sections.get("IMPORTS", ""),
        mother_init_body=mother_sections.get("INIT_BODY", ""),
        mother_step_body=mother_sections.get("STEP_BODY", ""),
        mother_zero_grad_body=mother_sections.get("ZERO_GRAD_BODY", ""),
        father_genetic_code=format_genetic_code(read_organism_genetic_code(father)),
        father_selection_reward=father.selection_reward,
        father_simple_reward=father.simple_reward,
        father_hard_reward=father.hard_reward,
        father_lineage_summary=format_lineage_summary(father.lineage),
        father_imports=father_sections.get("IMPORTS", ""),
        father_init_body=father_sections.get("INIT_BODY", ""),
        father_step_body=father_sections.get("STEP_BODY", ""),
        father_zero_grad_body=father_sections.get("ZERO_GRAD_BODY", ""),
    )
    return system, user


class CrossbreedingOperator:
    """Two-phase crossbreeding: inherited pool + LLM rewrite."""

    def __init__(self, p: float = 0.7, seed: int | None = None) -> None:
        self.p = p
        self.rng = random.Random(seed)

    def produce(
        self,
        mother: OrganismMeta,
        father: OrganismMeta,
        organism_id: str,
        generation: int,
        org_dir: Path,
        generator: Any,
    ) -> OrganismMeta:
        """Create a child organism via crossbreeding."""

        mother_genes = read_organism_genetic_code(mother).get("core_genes", [])
        father_genes = read_organism_genetic_code(father).get("core_genes", [])
        child_dna = crossbreed_idea_dna(
            mother_genes,
            father_genes,
            self.p,
            self.rng,
        )
        LOGGER.info(
            "Crossbreed %s x %s -> %d genes",
            mother.organism_id[:8],
            father.organism_id[:8],
            len(child_dna),
        )

        system_prompt, user_prompt = _build_crossbreed_prompt(
            child_dna,
            mother,
            father,
            generator.prompt_bundle,
        )
        prompt_hash = sha1_text(system_prompt + "\n" + user_prompt)

        raw_response = generator.call_llm(system_prompt, user_prompt, org_dir)
        parsed = parse_llm_response(raw_response)
        child_genes = [
            line.strip()[2:].strip() if line.strip().startswith("- ") else line.strip()
            for line in parsed.get("CORE_GENES", "").splitlines()
            if line.strip()
        ]
        diff_summary = summarize_crossover_gene_diff(mother_genes, father_genes, child_genes)
        return build_organism_from_response(
            parsed=parsed,
            organism_id=organism_id,
            island_id=mother.island_id,
            generation=generation,
            mother_id=mother.organism_id,
            father_id=father.organism_id,
            operator="crossover",
            org_dir=org_dir,
            model_name=generator.model_name,
            prompt_hash=prompt_hash,
            seed=generator.seed,
            timestamp=utc_now_iso(),
            parent_lineage=mother.lineage,
            gene_diff_summary=diff_summary,
            cross_island=mother.island_id != father.island_id,
            father_island_id=father.island_id,
        )
