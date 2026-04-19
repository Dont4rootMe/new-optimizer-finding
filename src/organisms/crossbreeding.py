"""Crossbreeding operator for canonical organism genetic code."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from src.evolve.prompt_utils import PromptBundle, compose_system_prompt
from src.evolve.storage import utc_now_iso
from src.evolve.types import OrganismMeta
from src.organisms.novelty import (
    NoveltyCheckContext,
    build_crossover_novelty_prompt,
    format_novelty_rejection_feedback,
)
from src.organisms.organism import (
    build_organism_from_response,
    format_genetic_code,
    format_lineage_summary,
    read_organism_genetic_code,
    read_organism_lineage,
)

LOGGER = logging.getLogger(__name__)


def merge_gene_pools(
    mother_genes: list[str],
    father_genes: list[str],
    inherit_probability: float = 0.7,
    rng: random.Random | None = None,
) -> list[str]:
    """Recombine maternal and paternal gene pools with maternal bias."""

    if rng is None:
        rng = random.Random()

    seen_lower: set[str] = set()
    child: list[str] = []

    for trait in mother_genes:
        if rng.random() < inherit_probability:
            key = trait.strip().lower()
            if key not in seen_lower:
                seen_lower.add(key)
                child.append(trait.strip())

    for trait in father_genes:
        if rng.random() < (1.0 - inherit_probability):
            key = trait.strip().lower()
            if key not in seen_lower:
                seen_lower.add(key)
                child.append(trait.strip())

    if not child:
        fallback = mother_genes if mother_genes else father_genes
        if fallback:
            child.append(rng.choice(fallback).strip())

    return child


def _build_crossbreed_prompt(
    inherited_genes: list[str],
    mother: OrganismMeta,
    father: OrganismMeta,
    prompts: PromptBundle,
    novelty_feedback: list[str] | None = None,
) -> tuple[str, str]:
    """Build `(system_prompt, user_prompt)` for crossover LLM call."""

    mother_lineage = read_organism_lineage(mother)
    father_lineage = read_organism_lineage(father)

    return build_crossover_prompt_from_artifacts(
        inherited_genes=inherited_genes,
        mother_genetic_code=read_organism_genetic_code(mother),
        mother_lineage=mother_lineage,
        father_genetic_code=read_organism_genetic_code(father),
        father_lineage=father_lineage,
        prompts=prompts,
        novelty_feedback=novelty_feedback,
    )


def build_crossover_prompt_from_artifacts(
    *,
    inherited_genes: list[str],
    mother_genetic_code: dict[str, Any],
    mother_lineage: list[dict[str, Any]],
    father_genetic_code: dict[str, Any],
    father_lineage: list[dict[str, Any]],
    prompts: PromptBundle,
    novelty_feedback: list[str] | None = None,
) -> tuple[str, str]:
    """Build crossover prompts from raw canonical artifacts."""

    system = compose_system_prompt(prompts.project_context, prompts.crossover_system)
    user = prompts.crossover_user.format(
        inherited_gene_pool="\n".join(f"- {gene}" for gene in inherited_genes) or "(none)",
        mother_genetic_code=format_genetic_code(dict(mother_genetic_code)),
        mother_lineage_summary=format_lineage_summary(list(mother_lineage)),
        father_genetic_code=format_genetic_code(dict(father_genetic_code)),
        father_lineage_summary=format_lineage_summary(list(father_lineage)),
        novelty_rejection_feedback=format_novelty_rejection_feedback(list(novelty_feedback or [])),
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
        child_dna = merge_gene_pools(
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
        novelty_context = NoveltyCheckContext(
            operator="crossover",
            build_design_prompts=lambda feedback: _build_crossbreed_prompt(
                child_dna,
                mother,
                father,
                generator.prompt_bundle,
                novelty_feedback=feedback,
            ),
            build_novelty_prompts=lambda candidate_design: build_crossover_novelty_prompt(
                inherited_genes=child_dna,
                mother=mother,
                father=father,
                candidate_design=candidate_design,
                prompts=generator.prompt_bundle,
                expected_core_gene_sections=getattr(generator, "expected_core_gene_sections", None),
            ),
        )
        run_creation = getattr(generator, "run_creation_stages_with_retries", generator.run_creation_stages)
        creation = run_creation(
            design_system_prompt=system_prompt,
            design_user_prompt=user_prompt,
            org_dir=org_dir,
            organism_id=organism_id,
            generation=generation,
            novelty_context=novelty_context,
        )
        mother_lineage = read_organism_lineage(mother)
        ancestor_ids = list(mother.ancestor_ids)
        for parent_id in (mother.organism_id, father.organism_id, *father.ancestor_ids):
            if parent_id and parent_id not in ancestor_ids:
                ancestor_ids.append(parent_id)
        return build_organism_from_response(
            parsed=creation.parsed_design,
            implementation_code=creation.implementation_code,
            organism_id=organism_id,
            island_id=mother.island_id,
            generation=generation,
            mother_id=mother.organism_id,
            father_id=father.organism_id,
            operator="crossover",
            org_dir=org_dir,
            llm_route_id=creation.llm_route_id,
            llm_provider=creation.llm_provider,
            provider_model_id=creation.provider_model_id,
            prompt_hash=creation.prompt_hash,
            seed=generator.seed,
            timestamp=utc_now_iso(),
            parent_lineage=mother_lineage,
            ancestor_ids=ancestor_ids,
            cross_island=mother.island_id != father.island_id,
            father_island_id=father.island_id,
            expected_core_gene_sections=getattr(generator, "expected_core_gene_sections", None),
        )
