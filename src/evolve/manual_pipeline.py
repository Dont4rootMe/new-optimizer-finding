"""Notebook-oriented helpers for manual organism prompting and simple evaluation."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.evolve.operators import SeedOperator
from src.evolve.prompt_utils import PromptBundle, load_prompt_bundle
from src.evolve.storage import ensure_dir, parse_genetic_code_text, sha1_text
from src.evolve.template_parser import parse_llm_response
from src.evolve.types import Island, OrganismMeta
from src.organisms.compatibility import (
    build_crossover_compatibility_prompt,
    build_mutation_compatibility_prompt,
    build_seed_compatibility_prompt,
)
from src.organisms.crossbreeding import build_crossover_prompt_from_artifacts, merge_gene_pools
from src.organisms.implementation_patch import (
    compute_changed_genome_sections,
    order_changed_sections_by_region_order,
    resolve_implementation_region_order,
)
from src.organisms.mutation import build_mutation_prompt_from_artifacts, prune_gene_pool
from src.organisms.novelty import build_crossover_novelty_prompt, build_mutation_novelty_prompt
from src.organisms.organism import (
    build_genetic_code_from_design_response,
    build_implementation_prompt,
    load_expected_core_gene_sections_from_config,
)

ROOT = Path(__file__).resolve().parents[2]


def _resolve_root_config_name(config_name: str) -> str:
    """Resolve a user-facing preset name to a top-level Hydra config name."""

    raw = str(config_name).strip()
    if not raw:
        raise ValueError("config_name must not be empty.")

    normalized = raw[:-5] if raw.endswith(".yaml") else raw
    conf_dir = ROOT / "conf"

    direct = conf_dir / f"{normalized}.yaml"
    if direct.exists():
        return normalized

    prefixed = conf_dir / f"config_{normalized}.yaml"
    if prefixed.exists():
        return f"config_{normalized}"

    raise FileNotFoundError(
        f"Could not find top-level config preset '{config_name}' in {conf_dir}. "
        f"Tried '{normalized}' and 'config_{normalized}'."
    )


@dataclass(slots=True)
class ManualPipelineContext:
    """Resolved config and prompt bundle for notebook-driven manual runs."""

    repo_root: Path
    config_name: str
    experiment_name: str
    cfg: DictConfig
    experiment_cfg: DictConfig
    prompt_bundle: PromptBundle

    @property
    def available_experiments(self) -> list[str]:
        return [str(name) for name in self.cfg.experiments.keys()]


def load_manual_pipeline_context(
    *,
    config_name: str,
    experiment_name: str | None = None,
    overrides: list[str] | None = None,
) -> ManualPipelineContext:
    """Load one repo config preset and resolve the selected experiment."""

    conf_dir = ROOT / "conf"
    resolved_config_name = _resolve_root_config_name(config_name)
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name=resolved_config_name, overrides=list(overrides or []))

    experiments = [str(name) for name in cfg.experiments.keys()]
    if not experiments:
        raise ValueError(f"Config '{config_name}' does not define cfg.experiments.")

    selected_experiment = str(experiment_name).strip() if experiment_name is not None else ""
    if not selected_experiment:
        if len(experiments) != 1:
            joined = ", ".join(experiments)
            raise ValueError(
                f"Config '{config_name}' defines multiple experiments: {joined}. "
                "Pass experiment_name explicitly for scoring."
            )
        selected_experiment = experiments[0]

    if selected_experiment not in cfg.experiments:
        joined = ", ".join(experiments)
        raise KeyError(
            f"Experiment '{selected_experiment}' is not present in cfg.experiments for '{config_name}'. "
            f"Available: {joined}"
        )

    prompt_bundle = load_prompt_bundle(cfg)
    experiment_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.experiments[selected_experiment], resolve=False),
    )
    return ManualPipelineContext(
        repo_root=ROOT,
        config_name=resolved_config_name,
        experiment_name=selected_experiment,
        cfg=cfg,
        experiment_cfg=experiment_cfg,
        prompt_bundle=prompt_bundle,
    )


def _parse_lineage_payload(payload: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        lineage = payload
    else:
        text = str(payload).strip()
        if not text:
            return []
        lineage = json.loads(text)
    if not isinstance(lineage, list):
        raise ValueError("Lineage payload must be a JSON list.")
    for idx, entry in enumerate(lineage):
        if not isinstance(entry, dict):
            raise ValueError(f"Lineage entry #{idx} must be a JSON object.")
    return [dict(entry) for entry in lineage]


def _resolve_seed(context: ManualPipelineContext, seed: int | None) -> int:
    return int(context.cfg.seed if seed is None else seed)


def _expected_sections(context: ManualPipelineContext) -> tuple[str, ...] | None:
    return load_expected_core_gene_sections_from_config(context.cfg)


def _implementation_regions(context: ManualPipelineContext) -> tuple[str, ...] | None:
    expected_sections = _expected_sections(context)
    if expected_sections is None:
        return None
    template = context.prompt_bundle.implementation_template
    if not template:
        return None
    return resolve_implementation_region_order(template, expected_section_names=expected_sections)


def _parse_design_text(context: ManualPipelineContext, design_text: str) -> dict[str, str]:
    parsed = parse_llm_response(design_text)
    build_genetic_code_from_design_response(parsed, expected_core_gene_sections=_expected_sections(context))
    return parsed


def _manual_organism_from_genetic_code_text(
    context: ManualPipelineContext,
    genetic_code_text: str,
    *,
    organism_id: str,
) -> OrganismMeta:
    organism_dir = ROOT / ".tmp_manual_pipeline" / "prompt_contexts" / sha1_text(organism_id + genetic_code_text)
    ensure_dir(organism_dir)
    genetic_code_path = organism_dir / "genetic_code.md"
    implementation_path = organism_dir / "implementation.py"
    lineage_path = organism_dir / "lineage.json"
    genetic_code_path.write_text(genetic_code_text, encoding="utf-8")
    if not implementation_path.exists():
        implementation_path.write_text("# manual prompt context placeholder\n", encoding="utf-8")
    if not lineage_path.exists():
        lineage_path.write_text("[]\n", encoding="utf-8")
    return OrganismMeta(
        organism_id=organism_id,
        island_id="manual",
        generation_created=0,
        current_generation_active=0,
        timestamp="manual",
        mother_id=None,
        father_id=None,
        operator="manual",
        genetic_code_path=str(genetic_code_path),
        implementation_path=str(implementation_path),
        lineage_path=str(lineage_path),
        organism_dir=str(organism_dir),
    )


def _resolve_manual_island(context: ManualPipelineContext, island_id: str | None) -> Island:
    islands_dir = (ROOT / str(context.cfg.evolver.islands.dir)).resolve()
    if not islands_dir.exists():
        raise FileNotFoundError(f"Configured islands directory was not found: {islands_dir}")
    if island_id:
        candidate = islands_dir / f"{island_id}.txt"
    else:
        candidates = sorted(path for path in islands_dir.glob("*.txt") if path.is_file())
        if not candidates:
            raise FileNotFoundError(f"No island descriptions found in {islands_dir}")
        candidate = candidates[0]
    if not candidate.exists():
        raise FileNotFoundError(f"Island description was not found: {candidate}")
    resolved_id = candidate.stem
    return Island(
        island_id=resolved_id,
        name=resolved_id.replace("_", " "),
        description_path=str(candidate),
        description_text=candidate.read_text(encoding="utf-8").strip(),
    )


def build_manual_seed_prompts(
    context: ManualPipelineContext,
    *,
    island_id: str | None = None,
) -> dict[str, Any]:
    """Build seed design prompts using the configured section-aware prompt bundle."""

    island = _resolve_manual_island(context, island_id)
    system_prompt, user_prompt = SeedOperator(island).build_prompts(context.prompt_bundle)
    return {
        "island_id": island.island_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def build_manual_crossover_prompts(
    context: ManualPipelineContext,
    *,
    mother_genetic_code_text: str,
    father_genetic_code_text: str,
    mother_lineage_json: str | list[dict[str, Any]] | None = None,
    father_lineage_json: str | list[dict[str, Any]] | None = None,
    inherit_probability: float | None = None,
    seed: int | None = None,
    novelty_feedback: list[str] | None = None,
) -> dict[str, Any]:
    """Sample a crossover child gene pool and build design prompts."""

    expected_sections = _expected_sections(context)
    mother_genetic_code = parse_genetic_code_text(
        mother_genetic_code_text,
        expected_section_names=expected_sections,
    )
    father_genetic_code = parse_genetic_code_text(
        father_genetic_code_text,
        expected_section_names=expected_sections,
    )
    mother_lineage = _parse_lineage_payload(mother_lineage_json)
    father_lineage = _parse_lineage_payload(father_lineage_json)
    probability = (
        float(context.cfg.evolver.operators.crossover.primary_parent_gene_inheritance_probability)
        if inherit_probability is None
        else float(inherit_probability)
    )
    child_gene_pool = merge_gene_pools(
        list(mother_genetic_code.get("core_genes", [])),
        list(father_genetic_code.get("core_genes", [])),
        inherit_probability=probability,
        rng=random.Random(_resolve_seed(context, seed)),
    )
    system_prompt, user_prompt = build_crossover_prompt_from_artifacts(
        inherited_genes=child_gene_pool,
        mother_genetic_code=mother_genetic_code,
        mother_lineage=mother_lineage,
        father_genetic_code=father_genetic_code,
        father_lineage=father_lineage,
        prompts=context.prompt_bundle,
        novelty_feedback=novelty_feedback,
    )
    return {
        "child_gene_pool": child_gene_pool,
        "inherit_probability": probability,
        "seed": _resolve_seed(context, seed),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def build_manual_mutation_prompts(
    context: ManualPipelineContext,
    *,
    parent_genetic_code_text: str,
    parent_lineage_json: str | list[dict[str, Any]] | None = None,
    delete_probability: float | None = None,
    seed: int | None = None,
    novelty_feedback: list[str] | None = None,
) -> dict[str, Any]:
    """Prune a parent gene pool and build mutation design prompts."""

    parent_genetic_code = parse_genetic_code_text(
        parent_genetic_code_text,
        expected_section_names=_expected_sections(context),
    )
    parent_lineage = _parse_lineage_payload(parent_lineage_json)
    probability = (
        float(context.cfg.evolver.operators.mutation.gene_removal_probability)
        if delete_probability is None
        else float(delete_probability)
    )
    inherited_genes, removed_genes = prune_gene_pool(
        list(parent_genetic_code.get("core_genes", [])),
        delete_probability=probability,
        rng=random.Random(_resolve_seed(context, seed)),
    )
    system_prompt, user_prompt = build_mutation_prompt_from_artifacts(
        inherited_genes=inherited_genes,
        removed_genes=removed_genes,
        parent_genetic_code=parent_genetic_code,
        parent_lineage=parent_lineage,
        prompts=context.prompt_bundle,
        novelty_feedback=novelty_feedback,
    )
    return {
        "inherited_genes": inherited_genes,
        "removed_genes": removed_genes,
        "delete_probability": probability,
        "seed": _resolve_seed(context, seed),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def build_manual_implementation_prompts(
    context: ManualPipelineContext,
    *,
    organism_genetic_code_text: str,
    novelty_summary: str,
    compilation_mode: str = "FULL",
    changed_sections: tuple[str, ...] | list[str] | None = None,
    base_parent_genetic_code_text: str | None = None,
    base_parent_implementation_text: str | None = None,
) -> dict[str, Any]:
    """Build implementation-stage prompts from raw `genetic_code.md` text."""

    expected_sections = _expected_sections(context)
    implementation_regions = _implementation_regions(context)
    genetic_code = parse_genetic_code_text(
        organism_genetic_code_text,
        expected_section_names=expected_sections,
    )
    normalized_mode = str(compilation_mode).strip().upper()
    if normalized_mode not in {"FULL", "PATCH"}:
        raise ValueError("compilation_mode must be FULL or PATCH.")
    if normalized_mode == "FULL":
        section_tuple = tuple(implementation_regions or ())
        base_genetic_code = "NONE"
        base_implementation = "NONE"
    else:
        if not base_parent_genetic_code_text or not base_parent_implementation_text:
            raise ValueError("PATCH manual implementation prompts require maternal base genetic code and implementation.")
        if changed_sections is None:
            if expected_sections is None or implementation_regions is None:
                raise ValueError("PATCH manual implementation prompts require schema-derived section names.")
            changed_genome_sections = compute_changed_genome_sections(
                base_parent_genetic_code_text,
                organism_genetic_code_text,
                expected_section_names=expected_sections,
            )
            section_tuple = order_changed_sections_by_region_order(
                changed_genome_sections,
                region_order=implementation_regions,
            )
        else:
            section_tuple = tuple(str(section).strip() for section in changed_sections if str(section).strip())
        base_genetic_code = base_parent_genetic_code_text
        base_implementation = base_parent_implementation_text
    rendered_changed_sections = "\n".join(section_tuple) if section_tuple else "NONE"
    system_prompt, user_prompt = build_implementation_prompt(
        genetic_code=genetic_code,
        change_description=str(novelty_summary),
        prompts=context.prompt_bundle,
        compilation_mode=normalized_mode,
        changed_sections=rendered_changed_sections,
        base_parent_genetic_code=base_genetic_code,
        base_parent_implementation=base_implementation,
    )
    return {
        "compilation_mode": normalized_mode,
        "changed_sections": section_tuple,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def build_manual_mutation_novelty_prompts(
    context: ManualPipelineContext,
    *,
    inherited_genes: list[str],
    removed_genes: list[str],
    parent_genetic_code_text: str,
    candidate_design_text: str,
) -> dict[str, Any]:
    """Build mutation novelty-check prompts from manual text artifacts."""

    parent = _manual_organism_from_genetic_code_text(context, parent_genetic_code_text, organism_id="manual_parent")
    parsed_candidate = _parse_design_text(context, candidate_design_text)
    system_prompt, user_prompt = build_mutation_novelty_prompt(
        inherited_genes=inherited_genes,
        removed_genes=removed_genes,
        parent=parent,
        candidate_design=parsed_candidate,
        prompts=context.prompt_bundle,
        expected_core_gene_sections=_expected_sections(context),
    )
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def build_manual_crossover_novelty_prompts(
    context: ManualPipelineContext,
    *,
    inherited_genes: list[str],
    mother_genetic_code_text: str,
    father_genetic_code_text: str,
    candidate_design_text: str,
) -> dict[str, Any]:
    """Build crossover novelty-check prompts from manual text artifacts."""

    mother = _manual_organism_from_genetic_code_text(context, mother_genetic_code_text, organism_id="manual_mother")
    father = _manual_organism_from_genetic_code_text(context, father_genetic_code_text, organism_id="manual_father")
    parsed_candidate = _parse_design_text(context, candidate_design_text)
    system_prompt, user_prompt = build_crossover_novelty_prompt(
        inherited_genes=inherited_genes,
        mother=mother,
        father=father,
        candidate_design=parsed_candidate,
        prompts=context.prompt_bundle,
        expected_core_gene_sections=_expected_sections(context),
    )
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def build_manual_seed_compatibility_prompts(
    context: ManualPipelineContext,
    *,
    candidate_design_text: str,
) -> dict[str, Any]:
    """Build seed compatibility-check prompts from a manual candidate design."""

    parsed_candidate = _parse_design_text(context, candidate_design_text)
    system_prompt, user_prompt = build_seed_compatibility_prompt(
        candidate_design=parsed_candidate,
        prompts=context.prompt_bundle,
        expected_core_gene_sections=_expected_sections(context),
    )
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def build_manual_mutation_compatibility_prompts(
    context: ManualPipelineContext,
    *,
    inherited_genes: list[str],
    removed_genes: list[str],
    parent_genetic_code_text: str,
    candidate_design_text: str,
) -> dict[str, Any]:
    """Build mutation compatibility-check prompts from manual text artifacts."""

    parent = _manual_organism_from_genetic_code_text(context, parent_genetic_code_text, organism_id="manual_parent")
    parsed_candidate = _parse_design_text(context, candidate_design_text)
    system_prompt, user_prompt = build_mutation_compatibility_prompt(
        inherited_genes=inherited_genes,
        removed_genes=removed_genes,
        parent=parent,
        candidate_design=parsed_candidate,
        prompts=context.prompt_bundle,
        expected_core_gene_sections=_expected_sections(context),
    )
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def build_manual_crossover_compatibility_prompts(
    context: ManualPipelineContext,
    *,
    inherited_genes: list[str],
    mother_genetic_code_text: str,
    father_genetic_code_text: str,
    candidate_design_text: str,
) -> dict[str, Any]:
    """Build crossover compatibility-check prompts from manual text artifacts."""

    mother = _manual_organism_from_genetic_code_text(context, mother_genetic_code_text, organism_id="manual_mother")
    father = _manual_organism_from_genetic_code_text(context, father_genetic_code_text, organism_id="manual_father")
    parsed_candidate = _parse_design_text(context, candidate_design_text)
    system_prompt, user_prompt = build_crossover_compatibility_prompt(
        inherited_genes=inherited_genes,
        mother=mother,
        father=father,
        candidate_design=parsed_candidate,
        prompts=context.prompt_bundle,
        expected_core_gene_sections=_expected_sections(context),
    )
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def _prepare_experiment_cfg(
    context: ManualPipelineContext,
    *,
    experiment_name: str,
    mode: str,
    organism_dir: str,
    device: str | None,
    precision: str | None,
    seed: int | None,
) -> DictConfig:
    exp_cfg = OmegaConf.create(
        OmegaConf.to_container(context.cfg.experiments[experiment_name], resolve=False),
    )
    exp_cfg.seed = _resolve_seed(context, seed)
    exp_cfg.deterministic = bool(context.cfg.deterministic)
    exp_cfg.runtime = {
        "mode": mode,
        "smoke": mode == "smoke",
        "organism_dir": str(organism_dir),
    }
    exp_cfg.compute.device = str(device or exp_cfg.compute.get("device") or context.cfg.device)
    exp_cfg.compute.precision = str(precision or exp_cfg.compute.get("precision") or context.cfg.precision)
    exp_cfg.compute.num_workers = int(context.cfg.num_workers)
    if mode == "smoke":
        exp_cfg.compute.max_steps = int(exp_cfg.compute.smoke_steps)
    exp_cfg.paths = OmegaConf.create(OmegaConf.to_container(context.cfg.paths, resolve=True))
    if "data" not in exp_cfg:
        exp_cfg.data = {}
    if not exp_cfg.data.get("root"):
        exp_cfg.data.root = str(Path(context.cfg.paths.data_root) / experiment_name)
    return exp_cfg


def _resolve_organism_dir_for_implementation(source_path: str | Path) -> tuple[Path, bool]:
    implementation_path = Path(source_path).expanduser().resolve()
    if not implementation_path.exists():
        raise FileNotFoundError(f"Implementation file was not found: {implementation_path}")
    if implementation_path.suffix != ".py":
        raise ValueError(f"Implementation path must point to a .py file, got: {implementation_path}")

    if implementation_path.name == "implementation.py":
        return implementation_path.parent, False

    alias_root = ensure_dir(ROOT / ".tmp_manual_pipeline" / "implementation_aliases")
    alias_dir = ensure_dir(alias_root / sha1_text(str(implementation_path.parent)))
    shutil.copy2(implementation_path, alias_dir / "implementation.py")
    return alias_dir, True


def run_manual_simple_evaluation(
    context: ManualPipelineContext,
    *,
    implementation_path: str | Path,
    experiment_name: str | None = None,
    mode: str = "smoke",
    device: str | None = None,
    precision: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate one implementation file through the existing experiment contract."""

    selected_experiment = str(experiment_name or context.experiment_name)
    if selected_experiment not in context.cfg.experiments:
        joined = ", ".join(context.available_experiments)
        raise KeyError(f"Unknown experiment '{selected_experiment}'. Available: {joined}")
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported manual evaluation mode '{mode}'. Use 'smoke' or 'full'.")

    organism_dir, used_alias = _resolve_organism_dir_for_implementation(implementation_path)
    exp_cfg = _prepare_experiment_cfg(
        context,
        experiment_name=selected_experiment,
        mode=mode,
        organism_dir=str(organism_dir),
        device=device,
        precision=precision,
        seed=seed,
    )
    experiment = instantiate(exp_cfg, _recursive_=False)
    if not hasattr(experiment, "evaluate_organism"):
        raise AttributeError(
            f"Experiment '{selected_experiment}' must define evaluate_organism(organism_dir, cfg)."
        )

    report = experiment.evaluate_organism(str(organism_dir), exp_cfg)
    if not isinstance(report, dict):
        raise TypeError(
            f"Experiment '{selected_experiment}' returned non-dict payload: {type(report).__name__}"
        )
    if "score" not in report:
        raise ValueError(f"Experiment '{selected_experiment}' report is missing required field 'score'.")

    payload = dict(report)
    payload["source_implementation_path"] = str(Path(implementation_path).expanduser().resolve())
    payload["manual_config_name"] = context.config_name
    payload["manual_experiment_name"] = selected_experiment
    payload["manual_eval_mode"] = mode
    payload["manual_used_alias_dir"] = used_alias
    return payload
