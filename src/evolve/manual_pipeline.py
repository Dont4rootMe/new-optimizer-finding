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

from src.evolve.prompt_utils import PromptBundle, load_prompt_bundle
from src.evolve.storage import ensure_dir, parse_genetic_code_text, sha1_text
from src.organisms.crossbreeding import build_crossover_prompt_from_artifacts, merge_gene_pools
from src.organisms.mutation import build_mutation_prompt_from_artifacts, prune_gene_pool
from src.organisms.organism import build_implementation_prompt, load_expected_core_gene_sections_from_config

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

    expected_sections = load_expected_core_gene_sections_from_config(context.cfg)
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
        expected_section_names=load_expected_core_gene_sections_from_config(context.cfg),
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
) -> dict[str, Any]:
    """Build implementation-stage prompts from raw `genetic_code.md` text."""

    genetic_code = parse_genetic_code_text(
        organism_genetic_code_text,
        expected_section_names=load_expected_core_gene_sections_from_config(context.cfg),
    )
    system_prompt, user_prompt = build_implementation_prompt(
        genetic_code=genetic_code,
        change_description=str(novelty_summary),
        prompts=context.prompt_bundle,
    )
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


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
