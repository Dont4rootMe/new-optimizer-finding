"""Unit tests for generic organism generator behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from api_platforms import LlmResponse
import src.evolve.operators as canonical_seed_operators
from src.evolve.generator import CandidateGenerator
from src.evolve.template_parser import parse_llm_response
from src.evolve.types import Island, OrganismMeta
from src.organisms.compatibility import (
    CompatibilityCheckContext,
    CompatibilityRejectionExhaustedError,
    CompatibilityValidationContext,
    format_compatibility_rejection_feedback,
)
from src.organisms.novelty import NoveltyCheckContext, NoveltyRejectionExhaustedError
from src.organisms.organism import save_organism_artifacts

CIRCLE_CORE_SECTIONS = (
    "INIT_GEOMETRY",
    "RADIUS_POLICY",
    "EXPANSION_POLICY",
    "CONFLICT_MODEL",
    "REPAIR_POLICY",
    "CONTROL_POLICY",
    "PARAMETERS",
    "OPTIONAL_CODE_SKETCH",
)
CIRCLE_IMPLEMENTATION_REGIONS = (
    "PARAMETERS",
    "INIT_GEOMETRY",
    "RADIUS_POLICY",
    "CONFLICT_MODEL",
    "REPAIR_POLICY",
    "EXPANSION_POLICY",
    "CONTROL_POLICY",
    "OPTIONAL_CODE_SKETCH",
)
OPTIMIZER_REGIONS = (
    "STATE_REPRESENTATION",
    "GRADIENT_PROCESSING",
    "UPDATE_RULE",
    "PARAMETER_GROUP_POLICY",
    "STEP_CONTROL_POLICY",
    "STABILITY_POLICY",
    "PARAMETERS",
    "OPTIONAL_CODE_SKETCH",
)
ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return OmegaConf.create(
        {
            "seed": 123,
            "evolver": {
                "creation": {
                    "max_attempts_to_create_organism": 2,
                    "max_attempts_to_repair_organism_after_error": 2,
                    "max_attempts_to_regenerate_organism_after_novelty_rejection": 1,
                    "max_attempts_to_regenerate_organism_after_compatibility_rejection": 1,
                },
                "prompts": {
                    "project_context": "conf/experiments/optimization_survey/prompts/shared/project_context.txt",
                    "genome_schema": "conf/experiments/optimization_survey/prompts/shared/genome_schema.txt",
                    "seed_system": "conf/experiments/optimization_survey/prompts/seed/system.txt",
                    "seed_user": "conf/experiments/optimization_survey/prompts/seed/user.txt",
                    "compatibility_seed_system": "conf/experiments/optimization_survey/prompts/compatibility/seed/system.txt",
                    "compatibility_seed_user": "conf/experiments/optimization_survey/prompts/compatibility/seed/user.txt",
                    "mutation_system": "conf/experiments/optimization_survey/prompts/mutation/system.txt",
                    "mutation_user": "conf/experiments/optimization_survey/prompts/mutation/user.txt",
                    "mutation_novelty_system": "conf/experiments/optimization_survey/prompts/novelty/mutation/system.txt",
                    "mutation_novelty_user": "conf/experiments/optimization_survey/prompts/novelty/mutation/user.txt",
                    "compatibility_mutation_system": "conf/experiments/optimization_survey/prompts/compatibility/mutation/system.txt",
                    "compatibility_mutation_user": "conf/experiments/optimization_survey/prompts/compatibility/mutation/user.txt",
                    "crossover_system": "conf/experiments/optimization_survey/prompts/crossover/system.txt",
                    "crossover_user": "conf/experiments/optimization_survey/prompts/crossover/user.txt",
                    "crossover_novelty_system": "conf/experiments/optimization_survey/prompts/novelty/crossover/system.txt",
                    "crossover_novelty_user": "conf/experiments/optimization_survey/prompts/novelty/crossover/user.txt",
                    "compatibility_crossover_system": "conf/experiments/optimization_survey/prompts/compatibility/crossover/system.txt",
                    "compatibility_crossover_user": "conf/experiments/optimization_survey/prompts/compatibility/crossover/user.txt",
                    "implementation_system": "conf/experiments/optimization_survey/prompts/implementation/system.txt",
                    "implementation_user": "conf/experiments/optimization_survey/prompts/implementation/user.txt",
                    "implementation_template": "conf/experiments/optimization_survey/prompts/shared/template.txt",
                    "repair_system": "conf/experiments/optimization_survey/prompts/repair/system.txt",
                    "repair_user": "conf/experiments/optimization_survey/prompts/repair/user.txt",
                },
                "llm": {"route_weights": {"mock": 1.0}, "seed": 123},
            },
            "api_platforms": {"mock": {"_target_": "api_platforms.mock.platform.build_platform"}},
            "paths": {"api_platform_runtime_root": ".tmp_api_platform_runtime"},
        }
    )


def _circle_cfg():
    return OmegaConf.create(
        {
            "seed": 123,
            "evolver": {
                "creation": {
                    "max_attempts_to_create_organism": 1,
                    "max_attempts_to_repair_organism_after_error": 1,
                    "max_attempts_to_regenerate_organism_after_novelty_rejection": 1,
                    "max_attempts_to_regenerate_organism_after_compatibility_rejection": 1,
                },
                "prompts": {
                    "project_context": "conf/experiments/circle_packing_shinka/prompts/shared/project_context.txt",
                    "genome_schema": "conf/experiments/circle_packing_shinka/prompts/shared/genome_schema.txt",
                    "seed_system": "conf/experiments/circle_packing_shinka/prompts/seed/system.txt",
                    "seed_user": "conf/experiments/circle_packing_shinka/prompts/seed/user.txt",
                    "mutation_system": "conf/experiments/circle_packing_shinka/prompts/mutation/system.txt",
                    "mutation_user": "conf/experiments/circle_packing_shinka/prompts/mutation/user.txt",
                    "mutation_novelty_system": "conf/experiments/circle_packing_shinka/prompts/novelty/mutation/system.txt",
                    "mutation_novelty_user": "conf/experiments/circle_packing_shinka/prompts/novelty/mutation/user.txt",
                    "crossover_system": "conf/experiments/circle_packing_shinka/prompts/crossover/system.txt",
                    "crossover_user": "conf/experiments/circle_packing_shinka/prompts/crossover/user.txt",
                    "crossover_novelty_system": "conf/experiments/circle_packing_shinka/prompts/novelty/crossover/system.txt",
                    "crossover_novelty_user": "conf/experiments/circle_packing_shinka/prompts/novelty/crossover/user.txt",
                    "implementation_system": "conf/experiments/circle_packing_shinka/prompts/implementation/system.txt",
                    "implementation_user": "conf/experiments/circle_packing_shinka/prompts/implementation/user.txt",
                    "implementation_template": "conf/experiments/circle_packing_shinka/prompts/shared/template.txt",
                    "repair_system": "conf/experiments/circle_packing_shinka/prompts/repair/system.txt",
                    "repair_user": "conf/experiments/circle_packing_shinka/prompts/repair/user.txt",
                },
                "llm": {"route_weights": {"mock": 1.0}, "seed": 123},
            },
            "api_platforms": {"mock": {"_target_": "api_platforms.mock.platform.build_platform"}},
            "paths": {"api_platform_runtime_root": ".tmp_api_platform_runtime"},
        }
    )


def _circle_genetic_code(*, radius_policy: str = "Use uniform radii.") -> dict[str, object]:
    entries = {
        "INIT_GEOMETRY": "Start from a centered scaffold.",
        "RADIUS_POLICY": radius_policy,
        "EXPANSION_POLICY": "Do not expand after initialization.",
        "CONFLICT_MODEL": "Track boundary and pairwise violations.",
        "REPAIR_POLICY": "Use deterministic local shifts.",
        "CONTROL_POLICY": "Run construction once.",
        "PARAMETERS": "Use radius 0.04.",
        "OPTIONAL_CODE_SKETCH": "None.",
    }
    return {
        "core_gene_sections": [
            {"name": name, "entries": [entries[name]]}
            for name in CIRCLE_CORE_SECTIONS
        ],
        "interaction_notes": "Sections are coherent.",
        "compute_notes": "Deterministic constructive code.",
        "change_description": "A sectioned test design.",
    }


def _circle_design_text(*, radius_policy: str = "Use uniform radii.") -> str:
    genes = _circle_genetic_code(radius_policy=radius_policy)
    core = "\n\n".join(
        f"### {section['name']}\n- {section['entries'][0]}"
        for section in genes["core_gene_sections"]  # type: ignore[index]
    )
    return (
        "## CORE_GENES\n"
        f"{core}\n\n"
        "## INTERACTION_NOTES\n"
        "Sections are coherent.\n\n"
        "## COMPUTE_NOTES\n"
        "Deterministic constructive code.\n\n"
        "## CHANGE_DESCRIPTION\n"
        "A sectioned test design.\n"
    )


def _circle_region_body(region: str, *, patched: bool = False) -> str:
    if region == "INIT_GEOMETRY":
        return (
            "    centers = np.asarray([[0.10 + 0.06 * (i % 13), "
            "0.20 + 0.20 * (i // 13)] for i in range(26)], dtype=float)\n"
        )
    if region == "RADIUS_POLICY":
        radius = "0.05" if patched else "0.04"
        return f"    radii = np.full(26, {radius}, dtype=float)\n"
    return f"    # {'patched' if patched else 'base'} {region}\n"


def _circle_patch_response(mode: str, regions: tuple[str, ...], *, patched: bool = False) -> str:
    pieces = ["## COMPILATION_MODE", mode]
    for region in regions:
        pieces.extend(("", f"## REGION {region}", _circle_region_body(region, patched=patched).rstrip("\n"), "## END_REGION"))
    return "\n".join(pieces) + "\n"


def _circle_scaffold_template() -> str:
    return (
        ROOT / "conf" / "experiments" / "circle_packing_shinka" / "prompts" / "shared" / "template.txt"
    ).read_text(encoding="utf-8").strip()


def _circle_base_source() -> str:
    return (
        "import numpy as np\n\n"
        "def run_packing():\n"
        "    centers = None\n"
        "    radii = None\n\n"
        "    # EVOLVE-BLOCK-START\n"
        "    # SECTION: PARAMETERS\n"
        "    # base PARAMETERS\n"
        "    # SECTION: INIT_GEOMETRY\n"
        "    centers = np.asarray([[0.10 + 0.06 * (i % 13), 0.20 + 0.20 * (i // 13)] for i in range(26)], dtype=float)\n"
        "    # SECTION: RADIUS_POLICY\n"
        "    radii = np.full(26, 0.04, dtype=float)\n"
        "    # SECTION: CONFLICT_MODEL\n"
        "    # base CONFLICT_MODEL\n"
        "    # SECTION: REPAIR_POLICY\n"
        "    # base REPAIR_POLICY\n"
        "    # SECTION: EXPANSION_POLICY\n"
        "    # base EXPANSION_POLICY\n"
        "    # SECTION: CONTROL_POLICY\n"
        "    # base CONTROL_POLICY\n"
        "    # SECTION: OPTIONAL_CODE_SKETCH\n"
        "    # base OPTIONAL_CODE_SKETCH\n"
        "    # EVOLVE-BLOCK-END\n\n"
        "    centers = np.asarray(centers, dtype=float)\n"
        "    radii = np.asarray(radii, dtype=float)\n"
        "    reported_sum = float(np.sum(radii))\n"
        "    return centers, radii, reported_sum\n"
    )


def _make_circle_parent(tmp_path: Path, *, implementation_source: str | None = None) -> OrganismMeta:
    organism_dir = tmp_path / "circle_parent"
    organism_dir.mkdir(parents=True, exist_ok=True)
    implementation_path = organism_dir / "implementation.py"
    implementation_path.write_text(implementation_source or _circle_base_source(), encoding="utf-8")
    organism = OrganismMeta(
        organism_id="circle_parent",
        island_id="symmetric_constructions",
        generation_created=0,
        current_generation_active=0,
        timestamp="2026-01-01T00:00:00Z",
        mother_id=None,
        father_id=None,
        operator="seed",
        genetic_code_path=str(organism_dir / "genetic_code.md"),
        implementation_path=str(implementation_path),
        lineage_path=str(organism_dir / "lineage.json"),
        organism_dir=str(organism_dir),
        llm_route_id="mock",
        llm_provider="mock",
        provider_model_id="mock-model",
        prompt_hash="abc",
        seed=123,
    )
    save_organism_artifacts(
        organism,
        genetic_code=_circle_genetic_code(),
        lineage=[
            {
                "generation": 0,
                "operator": "seed",
                "mother_id": None,
                "father_id": None,
                "change_description": "Initial sectioned circle organism.",
                "selected_simple_experiments": ["unit_square_26"],
                "selected_hard_experiments": [],
                "simple_score": 1.0,
                "hard_score": None,
                "cross_island": False,
                "father_island_id": None,
            }
        ],
    )
    implementation_path.write_text(implementation_source or _circle_base_source(), encoding="utf-8")
    return organism


def _make_repairable_organism(tmp_path: Path) -> OrganismMeta:
    organism_dir = tmp_path / "org_repair"
    organism_dir.mkdir(parents=True, exist_ok=True)
    implementation_path = organism_dir / "implementation.py"
    implementation_path.write_text(
        "import torch.nn as nn\n\n"
        "class Impl:\n"
        "    def __init__(self, model: nn.Module, max_steps: int) -> None:\n"
        "        self.model = model\n"
        "        self.max_steps = max_steps\n\n"
        "    def step(self, weights, grads, activations, step_fn) -> None:\n"
        "        del weights, grads, activations, step_fn\n\n"
        "    def zero_grad(self, set_to_none: bool = True) -> None:\n"
        "        del set_to_none\n\n"
        "def build_optimizer(model: nn.Module, max_steps: int):\n"
        "    return Impl(model, max_steps)\n",
        encoding="utf-8",
    )
    organism = OrganismMeta(
        organism_id="repair01",
        island_id="gradient_methods",
        generation_created=0,
        current_generation_active=0,
        timestamp="2026-01-01T00:00:00Z",
        mother_id=None,
        father_id=None,
        operator="seed",
        genetic_code_path=str(organism_dir / "genetic_code.md"),
        implementation_path=str(implementation_path),
        lineage_path=str(organism_dir / "lineage.json"),
        organism_dir=str(organism_dir),
        llm_route_id="mock",
        llm_provider="mock",
        provider_model_id="mock-model",
        prompt_hash="abc",
        seed=123,
    )
    save_organism_artifacts(
        organism,
        genetic_code={
            "core_genes": [
                "Per-parameter first moment tracking for stable updates",
                "Second-moment normalization to damp noisy gradients",
                "Deterministic schedule that shrinks the effective step size late in training",
            ],
            "interaction_notes": "Baseline optimizer notes.",
            "compute_notes": "Low-overhead baseline.",
        },
        lineage=[
            {
                "generation": 0,
                "operator": "seed",
                "mother_id": None,
                "father_id": None,
                "change_description": "Initial optimizer baseline.",
                "selected_simple_experiments": ["simple_a"],
                "selected_hard_experiments": [],
                "simple_score": 0.5,
                "hard_score": None,
                "cross_island": False,
                "father_island_id": None,
            }
        ],
    )
    return organism


def _design_text(change_description: str = "Candidate change.") -> str:
    core = "\n\n".join(
        (
            f"### {section}\n"
            f"{'- None.' if section == OPTIMIZER_REGIONS[-1] else '- candidate optimizer idea for ' + section.lower()}"
        )
        for section in OPTIMIZER_REGIONS
    )
    return (
        "## CORE_GENES\n"
        f"{core}\n\n"
        "## INTERACTION_NOTES\n"
        "Candidate.\n\n"
        "## COMPUTE_NOTES\n"
        "Cheap.\n\n"
        "## CHANGE_DESCRIPTION\n"
        f"{change_description}\n"
    )


def _implementation_text() -> str:
    pieces = ["## COMPILATION_MODE", "FULL"]
    for region in OPTIMIZER_REGIONS:
        pieces.extend(("", f"## REGION {region}", f"        # implementation body for {region}", "## END_REGION"))
    return "\n".join(pieces) + "\n"


def _novelty_accepted_text() -> str:
    return (
        "## NOVELTY_VERDICT\n"
        "NOVELTY_ACCEPTED\n\n"
        "## REJECTION_REASON\n"
        "N/A\n\n"
        "## SECTIONS_AT_ISSUE\n"
        "NONE\n"
    )


def _novelty_rejected_text(reason: str) -> str:
    return (
        "## NOVELTY_VERDICT\n"
        "NOVELTY_REJECTED\n\n"
        "## REJECTION_REASON\n"
        f"{reason}\n\n"
        "## SECTIONS_AT_ISSUE\n"
        "NONE\n"
    )


def _compatibility_accepted_text() -> str:
    return (
        "## COMPATIBILITY_VERDICT\n"
        "COMPATIBILITY_ACCEPTED\n\n"
        "## REJECTION_REASON\n"
        "N/A\n"
    )


def _compatibility_rejected_text(reason: str, sections: str = "NONE") -> str:
    del sections
    return (
        "## COMPATIBILITY_VERDICT\n"
        "COMPATIBILITY_REJECTED\n\n"
        "## REJECTION_REASON\n"
        f"{reason}\n"
    )


def _llm_response(
    stage: str,
    text: str,
    *,
    content: str | None = None,
    thinking: str | None = None,
) -> LlmResponse:
    raw_response = {"stage": stage}
    if content is not None or thinking is not None:
        raw_response["message"] = {
            "content": content or "",
            "thinking": thinking or "",
        }
    return LlmResponse(
        text=text,
        route_id="mock",
        provider="mock",
        provider_model_id="mock-model",
        raw_request={"stage": stage},
        raw_response=raw_response,
        usage={},
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
    )


def test_generator_loads_only_current_prompt_bundle_assets(monkeypatch) -> None:
    original_read_text = Path.read_text
    seen_paths: list[Path] = []

    def guarded_read_text(self: Path, *args, **kwargs):  # type: ignore[override]
        seen_paths.append(self)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    generator = CandidateGenerator(_cfg())

    observed = {
        "/".join(path.parts[path.parts.index("prompts") + 1 :])
        for path in seen_paths
        if "prompts" in path.parts
    }
    assert observed == {
        "shared/project_context.txt",
        "shared/genome_schema.txt",
        "shared/template.txt",
        "seed/system.txt",
        "seed/user.txt",
        "compatibility/seed/system.txt",
        "compatibility/seed/user.txt",
        "mutation/system.txt",
        "mutation/user.txt",
        "novelty/mutation/system.txt",
        "novelty/mutation/user.txt",
        "compatibility/mutation/system.txt",
        "compatibility/mutation/user.txt",
        "crossover/system.txt",
        "crossover/user.txt",
        "novelty/crossover/system.txt",
        "novelty/crossover/user.txt",
        "compatibility/crossover/system.txt",
        "compatibility/crossover/user.txt",
        "implementation/system.txt",
        "implementation/user.txt",
        "repair/system.txt",
        "repair/user.txt",
    }
    assert "## CORE_GENES" in generator.prompt_bundle.seed_system
    generator.close()


def test_canonical_generator_seeds_real_island_organism(tmp_path: Path) -> None:
    generator = CandidateGenerator(_cfg())
    island = Island(
        island_id="gradient_methods",
        name="gradient methods",
        description_path=str(tmp_path / "gradient_methods.txt"),
        description_text="First-order optimization heuristics.",
    )
    organism_dir = tmp_path / "org_seed"
    organism_dir.mkdir(parents=True, exist_ok=True)

    try:
        organism = generator.generate_seed_organism(
            island=island,
            organism_id="seed01",
            generation=0,
            organism_dir=organism_dir,
        )
    finally:
        generator.close()

    assert organism.operator == "seed"
    assert organism.island_id == "gradient_methods"
    assert organism.implementation_path == str(organism_dir / "implementation.py")
    assert (organism_dir / "implementation.py").exists()
    assert (organism_dir / "genetic_code.md").exists()
    assert (organism_dir / "lineage.json").exists()
    assert (organism_dir / "organism.json").exists()
    assert organism.llm_route_id == "mock"
    assert organism.llm_provider == "mock"
    assert organism.provider_model_id == "mock-model"
    organism_meta = json.loads((organism_dir / "organism.json").read_text(encoding="utf-8"))
    assert organism_meta["llm_route_id"] == "mock"
    assert organism_meta["llm_provider"] == "mock"
    assert organism_meta["provider_model_id"] == "mock-model"
    assert "design" in (organism_dir / "llm_request.json").read_text(encoding="utf-8")
    assert "implementation" in (organism_dir / "llm_request.json").read_text(encoding="utf-8")
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    llm_response = json.loads((organism_dir / "llm_response.json").read_text(encoding="utf-8"))
    assert llm_request["route_id"] == "mock"
    assert llm_request["design"]["route_id"] == "mock"
    assert llm_request["implementation"]["route_id"] == "mock"
    assert len(llm_request["design_attempts"]) == 1
    assert len(llm_request["compatibility_checks"]) == 1
    assert "novelty_checks" not in llm_request
    assert llm_response["route_id"] == "mock"
    assert llm_response["design"]["route_id"] == "mock"
    assert llm_response["implementation"]["route_id"] == "mock"


def test_canonical_seed_operator_module_excludes_prompt_only_mutation_and_crossover() -> None:
    assert hasattr(canonical_seed_operators, "SeedOperator")
    assert not hasattr(canonical_seed_operators, "MutationOperator")
    assert not hasattr(canonical_seed_operators, "CrossoverOperator")


def test_run_creation_stages_persists_design_exchange_before_parse_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_partial"
    organism_dir.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []

    def fake_generate(request):
        calls.append(request.stage)
        return LlmResponse(
            text=(
                "## CORE_GENES\n"
                "- valid enough gene one\n"
                "- valid enough gene two\n"
                "- valid enough gene three\n\n"
                "## INTERACTION_NOTES\n"
                "Partial design only.\n"
            ),
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": request.stage},
            raw_response={"stage": request.stage},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)

    try:
        with pytest.raises(ValueError, match="COMPUTE_NOTES|CHANGE_DESCRIPTION"):
            generator.run_creation_stages(
                design_system_prompt="design system",
                design_user_prompt="design user",
                org_dir=organism_dir,
                organism_id="org_partial",
                generation=0,
            )
    finally:
        generator.close()

    assert calls == ["design"]
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    llm_response = json.loads((organism_dir / "llm_response.json").read_text(encoding="utf-8"))
    assert llm_request["design"]["request"] == {"stage": "design"}
    assert llm_request["design"]["status"] == "completed"
    assert llm_response["design"]["response"] == {"stage": "design"}
    assert llm_response["design"]["status"] == "completed"
    assert "implementation" not in llm_request
    assert "implementation" not in llm_response


def test_run_creation_stages_rejects_raw_design_with_extra_top_level_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_extra_section"
    organism_dir.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []

    def fake_generate(request):
        calls.append(request.stage)
        return LlmResponse(
            text=_design_text("Candidate with extra section.") + "\n## DEBUG\nThis must fail.\n",
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": request.stage},
            raw_response={"stage": request.stage},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)

    try:
        with pytest.raises(ValueError, match="unexpected section"):
            generator.run_creation_stages(
                design_system_prompt="design system",
                design_user_prompt="design user",
                org_dir=organism_dir,
                organism_id="org_extra_section",
                generation=0,
            )
    finally:
        generator.close()

    assert calls == ["design"]
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    assert "implementation" not in llm_request


def test_run_creation_stages_persists_raw_implementation_exchange_before_extract_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_impl_failure"
    organism_dir.mkdir(parents=True, exist_ok=True)

    def fake_generate(request):
        if request.stage == "design":
            text = (
                "## CORE_GENES\n"
                "- candidate gene one\n"
                "- candidate gene two\n"
                "- candidate gene three\n\n"
                "## INTERACTION_NOTES\n"
                "Candidate.\n\n"
                "## COMPUTE_NOTES\n"
                "Cheap.\n\n"
                "## CHANGE_DESCRIPTION\n"
                "Candidate change.\n"
            )
        else:
            pieces = ["## COMPILATION_MODE", "FULL"]
            for region in OPTIMIZER_REGIONS:
                body = "        def broken(\n" if region == "STATE_REPRESENTATION" else f"        # body for {region}\n"
                pieces.extend(("", f"## REGION {region}", body.rstrip("\n"), "## END_REGION"))
            text = "\n".join(pieces) + "\n"
        return LlmResponse(
            text=text,
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": request.stage},
            raw_response={"stage": request.stage},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)

    try:
        with pytest.raises(ValueError, match="syntactically invalid Python"):
            generator.run_creation_stages(
                design_system_prompt="design system",
                design_user_prompt="design user",
                org_dir=organism_dir,
                organism_id="org_impl_failure",
                generation=0,
            )
    finally:
        generator.close()

    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    llm_response = json.loads((organism_dir / "llm_response.json").read_text(encoding="utf-8"))
    assert llm_request["implementation"]["request"] == {"stage": "implementation"}
    assert llm_request["implementation"]["status"] == "failed"
    assert "syntactically invalid Python" in llm_request["implementation"]["error_msg"]
    assert llm_response["implementation"]["response"] == {"stage": "implementation"}
    assert "## REGION STATE_REPRESENTATION" in llm_response["implementation"]["text"]
    assert llm_response["implementation"]["status"] == "failed"
    assert "syntactically invalid Python" in llm_response["implementation"]["error_msg"]


def test_circle_seed_implementation_stage_uses_full_source_compilation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_circle_cfg())
    organism_dir = tmp_path / "circle_seed_full"
    organism_dir.mkdir(parents=True, exist_ok=True)
    stages: list[str] = []

    def fake_generate(request):
        stages.append(request.stage)
        if request.stage == "design":
            return _llm_response(request.stage, _circle_design_text())
        assert request.stage == "implementation"
        assert "Single rewrite contract:" in request.system_prompt
        assert "=== CHANGED_SECTIONS ===\n" + "\n".join(CIRCLE_IMPLEMENTATION_REGIONS) in request.user_prompt
        assert "=== MATERNAL BASE GENETIC CODE ===\nNONE" in request.user_prompt
        assert "Return the final full Python file only." in request.user_prompt
        return _llm_response(request.stage, _circle_base_source())

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    try:
        result = generator.run_creation_stages(
            design_system_prompt="design system",
            design_user_prompt="design user",
            org_dir=organism_dir,
            organism_id="circle_seed_full",
            generation=0,
        )
    finally:
        generator.close()

    assert stages == ["design", "implementation"]
    assert "## COMPILATION_MODE" not in result.implementation_code
    assert "# EVOLVE-BLOCK-START" in result.implementation_code
    assert "# SECTION: INIT_GEOMETRY" in result.implementation_code
    assert "radii = np.full(26, 0.04, dtype=float)" in result.implementation_code
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    llm_response = json.loads((organism_dir / "llm_response.json").read_text(encoding="utf-8"))
    assert llm_request["implementation"]["implementation_strategy"] == "full_source_rewrite"
    assert "compilation_mode" not in llm_request["implementation"]
    assert llm_request["implementation"]["changed_sections"] == list(CIRCLE_IMPLEMENTATION_REGIONS)
    assert llm_response["implementation"]["text"] == result.implementation_code


def test_circle_full_source_extraction_accepts_markdown_fenced_python(
    tmp_path: Path,
) -> None:
    del tmp_path
    generator = CandidateGenerator(_circle_cfg())
    prepared = generator._prepare_implementation_stage(  # noqa: SLF001
        parse_llm_response(_circle_design_text()),
    )

    try:
        extracted = generator._extract_implementation_stage_code(  # noqa: SLF001
            f"```python\n{_circle_base_source()}\n```",
            prepared=prepared,
        )
    finally:
        generator.close()

    assert extracted == _circle_base_source()


def test_circle_full_source_extraction_strips_legacy_full_mode_preamble(
    tmp_path: Path,
) -> None:
    del tmp_path
    generator = CandidateGenerator(_circle_cfg())
    prepared = generator._prepare_implementation_stage(  # noqa: SLF001
        parse_llm_response(_circle_design_text()),
    )

    try:
        extracted = generator._extract_implementation_stage_code(  # noqa: SLF001
            "## COMPILATION_MODE\nFULL\n\n" + _circle_base_source(),
            prepared=prepared,
        )
    finally:
        generator.close()

    assert extracted == _circle_base_source()


def test_circle_full_source_extraction_rejects_full_patch_artifact(
    tmp_path: Path,
) -> None:
    generator = CandidateGenerator(_circle_cfg())
    prepared = generator._prepare_implementation_stage(  # noqa: SLF001
        parse_llm_response(_circle_design_text()),
    )

    try:
        with pytest.raises(ValueError, match="syntactically invalid Python"):
            generator._extract_implementation_stage_code(  # noqa: SLF001
                _circle_patch_response("FULL", CIRCLE_IMPLEMENTATION_REGIONS),
                prepared=prepared,
            )
    finally:
        generator.close()


def test_circle_mutation_implementation_stage_uses_parent_full_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_circle_cfg())
    organism_dir = tmp_path / "circle_mutation_full_rewrite"
    organism_dir.mkdir(parents=True, exist_ok=True)
    parent = _make_circle_parent(tmp_path)
    base_source = Path(parent.implementation_path).read_text(encoding="utf-8")
    child_source = base_source.replace(
        "radii = np.full(26, 0.04, dtype=float)",
        "radii = np.full(26, 0.05, dtype=float)",
    )

    def fake_generate(request):
        if request.stage == "design":
            return _llm_response(
                request.stage,
                _circle_design_text(radius_policy="Use role-dependent non-uniform radii."),
            )
        assert request.stage == "implementation"
        assert "=== CHANGED_SECTIONS ===\nRADIUS_POLICY" in request.user_prompt
        assert "=== MATERNAL BASE IMPLEMENTATION ===\n" + base_source in request.user_prompt
        assert "rewrite the complete child file" in request.user_prompt
        return _llm_response(request.stage, child_source)

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    try:
        result = generator.run_creation_stages(
            design_system_prompt="design system",
            design_user_prompt="design user",
            org_dir=organism_dir,
            organism_id="circle_mutation_full_rewrite",
            generation=1,
            implementation_base_parent=parent,
        )
    finally:
        generator.close()

    assert result.implementation_code == child_source
    assert "radii = np.full(26, 0.05, dtype=float)" in result.implementation_code
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    llm_response = json.loads((organism_dir / "llm_response.json").read_text(encoding="utf-8"))
    assert llm_request["implementation"]["implementation_strategy"] == "full_source_rewrite"
    assert "compilation_mode" not in llm_request["implementation"]
    assert llm_request["implementation"]["changed_sections"] == ["RADIUS_POLICY"]
    assert llm_response["implementation"]["text"] == result.implementation_code


def test_circle_parent_full_rewrite_accepts_non_scaffold_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_circle_cfg())
    organism_dir = tmp_path / "circle_parent_full_non_scaffold"
    organism_dir.mkdir(parents=True, exist_ok=True)
    parent = _make_circle_parent(
        tmp_path,
        implementation_source="import numpy as np\n\ndef run_packing():\n    return None\n",
    )
    stages: list[str] = []
    rewritten_source = (
        "import numpy as np\n\n"
        "def run_packing():\n"
        "    centers = np.asarray([[0.0, 0.0]], dtype=float)\n"
        "    radii = np.asarray([0.05], dtype=float)\n"
        "    reported_sum = float(np.sum(radii))\n"
        "    return centers, radii, reported_sum\n"
    )

    def fake_generate(request):
        stages.append(request.stage)
        if request.stage == "design":
            return _llm_response(
                request.stage,
                _circle_design_text(radius_policy="Use role-dependent non-uniform radii."),
            )
        assert request.stage == "implementation"
        return _llm_response(request.stage, rewritten_source)

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    try:
        result = generator.run_creation_stages(
            design_system_prompt="design system",
            design_user_prompt="design user",
            org_dir=organism_dir,
            organism_id="circle_parent_full_non_scaffold",
            generation=1,
            implementation_base_parent=parent,
        )
    finally:
        generator.close()

    assert stages == ["design", "implementation"]
    assert result.implementation_code == rewritten_source
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    assert llm_request["implementation"]["implementation_strategy"] == "full_source_rewrite"
    assert "compilation_mode" not in llm_request["implementation"]
    assert llm_request["implementation"]["status"] == "completed"


def test_migrated_family_malformed_scaffold_fails_instead_of_legacy_fallback(
    tmp_path: Path,
) -> None:
    malformed_template = tmp_path / "malformed_template.txt"
    malformed_template.write_text("# no section markers here\n", encoding="utf-8")
    cfg = _circle_cfg()
    cfg.evolver.prompts.implementation_template = str(malformed_template)
    generator = CandidateGenerator(cfg)

    try:
        with pytest.raises(ValueError, match="Section-aware implementation scaffold is invalid"):
            generator.uses_section_patch_compilation()
    finally:
        generator.close()


def test_migrated_family_missing_mode_prompt_contract_fails_instead_of_legacy_fallback(
    tmp_path: Path,
) -> None:
    old_system = tmp_path / "old_implementation_system.txt"
    old_system.write_text("Return ONLY valid Python source code.\n", encoding="utf-8")
    cfg = _circle_cfg()
    cfg.evolver.prompts.implementation_system = str(old_system)
    generator = CandidateGenerator(cfg)

    try:
        with pytest.raises(ValueError, match="supported implementation contract"):
            generator.uses_section_patch_compilation()
    finally:
        generator.close()


def test_non_migrated_family_can_still_report_legacy_implementation_mode(
    tmp_path: Path,
) -> None:
    old_system = tmp_path / "old_implementation_system.txt"
    old_user = tmp_path / "old_implementation_user.txt"
    old_template = tmp_path / "old_implementation_template.txt"
    old_system.write_text("Return ONLY valid Python source code.\n", encoding="utf-8")
    old_user.write_text("{organism_genetic_code}\n{change_description}\n{implementation_template}\n", encoding="utf-8")
    old_template.write_text("# old template\n", encoding="utf-8")
    cfg = _circle_cfg()
    del cfg.evolver.prompts.genome_schema
    cfg.evolver.prompts.implementation_system = str(old_system)
    cfg.evolver.prompts.implementation_user = str(old_user)
    cfg.evolver.prompts.implementation_template = str(old_template)
    generator = CandidateGenerator(cfg)

    try:
        assert generator.expected_core_gene_sections is None
        assert generator.uses_section_patch_compilation() is False
    finally:
        generator.close()


def test_migrated_family_missing_scaffold_fails_during_prompt_loading(tmp_path: Path) -> None:
    cfg = _circle_cfg()
    cfg.evolver.prompts.implementation_template = str(tmp_path / "missing_template.txt")

    with pytest.raises(FileNotFoundError, match="implementation_template"):
        CandidateGenerator(cfg)


def test_circle_parent_rewrite_requires_maternal_base_files(tmp_path: Path) -> None:
    generator = CandidateGenerator(_circle_cfg())
    parent = _make_circle_parent(tmp_path)
    Path(parent.implementation_path).unlink()

    try:
        with pytest.raises(FileNotFoundError, match="Maternal base implementation"):
            generator._prepare_implementation_stage(  # noqa: SLF001
                parse_llm_response(_circle_design_text(radius_policy="Use role-dependent non-uniform radii.")),
                implementation_base_parent=parent,
            )
    finally:
        generator.close()


def test_parent_full_rewrite_extraction_rejects_patch_artifact(tmp_path: Path) -> None:
    generator = CandidateGenerator(_circle_cfg())
    parent = _make_circle_parent(tmp_path)
    prepared = generator._prepare_implementation_stage(  # noqa: SLF001
        parse_llm_response(_circle_design_text(radius_policy="Use role-dependent non-uniform radii.")),
        implementation_base_parent=parent,
    )
    response_text = _circle_patch_response("PATCH", ("RADIUS_POLICY",), patched=True)

    try:
        with pytest.raises(ValueError, match="syntactically invalid Python"):
            generator._extract_implementation_stage_code(response_text, prepared=prepared)  # noqa: SLF001
    finally:
        generator.close()


def test_run_creation_stages_persists_pending_design_exchange_before_transport_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_transport_failure"
    organism_dir.mkdir(parents=True, exist_ok=True)

    def fake_generate(_request):
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(generator.registry, "generate", fake_generate)

    try:
        with pytest.raises(RuntimeError, match="broker unavailable"):
            generator.run_creation_stages(
                design_system_prompt="design system",
                design_user_prompt="design user",
                org_dir=organism_dir,
                organism_id="org_transport_failure",
                generation=0,
            )
    finally:
        generator.close()

    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    llm_response = json.loads((organism_dir / "llm_response.json").read_text(encoding="utf-8"))
    assert llm_request["design"]["system_prompt"] == "design system"
    assert llm_request["design"]["user_prompt"] == "design user"
    assert llm_request["design"]["request"] is None
    assert llm_request["design"]["status"] == "failed"
    assert "broker unavailable" in llm_request["design"]["error_msg"]
    assert llm_response["design"]["response"] is None
    assert llm_response["design"]["text"] is None
    assert llm_response["design"]["status"] == "failed"
    assert "broker unavailable" in llm_response["design"]["error_msg"]


def test_repair_stage_persists_raw_exchange_before_extract_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism = _make_repairable_organism(tmp_path)

    def fake_generate(_request):
        return LlmResponse(
            text="def broken(\n",
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": "repair"},
            raw_response={"stage": "repair"},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)

    try:
        with pytest.raises(ValueError, match="syntactically invalid Python"):
            generator.repair_organism_after_error(
                organism=organism,
                phase="simple",
                experiment_name="simple_a",
                errors=[
                    {
                        "attempt": 1,
                        "status": "failed",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "error_msg": "name 'np' is not defined",
                    }
                ],
            )
    finally:
        generator.close()

    llm_request = json.loads((Path(organism.organism_dir) / "llm_request.json").read_text(encoding="utf-8"))
    llm_response = json.loads((Path(organism.organism_dir) / "llm_response.json").read_text(encoding="utf-8"))
    assert llm_request["repair_attempts"][0]["request"] == {"stage": "repair"}
    assert llm_request["repair_attempts"][0]["status"] == "failed"
    assert "syntactically invalid Python" in llm_request["repair_attempts"][0]["error_msg"]
    assert llm_response["repair_attempts"][0]["response"] == {"stage": "repair"}
    assert llm_response["repair_attempts"][0]["text"] == "def broken(\n"
    assert llm_response["repair_attempts"][0]["status"] == "failed"
    assert "syntactically invalid Python" in llm_response["repair_attempts"][0]["error_msg"]


def test_run_creation_stages_with_novelty_retries_design_before_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_novelty_retry"
    organism_dir.mkdir(parents=True, exist_ok=True)
    calls: list[tuple[str, str]] = []
    design_responses = iter(
        [
            (
                "## CORE_GENES\n"
                "- first candidate gene one\n"
                "- first candidate gene two\n"
                "- first candidate gene three\n\n"
                "## INTERACTION_NOTES\n"
                "First candidate.\n\n"
                "## COMPUTE_NOTES\n"
                "Cheap.\n\n"
                "## CHANGE_DESCRIPTION\n"
                "First candidate change.\n"
            ),
            (
                "## CORE_GENES\n"
                "- second candidate gene one\n"
                "- second candidate gene two\n"
                "- second candidate gene three\n\n"
                "## INTERACTION_NOTES\n"
                "Second candidate.\n\n"
                "## COMPUTE_NOTES\n"
                "Cheap.\n\n"
                "## CHANGE_DESCRIPTION\n"
                "Second candidate change.\n"
            ),
        ]
    )
    novelty_responses = iter(
        [
            (
                "## NOVELTY_VERDICT\nNOVELTY_REJECTED\n\n"
                "## REJECTION_REASON\nToo close to the parent.\n\n"
                "## SECTIONS_AT_ISSUE\nNONE\n"
            ),
            (
                "## NOVELTY_VERDICT\nNOVELTY_ACCEPTED\n\n"
                "## REJECTION_REASON\nN/A\n\n"
                "## SECTIONS_AT_ISSUE\nNONE\n"
            ),
        ]
    )

    def fake_generate(request):
        calls.append((request.stage, request.user_prompt))
        if request.stage == "design":
            text = next(design_responses)
        elif request.stage == "novelty_check":
            text = next(novelty_responses)
        else:
            text = _implementation_text()
        return LlmResponse(
            text=text,
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": request.stage},
            raw_response={"stage": request.stage},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    novelty_context = NoveltyCheckContext(
        operator="mutation",
        build_design_prompts=lambda feedback: ("design system retry", f"retry prompt :: {feedback[-1]}"),
        build_novelty_prompts=lambda _parsed: ("novelty system", "novelty user"),
    )

    try:
        result = generator.run_creation_stages(
            design_system_prompt="design system",
            design_user_prompt="design user",
            novelty_context=novelty_context,
            org_dir=organism_dir,
            organism_id="org_novelty_retry",
            generation=0,
        )
    finally:
        generator.close()

    assert [stage for stage, _ in calls] == [
        "design",
        "novelty_check",
        "design",
        "novelty_check",
        "implementation",
    ]
    assert "retry prompt :: Too close to the parent." in calls[2][1]
    assert result.parsed_design["CHANGE_DESCRIPTION"] == "Second candidate change."
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    llm_response = json.loads((organism_dir / "llm_response.json").read_text(encoding="utf-8"))
    assert len(llm_request["design_attempts"]) == 2
    assert len(llm_request["novelty_checks"]) == 2
    assert llm_response["novelty_checks"][0]["rejection_reason"] == "Too close to the parent."
    assert llm_request["design"]["user_prompt"] == "retry prompt :: Too close to the parent."


def test_run_creation_stages_with_novelty_rejection_exhaustion_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_novelty_exhausted"
    organism_dir.mkdir(parents=True, exist_ok=True)

    def fake_generate(request):
        if request.stage == "design":
            text = (
                "## CORE_GENES\n"
                "- first candidate gene one\n"
                "- first candidate gene two\n"
                "- first candidate gene three\n\n"
                "## INTERACTION_NOTES\n"
                "Candidate.\n\n"
                "## COMPUTE_NOTES\n"
                "Cheap.\n\n"
                "## CHANGE_DESCRIPTION\n"
                "Candidate change.\n"
            )
        else:
            text = (
                "## NOVELTY_VERDICT\n"
                "NOVELTY_REJECTED\n\n"
                "## REJECTION_REASON\n"
                "Still paraphrases the parent.\n\n"
                "## SECTIONS_AT_ISSUE\n"
                "NONE\n"
            )
        return LlmResponse(
            text=text,
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": request.stage},
            raw_response={"stage": request.stage},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    novelty_context = NoveltyCheckContext(
        operator="mutation",
        build_design_prompts=lambda feedback: ("design retry", f"retry :: {feedback[-1]}"),
        build_novelty_prompts=lambda _parsed: ("novelty system", "novelty user"),
    )

    try:
        with pytest.raises(NoveltyRejectionExhaustedError, match="Novelty validation rejected organism"):
            generator.run_creation_stages_with_retries(
                design_system_prompt="design system",
                design_user_prompt="design user",
                novelty_context=novelty_context,
                org_dir=organism_dir,
                organism_id="org_novelty_exhausted",
                generation=0,
            )
    finally:
        generator.close()

    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    assert len(llm_request["design_attempts"]) == 2
    assert len(llm_request["novelty_checks"]) == 2
    assert "implementation" not in llm_request


def test_run_creation_stages_with_novelty_parse_failure_retries_full_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_novelty_parse_retry"
    organism_dir.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []
    novelty_texts = iter(
        [
            "## NOVELTY_VERDICT\nMAYBE\n",
            "## NOVELTY_VERDICT\nNOVELTY_ACCEPTED\n\n## REJECTION_REASON\nN/A\n\n## SECTIONS_AT_ISSUE\nNONE\n",
        ]
    )

    def fake_generate(request):
        calls.append(request.stage)
        if request.stage == "design":
            text = (
                "## CORE_GENES\n"
                "- candidate gene one\n"
                "- candidate gene two\n"
                "- candidate gene three\n\n"
                "## INTERACTION_NOTES\n"
                "Candidate.\n\n"
                "## COMPUTE_NOTES\n"
                "Cheap.\n\n"
                "## CHANGE_DESCRIPTION\n"
                "Candidate change.\n"
            )
        elif request.stage == "novelty_check":
            text = next(novelty_texts)
        else:
            text = _implementation_text()
        return LlmResponse(
            text=text,
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": request.stage},
            raw_response={"stage": request.stage},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    novelty_context = NoveltyCheckContext(
        operator="mutation",
        build_design_prompts=lambda feedback: ("design retry", f"retry :: {feedback[-1]}"),
        build_novelty_prompts=lambda _parsed: ("novelty system", "novelty user"),
    )

    try:
        generator.run_creation_stages_with_retries(
            design_system_prompt="design system",
            design_user_prompt="design user",
            novelty_context=novelty_context,
            org_dir=organism_dir,
            organism_id="org_novelty_parse_retry",
            generation=0,
        )
    finally:
        generator.close()

    assert calls == ["design", "novelty_check", "design", "novelty_check", "implementation"]


def test_run_creation_stages_seed_runs_compatibility_before_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_seed_compatibility"
    organism_dir.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []
    pipeline_states: list[str] = []

    def fake_generate(request):
        calls.append(request.stage)
        if request.stage == "design":
            text = _design_text("Seed candidate.")
        elif request.stage == "compatibility_check":
            text = _compatibility_accepted_text()
        else:
            text = _implementation_text()
        return LlmResponse(
            text=text,
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": request.stage},
            raw_response={"stage": request.stage},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    compatibility_context = CompatibilityValidationContext(
        check=CompatibilityCheckContext(operator_kind="seed"),
        build_design_prompts=lambda _novelty_feedback, compatibility_feedback: (
            "design retry",
            format_compatibility_rejection_feedback(compatibility_feedback),
        ),
        build_compatibility_prompts=lambda _parsed: ("compatibility system", "compatibility user"),
    )

    try:
        result = generator.run_creation_stages(
            design_system_prompt="design system",
            design_user_prompt="design user",
            compatibility_context=compatibility_context,
            org_dir=organism_dir,
            organism_id="org_seed_compatibility",
            generation=0,
            pipeline_state_callback=pipeline_states.append,
        )
    finally:
        generator.close()

    assert calls == ["design", "compatibility_check", "implementation"]
    assert "compatibility_check" in pipeline_states
    assert result.parsed_design["CHANGE_DESCRIPTION"] == "Seed candidate."
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    assert "novelty_checks" not in llm_request
    assert len(llm_request["compatibility_checks"]) == 1


def test_structured_stages_parse_raw_content_before_thinking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_structured_content"
    organism_dir.mkdir(parents=True, exist_ok=True)
    dirty_compatibility_text = "reasoning before final answer\n\n" + _compatibility_accepted_text()

    def fake_generate(request):
        if request.stage == "design":
            return _llm_response("design", _design_text("Seed candidate."))
        if request.stage == "compatibility_check":
            return _llm_response(
                "compatibility_check",
                dirty_compatibility_text,
                content=_compatibility_accepted_text(),
                thinking="reasoning before final answer",
            )
        return _llm_response("implementation", _implementation_text())

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    compatibility_context = CompatibilityValidationContext(
        check=CompatibilityCheckContext(operator_kind="seed"),
        build_design_prompts=lambda _novelty_feedback, compatibility_feedback: (
            "design retry",
            format_compatibility_rejection_feedback(compatibility_feedback),
        ),
        build_compatibility_prompts=lambda _parsed: ("compatibility system", "compatibility user"),
    )

    try:
        result = generator.run_creation_stages(
            design_system_prompt="design system",
            design_user_prompt="design user",
            compatibility_context=compatibility_context,
            org_dir=organism_dir,
            organism_id="org_structured_content",
            generation=0,
        )
    finally:
        generator.close()

    assert result.parsed_design["CHANGE_DESCRIPTION"] == "Seed candidate."
    llm_response = json.loads((organism_dir / "llm_response.json").read_text(encoding="utf-8"))
    compatibility_entry = llm_response["compatibility_checks"][0]
    assert compatibility_entry["text"] == dirty_compatibility_text
    assert compatibility_entry["content_text"] == _compatibility_accepted_text().strip()
    assert compatibility_entry["thinking_text"] == "reasoning before final answer"
    assert compatibility_entry["parse_text"] == _compatibility_accepted_text().strip()
    assert compatibility_entry["parse_source"] == "message.content"
    assert compatibility_entry["verdict"] == "COMPATIBILITY_ACCEPTED"


def test_implementation_patch_parses_raw_content_before_thinking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_implementation_content"
    organism_dir.mkdir(parents=True, exist_ok=True)
    dirty_implementation_text = "analysis that would break strict parsing\n\n" + _implementation_text()

    def fake_generate(request):
        if request.stage == "design":
            return _llm_response("design", _design_text("Seed candidate."))
        return _llm_response(
            "implementation",
            dirty_implementation_text,
            content=_implementation_text(),
            thinking="analysis that would break strict parsing",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)

    try:
        result = generator.run_creation_stages(
            design_system_prompt="design system",
            design_user_prompt="design user",
            org_dir=organism_dir,
            organism_id="org_implementation_content",
            generation=0,
        )
    finally:
        generator.close()

    assert "# implementation body for STATE_REPRESENTATION" in result.implementation_code
    llm_response = json.loads((organism_dir / "llm_response.json").read_text(encoding="utf-8"))
    implementation_entry = llm_response["implementation"]
    assert implementation_entry["text"] == result.implementation_code
    assert implementation_entry["content_text"] == _implementation_text().strip()
    assert implementation_entry["thinking_text"] == "analysis that would break strict parsing"
    assert implementation_entry["parse_text"] == _implementation_text().strip()
    assert implementation_entry["parse_source"] == "message.content"


def test_structured_parse_failure_records_content_thinking_and_parse_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    generator.evolver_cfg.creation.max_attempts_to_regenerate_organism_after_compatibility_rejection = 0
    organism_dir = tmp_path / "org_structured_parse_failure"
    organism_dir.mkdir(parents=True, exist_ok=True)

    def fake_generate(request):
        if request.stage == "design":
            return _llm_response("design", _design_text("Seed candidate."))
        return _llm_response(
            "compatibility_check",
            "thinking\n\nCOMPATIBILITY_ACCEPTED\nN/A",
            content="COMPATIBILITY_ACCEPTED\nN/A",
            thinking="thinking",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    compatibility_context = CompatibilityValidationContext(
        check=CompatibilityCheckContext(operator_kind="seed"),
        build_design_prompts=lambda _novelty_feedback, compatibility_feedback: (
            "design retry",
            format_compatibility_rejection_feedback(compatibility_feedback),
        ),
        build_compatibility_prompts=lambda _parsed: ("compatibility system", "compatibility user"),
    )

    try:
        with pytest.raises(ValueError, match="text before the first section|exactly these sections"):
            generator.run_creation_stages(
                design_system_prompt="design system",
                design_user_prompt="design user",
                compatibility_context=compatibility_context,
                org_dir=organism_dir,
                organism_id="org_structured_parse_failure",
                generation=0,
            )
    finally:
        generator.close()

    llm_response = json.loads((organism_dir / "llm_response.json").read_text(encoding="utf-8"))
    compatibility_entry = llm_response["compatibility_checks"][0]
    assert compatibility_entry["content_text"] == "COMPATIBILITY_ACCEPTED\nN/A"
    assert compatibility_entry["thinking_text"] == "thinking"
    assert compatibility_entry["parse_text"] == "COMPATIBILITY_ACCEPTED\nN/A"
    assert compatibility_entry["error_kind"] == "compatibility_judgment_parse_failed"


def test_run_creation_stages_mutation_runs_novelty_before_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_mutation_validation"
    organism_dir.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []
    pipeline_states: list[str] = []

    def fake_generate(request):
        calls.append(request.stage)
        if request.stage == "design":
            text = _design_text("Mutation candidate.")
        elif request.stage == "novelty_check":
            text = _novelty_accepted_text()
        elif request.stage == "compatibility_check":
            text = _compatibility_accepted_text()
        else:
            text = _implementation_text()
        return LlmResponse(
            text=text,
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": request.stage},
            raw_response={"stage": request.stage},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    novelty_context = NoveltyCheckContext(
        operator="mutation",
        build_design_prompts=lambda feedback: ("design retry", f"novelty :: {feedback[-1]}"),
        build_novelty_prompts=lambda _parsed: ("novelty system", "novelty user"),
    )
    compatibility_context = CompatibilityValidationContext(
        check=CompatibilityCheckContext(operator_kind="mutation"),
        build_design_prompts=lambda novelty_feedback, compatibility_feedback: (
            "design retry",
            "\n".join(novelty_feedback)
            + "\n"
            + format_compatibility_rejection_feedback(compatibility_feedback),
        ),
        build_compatibility_prompts=lambda _parsed: ("compatibility system", "compatibility user"),
    )

    try:
        generator.run_creation_stages(
            design_system_prompt="design system",
            design_user_prompt="design user",
            novelty_context=novelty_context,
            compatibility_context=compatibility_context,
            org_dir=organism_dir,
            organism_id="org_mutation_validation",
            generation=0,
            pipeline_state_callback=pipeline_states.append,
        )
    finally:
        generator.close()

    assert calls == ["design", "novelty_check", "compatibility_check", "implementation"]
    assert "compatibility_check" in pipeline_states


def test_run_creation_stages_crossover_runs_novelty_before_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_crossover_validation"
    organism_dir.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []
    pipeline_states: list[str] = []

    def fake_generate(request):
        calls.append(request.stage)
        if request.stage == "design":
            text = _design_text("Crossover candidate.")
        elif request.stage == "novelty_check":
            text = _novelty_accepted_text()
        elif request.stage == "compatibility_check":
            text = _compatibility_accepted_text()
        else:
            text = _implementation_text()
        return LlmResponse(
            text=text,
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": request.stage},
            raw_response={"stage": request.stage},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    novelty_context = NoveltyCheckContext(
        operator="crossover",
        build_design_prompts=lambda feedback: ("design retry", f"novelty :: {feedback[-1]}"),
        build_novelty_prompts=lambda _parsed: ("novelty system", "novelty user"),
    )
    compatibility_context = CompatibilityValidationContext(
        check=CompatibilityCheckContext(operator_kind="crossover"),
        build_design_prompts=lambda novelty_feedback, compatibility_feedback: (
            "design retry",
            "\n".join(novelty_feedback)
            + "\n"
            + format_compatibility_rejection_feedback(compatibility_feedback),
        ),
        build_compatibility_prompts=lambda _parsed: ("compatibility system", "compatibility user"),
    )

    try:
        generator.run_creation_stages(
            design_system_prompt="design system",
            design_user_prompt="design user",
            novelty_context=novelty_context,
            compatibility_context=compatibility_context,
            org_dir=organism_dir,
            organism_id="org_crossover_validation",
            generation=0,
            pipeline_state_callback=pipeline_states.append,
        )
    finally:
        generator.close()

    assert calls == ["design", "novelty_check", "compatibility_check", "implementation"]
    assert "compatibility_check" in pipeline_states


def test_run_creation_stages_skips_compatibility_when_novelty_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    generator.evolver_cfg.creation.max_attempts_to_regenerate_organism_after_novelty_rejection = 0
    organism_dir = tmp_path / "org_novelty_blocks_compatibility"
    organism_dir.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []

    def fake_generate(request):
        calls.append(request.stage)
        if request.stage == "design":
            text = _design_text("Rejected candidate.")
        else:
            text = _novelty_rejected_text("Too close to the parent.")
        return LlmResponse(
            text=text,
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": request.stage},
            raw_response={"stage": request.stage},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    novelty_context = NoveltyCheckContext(
        operator="mutation",
        build_design_prompts=lambda feedback: ("design retry", f"novelty :: {feedback[-1]}"),
        build_novelty_prompts=lambda _parsed: ("novelty system", "novelty user"),
    )
    compatibility_context = CompatibilityValidationContext(
        check=CompatibilityCheckContext(operator_kind="mutation"),
        build_design_prompts=lambda _novelty_feedback, _compatibility_feedback: ("design retry", "retry"),
        build_compatibility_prompts=lambda _parsed: ("compatibility system", "compatibility user"),
    )

    try:
        with pytest.raises(NoveltyRejectionExhaustedError):
            generator.run_creation_stages(
                design_system_prompt="design system",
                design_user_prompt="design user",
                novelty_context=novelty_context,
                compatibility_context=compatibility_context,
                org_dir=organism_dir,
                organism_id="org_novelty_blocks_compatibility",
                generation=0,
            )
    finally:
        generator.close()

    assert calls == ["design", "novelty_check"]
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    assert llm_request["compatibility_checks"] == []
    assert "implementation" not in llm_request


def test_run_creation_stages_skips_implementation_when_compatibility_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    generator.evolver_cfg.creation.max_attempts_to_regenerate_organism_after_compatibility_rejection = 0
    organism_dir = tmp_path / "org_compatibility_blocks_implementation"
    organism_dir.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []

    def fake_generate(request):
        calls.append(request.stage)
        if request.stage == "design":
            text = _design_text("Rejected compatibility candidate.")
        elif request.stage == "novelty_check":
            text = _novelty_accepted_text()
        else:
            text = _compatibility_rejected_text("Update depends on absent gradient processing.", "GRADIENT_PROCESSING")
        return LlmResponse(
            text=text,
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": request.stage},
            raw_response={"stage": request.stage},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    novelty_context = NoveltyCheckContext(
        operator="mutation",
        build_design_prompts=lambda feedback: ("design retry", f"novelty :: {feedback[-1]}"),
        build_novelty_prompts=lambda _parsed: ("novelty system", "novelty user"),
    )
    compatibility_context = CompatibilityValidationContext(
        check=CompatibilityCheckContext(operator_kind="mutation"),
        build_design_prompts=lambda _novelty_feedback, _compatibility_feedback: ("design retry", "retry"),
        build_compatibility_prompts=lambda _parsed: ("compatibility system", "compatibility user"),
    )

    try:
        with pytest.raises(CompatibilityRejectionExhaustedError):
            generator.run_creation_stages(
                design_system_prompt="design system",
                design_user_prompt="design user",
                novelty_context=novelty_context,
                compatibility_context=compatibility_context,
                org_dir=organism_dir,
                organism_id="org_compatibility_blocks_implementation",
                generation=0,
            )
    finally:
        generator.close()

    assert calls == ["design", "novelty_check", "compatibility_check"]
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    assert len(llm_request["compatibility_checks"]) == 1
    assert "implementation" not in llm_request


def test_run_creation_stages_compatibility_retry_uses_own_budget_and_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    generator.evolver_cfg.creation.max_attempts_to_regenerate_organism_after_novelty_rejection = 0
    generator.evolver_cfg.creation.max_attempts_to_regenerate_organism_after_compatibility_rejection = 1
    organism_dir = tmp_path / "org_compatibility_retry"
    organism_dir.mkdir(parents=True, exist_ok=True)
    calls: list[tuple[str, str]] = []
    compatibility_responses = iter(
        [
            _compatibility_rejected_text(
                "Update depends on an absent gradient-processing rule.",
                "GRADIENT_PROCESSING, UPDATE_RULE",
            ),
            _compatibility_accepted_text(),
        ]
    )

    def fake_generate(request):
        calls.append((request.stage, request.user_prompt))
        if request.stage == "design":
            text = _design_text("Candidate after compatibility retry.")
        elif request.stage == "novelty_check":
            text = _novelty_accepted_text()
        elif request.stage == "compatibility_check":
            text = next(compatibility_responses)
        else:
            text = _implementation_text()
        return LlmResponse(
            text=text,
            route_id="mock",
            provider="mock",
            provider_model_id="mock-model",
            raw_request={"stage": request.stage},
            raw_response={"stage": request.stage},
            usage={},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
        )

    monkeypatch.setattr(generator.registry, "generate", fake_generate)
    novelty_context = NoveltyCheckContext(
        operator="mutation",
        build_design_prompts=lambda feedback: ("design retry", f"novelty :: {feedback[-1]}"),
        build_novelty_prompts=lambda _parsed: ("novelty system", "novelty user"),
    )
    compatibility_context = CompatibilityValidationContext(
        check=CompatibilityCheckContext(operator_kind="mutation"),
        build_design_prompts=lambda novelty_feedback, compatibility_feedback: (
            "design retry",
            "\n".join(novelty_feedback)
            + "\n"
            + format_compatibility_rejection_feedback(compatibility_feedback),
        ),
        build_compatibility_prompts=lambda _parsed: ("compatibility system", "compatibility user"),
    )

    try:
        generator.run_creation_stages(
            design_system_prompt="design system",
            design_user_prompt="design user",
            novelty_context=novelty_context,
            compatibility_context=compatibility_context,
            org_dir=organism_dir,
            organism_id="org_compatibility_retry",
            generation=0,
        )
    finally:
        generator.close()

    assert [stage for stage, _ in calls] == [
        "design",
        "novelty_check",
        "compatibility_check",
        "design",
        "novelty_check",
        "compatibility_check",
        "implementation",
    ]
    assert "Update depends on an absent gradient-processing rule." in calls[3][1]
    assert "Sections at issue" not in calls[3][1]
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    assert len(llm_request["compatibility_checks"]) == 2
    assert "Update depends on an absent gradient-processing rule." in llm_request["design_attempts"][1][
        "compatibility_rejection_feedback"
    ]
