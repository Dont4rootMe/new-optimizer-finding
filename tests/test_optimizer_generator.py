"""Unit tests for generic organism generator behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from api_platforms import LlmResponse
import src.evolve.operators as canonical_seed_operators
from src.evolve.generator import CandidateGenerator
from src.evolve.types import Island, OrganismMeta
from src.organisms.compatibility import (
    CompatibilityCheckContext,
    CompatibilityRejectionExhaustedError,
    CompatibilityValidationContext,
    format_compatibility_rejection_feedback,
)
from src.organisms.novelty import NoveltyCheckContext, NoveltyRejectionExhaustedError
from src.organisms.organism import save_organism_artifacts


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
                    "seed_system": "conf/experiments/optimization_survey/prompts/seed/system.txt",
                    "seed_user": "conf/experiments/optimization_survey/prompts/seed/user.txt",
                    "mutation_system": "conf/experiments/optimization_survey/prompts/mutation/system.txt",
                    "mutation_user": "conf/experiments/optimization_survey/prompts/mutation/user.txt",
                    "mutation_novelty_system": "conf/experiments/optimization_survey/prompts/novelty/mutation/system.txt",
                    "mutation_novelty_user": "conf/experiments/optimization_survey/prompts/novelty/mutation/user.txt",
                    "crossover_system": "conf/experiments/optimization_survey/prompts/crossover/system.txt",
                    "crossover_user": "conf/experiments/optimization_survey/prompts/crossover/user.txt",
                    "crossover_novelty_system": "conf/experiments/optimization_survey/prompts/novelty/crossover/system.txt",
                    "crossover_novelty_user": "conf/experiments/optimization_survey/prompts/novelty/crossover/user.txt",
                    "implementation_system": "conf/experiments/optimization_survey/prompts/implementation/system.txt",
                    "implementation_user": "conf/experiments/optimization_survey/prompts/implementation/user.txt",
                    "implementation_template": "conf/experiments/optimization_survey/prompts/implementation/template.txt",
                    "repair_system": "conf/experiments/optimization_survey/prompts/repair/system.txt",
                    "repair_user": "conf/experiments/optimization_survey/prompts/repair/user.txt",
                },
                "llm": {"route_weights": {"mock": 1.0}, "seed": 123},
            },
            "api_platforms": {"mock": {"_target_": "api_platforms.mock.platform.build_platform"}},
            "paths": {"api_platform_runtime_root": ".tmp_api_platform_runtime"},
        }
    )


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
    return (
        "## CORE_GENES\n"
        "- candidate gene one\n"
        "- candidate gene two\n"
        "- candidate gene three\n\n"
        "## INTERACTION_NOTES\n"
        "Candidate.\n\n"
        "## COMPUTE_NOTES\n"
        "Cheap.\n\n"
        "## CHANGE_DESCRIPTION\n"
        f"{change_description}\n"
    )


def _implementation_text() -> str:
    return (
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
        "    return Impl(model, max_steps)\n"
    )


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
        "N/A\n\n"
        "## SECTIONS_AT_ISSUE\n"
        "NONE\n"
    )


def _compatibility_rejected_text(reason: str, sections: str = "NONE") -> str:
    return (
        "## COMPATIBILITY_VERDICT\n"
        "COMPATIBILITY_REJECTED\n\n"
        "## REJECTION_REASON\n"
        f"{reason}\n\n"
        "## SECTIONS_AT_ISSUE\n"
        f"{sections}\n"
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
        "seed/system.txt",
        "seed/user.txt",
        "mutation/system.txt",
        "mutation/user.txt",
        "novelty/mutation/system.txt",
        "novelty/mutation/user.txt",
        "crossover/system.txt",
        "crossover/user.txt",
        "novelty/crossover/system.txt",
        "novelty/crossover/user.txt",
        "implementation/system.txt",
        "implementation/user.txt",
        "implementation/template.txt",
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
    assert "design_attempts" not in llm_request
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
            text = "def broken(\n"
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
    assert llm_response["implementation"]["text"] == "def broken(\n"
    assert llm_response["implementation"]["status"] == "failed"
    assert "syntactically invalid Python" in llm_response["implementation"]["error_msg"]


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
            "## NOVELTY_VERDICT\nNOVELTY_REJECTED\n\n## REJECTION_REASON\nToo close to the parent.\n",
            "## NOVELTY_VERDICT\nNOVELTY_ACCEPTED\n",
        ]
    )

    def fake_generate(request):
        calls.append((request.stage, request.user_prompt))
        if request.stage == "design":
            text = next(design_responses)
        elif request.stage == "novelty_check":
            text = next(novelty_responses)
        else:
            text = (
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
                "    return Impl(model, max_steps)\n"
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
                "Still paraphrases the parent.\n"
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
            "## NOVELTY_VERDICT\nNOVELTY_ACCEPTED\n",
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
            text = (
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
                "    return Impl(model, max_steps)\n"
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
        )
    finally:
        generator.close()

    assert calls == ["design", "compatibility_check", "implementation"]
    assert result.parsed_design["CHANGE_DESCRIPTION"] == "Seed candidate."
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    assert "novelty_checks" not in llm_request
    assert len(llm_request["compatibility_checks"]) == 1


def test_run_creation_stages_mutation_runs_novelty_before_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_mutation_validation"
    organism_dir.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []

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
        )
    finally:
        generator.close()

    assert calls == ["design", "novelty_check", "compatibility_check", "implementation"]


def test_run_creation_stages_crossover_runs_novelty_before_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = CandidateGenerator(_cfg())
    organism_dir = tmp_path / "org_crossover_validation"
    organism_dir.mkdir(parents=True, exist_ok=True)
    calls: list[str] = []

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
        )
    finally:
        generator.close()

    assert calls == ["design", "novelty_check", "compatibility_check", "implementation"]


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
            text = _compatibility_rejected_text("Repair depends on an absent conflict model.", "CONFLICT_MODEL")
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
                "Repair depends on an absent conflict ranking.",
                "CONFLICT_MODEL, REPAIR_POLICY",
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
    assert "Repair depends on an absent conflict ranking." in calls[3][1]
    assert "Sections at issue: CONFLICT_MODEL, REPAIR_POLICY" in calls[3][1]
    llm_request = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    assert len(llm_request["compatibility_checks"]) == 2
    assert "Repair depends on an absent conflict ranking." in llm_request["design_attempts"][1][
        "compatibility_rejection_feedback"
    ]
