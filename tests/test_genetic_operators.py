"""Direct operator tests with deterministic fake LLM responses."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from src.evolve.prompt_utils import load_prompt_bundle
from src.evolve.storage import read_lineage, sha1_text, write_json
from src.evolve.types import CreationStageResult, OrganismMeta
from src.evolve.template_parser import parse_llm_response
from src.organisms.crossbreeding import CrossbreedingOperator
from src.organisms.mutation import MutationOperator
from src.organisms.novelty import NoveltyRejectionExhaustedError, parse_novelty_judgment
from src.organisms.organism import (
    build_implementation_prompt_from_design,
    save_organism_artifacts,
)


def _cfg():
    return OmegaConf.create(
        {
            "seed": 42,
            "evolver": {
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


def _implementation_code(name: str = "ParentOpt") -> str:
    return (
        "import torch.nn as nn\n\n"
        f"OPTIMIZER_NAME = '{name}'\n\n"
        f"class {name}:\n"
        "    def __init__(self, model: nn.Module, max_steps: int) -> None:\n"
        "        self.model = model\n"
        "        self.max_steps = max_steps\n\n"
        "    def step(self, weights, grads, activations, step_fn) -> None:\n"
        "        del weights, grads, activations, step_fn\n\n"
        "    def zero_grad(self, set_to_none: bool = True) -> None:\n"
        "        del set_to_none\n\n"
        "def build_optimizer(model: nn.Module, max_steps: int):\n"
        f"    return {name}(model, max_steps)\n"
    )


def _make_parent(
    tmp_path: Path,
    org_id: str,
    island_id: str,
    genes: list[str],
) -> OrganismMeta:
    org_dir_path = tmp_path / f"org_{org_id}"
    org_dir_path.mkdir(parents=True, exist_ok=True)
    (org_dir_path / "implementation.py").write_text(_implementation_code(), encoding="utf-8")

    parent = OrganismMeta(
        organism_id=org_id,
        island_id=island_id,
        generation_created=0,
        current_generation_active=0,
        timestamp="2026-01-01T00:00:00Z",
        mother_id=None,
        father_id=None,
        operator="seed",
        genetic_code_path=str(org_dir_path / "genetic_code.md"),
        lineage_path=str(org_dir_path / "lineage.json"),
        model_name="mock",
        prompt_hash="abc",
        seed=123,
        organism_dir=str(org_dir_path),
        implementation_path=str(org_dir_path / "implementation.py"),
        ancestor_ids=[],
        experiment_report_index={},
        simple_score=0.5,
    )
    save_organism_artifacts(
        parent,
        genetic_code={
            "core_genes": genes,
            "interaction_notes": "Parent baseline organism.",
            "compute_notes": "No step_fn use.",
        },
        lineage=[
            {
                "generation": 0,
                "operator": "seed",
                "mother_id": None,
                "father_id": None,
                "change_description": "Initial creation",
                "selected_simple_experiments": ["simple_a"],
                "selected_hard_experiments": [],
                "simple_score": 0.5,
                "hard_score": None,
                "cross_island": False,
                "father_island_id": None,
            }
        ],
    )
    return parent


class _FakeCanonicalGenerator:
    def __init__(
        self,
        *,
        design_response_text: str,
        implementation_response_text: str,
        novelty_response_texts: list[str] | None = None,
    ) -> None:
        self.prompt_bundle = load_prompt_bundle(_cfg())
        self.seed = 123
        self.design_response_text = design_response_text
        self.implementation_response_text = implementation_response_text
        self.novelty_response_texts = novelty_response_texts or ["## NOVELTY_VERDICT\nNOVELTY_ACCEPTED\n"]
        self.calls: list[tuple[str, str, str, Path]] = []
        self.implementation_base_parent: OrganismMeta | None = None

    def run_creation_stages(
        self,
        *,
        design_system_prompt: str,
        design_user_prompt: str,
        org_dir: Path,
        organism_id: str,
        generation: int,
        island_id: str | None = None,
        novelty_context=None,
        compatibility_context=None,
        implementation_base_parent: OrganismMeta | None = None,
    ) -> CreationStageResult:
        self.implementation_base_parent = implementation_base_parent
        design_prompts: list[tuple[str, str]] = []
        novelty_exchanges: list[tuple[str, str, str]] = []
        rejection_feedback: list[str] = []
        parsed = None
        accepted_design_system_prompt = design_system_prompt
        accepted_design_user_prompt = design_user_prompt

        while True:
            current_design_system_prompt, current_design_user_prompt = (
                (design_system_prompt, design_user_prompt)
                if not design_prompts
                else novelty_context.build_design_prompts(rejection_feedback)
            )
            self.calls.append(("design", current_design_system_prompt, current_design_user_prompt, org_dir))
            design_prompts.append((current_design_system_prompt, current_design_user_prompt))
            parsed = parse_llm_response(self.design_response_text)
            if novelty_context is None:
                accepted_design_system_prompt = current_design_system_prompt
                accepted_design_user_prompt = current_design_user_prompt
                break

            novelty_system_prompt, novelty_user_prompt = novelty_context.build_novelty_prompts(parsed)
            self.calls.append(("novelty_check", novelty_system_prompt, novelty_user_prompt, org_dir))
            novelty_text = self.novelty_response_texts[len(novelty_exchanges)]
            novelty_exchanges.append((novelty_system_prompt, novelty_user_prompt, novelty_text))
            judgment = parse_novelty_judgment(novelty_text)
            if judgment.is_accepted:
                accepted_design_system_prompt = current_design_system_prompt
                accepted_design_user_prompt = current_design_user_prompt
                break

            rejection_feedback.append(judgment.rejection_reason or "Novelty rejected.")
            if len(novelty_exchanges) >= len(self.novelty_response_texts):
                raise NoveltyRejectionExhaustedError("novelty rejected in fake generator")

        implementation_system_prompt, implementation_user_prompt = build_implementation_prompt_from_design(
            parsed,
            self.prompt_bundle,
        )
        self.calls.append(("implementation", implementation_system_prompt, implementation_user_prompt, org_dir))
        request_payload = {
            "route_id": "mock",
            "provider": "test",
            "provider_model_id": "test-model",
            "design": {
                "system_prompt": accepted_design_system_prompt,
                "user_prompt": accepted_design_user_prompt,
                "request": {"provider": "test", "stage": "design"},
            },
            "implementation": {
                "system_prompt": implementation_system_prompt,
                "user_prompt": implementation_user_prompt,
                "request": {"provider": "test", "stage": "implementation"},
            },
        }
        response_payload = {
            "route_id": "mock",
            "provider": "test",
            "provider_model_id": "test-model",
            "design": {
                "text": self.design_response_text,
                "response": {"provider": "test", "stage": "design"},
            },
            "implementation": {
                "text": self.implementation_response_text,
                "response": {"provider": "test", "stage": "implementation"},
            },
        }
        if novelty_context is not None:
            request_payload["design_attempts"] = [
                {
                    "attempt": index + 1,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "request": {"provider": "test", "stage": "design"},
                }
                for index, (system_prompt, user_prompt) in enumerate(design_prompts)
            ]
            response_payload["design_attempts"] = [
                {
                    "attempt": index + 1,
                    "text": self.design_response_text,
                    "response": {"provider": "test", "stage": "design"},
                }
                for index, _ in enumerate(design_prompts)
            ]
            request_payload["novelty_checks"] = [
                {
                    "attempt": index + 1,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "request": {"provider": "test", "stage": "novelty_check"},
                }
                for index, (system_prompt, user_prompt, _) in enumerate(novelty_exchanges)
            ]
            response_payload["novelty_checks"] = [
                {
                    "attempt": index + 1,
                    "text": novelty_text,
                    "response": {"provider": "test", "stage": "novelty_check"},
                }
                for index, (_, _, novelty_text) in enumerate(novelty_exchanges)
            ]
        write_json(org_dir / "llm_request.json", request_payload)
        write_json(org_dir / "llm_response.json", response_payload)
        prompt_hash = sha1_text(
            "\n".join(
                (
                    *(piece for prompts in design_prompts for piece in prompts),
                    *(piece for system_prompt, user_prompt, _ in novelty_exchanges for piece in (system_prompt, user_prompt)),
                    implementation_system_prompt,
                    implementation_user_prompt,
                )
            )
        )
        return CreationStageResult(
            parsed_design=parsed,
            implementation_code=self.implementation_response_text,
            prompt_hash=prompt_hash,
            llm_route_id="mock",
            llm_provider="test",
            provider_model_id="test-model",
        )


class _FakePatchCompilationGenerator(_FakeCanonicalGenerator):
    def uses_section_patch_compilation(self) -> bool:
        return True


def test_mutation_operator_produce_persists_artifacts_and_lineage(tmp_path: Path) -> None:
    parent = _make_parent(
        tmp_path,
        org_id="parent01",
        island_id="gradient_methods",
        genes=["adaptive momentum", "warmup schedule", "gradient clipping"],
    )
    design_response_text = (
        "## CORE_GENES\n"
        "- adaptive momentum\n"
        "- warmup schedule\n"
        "- trust ratio scaling\n\n"
        "## INTERACTION_NOTES\n"
        "Retain stable momentum and warmup while adding trust-ratio scaling.\n\n"
        "## COMPUTE_NOTES\n"
        "Keep step_fn unused and update compute light.\n\n"
        "## CHANGE_DESCRIPTION\n"
        "Dropped clipping and replaced it with trust-ratio scaling.\n"
    )
    implementation_response_text = _implementation_code("MutationChild")
    generator = _FakeCanonicalGenerator(
        design_response_text=design_response_text,
        implementation_response_text=implementation_response_text,
    )
    operator = MutationOperator(q=0.6, seed=1)
    org_dir = tmp_path / "child_mutation"
    org_dir.mkdir(parents=True, exist_ok=True)

    child = operator.produce(
        parent=parent,
        organism_id="child01",
        generation=1,
        org_dir=org_dir,
        generator=generator,
    )

    _, _, user_prompt, _ = generator.calls[0]
    assert "=== INHERITED GENE POOL ===" in user_prompt
    assert "=== REMOVED GENES ===" in user_prompt
    assert "=== NOVELTY REJECTION FEEDBACK ===" in user_prompt
    assert "=== PARENT GENETIC CODE ===" in user_prompt
    assert "IMPLEMENTATION CODE" not in user_prompt
    assert generator.calls[1][0] == "novelty_check"
    assert "=== CANDIDATE CHILD GENETIC CODE ===" in generator.calls[1][2]
    assert generator.calls[2][0] == "implementation"
    assert "=== FIXED IMPLEMENTATION TEMPLATE ===" in generator.calls[2][2]
    assert child.island_id == parent.island_id
    assert child.mother_id == parent.organism_id
    assert child.implementation_path == str(org_dir / "implementation.py")
    assert (org_dir / "implementation.py").read_text(encoding="utf-8") == implementation_response_text
    latest = read_lineage(org_dir / "lineage.json")[-1]
    assert latest["operator"] == "mutation"
    assert latest["change_description"] == "Dropped clipping and replaced it with trust-ratio scaling."
    assert "gene_diff_summary" not in latest
    llm_request = json.loads((org_dir / "llm_request.json").read_text(encoding="utf-8"))
    assert llm_request["route_id"] == "mock"
    assert {"design", "implementation"}.issubset(llm_request.keys())
    assert child.llm_route_id == "mock"
    assert child.llm_provider == "test"
    assert child.provider_model_id == "test-model"


def test_crossover_operator_produce_records_cross_island_lineage(tmp_path: Path) -> None:
    mother = _make_parent(
        tmp_path / "mother",
        org_id="mother01",
        island_id="gradient_methods",
        genes=["adaptive momentum", "warmup schedule", "gradient clipping"],
    )
    father = _make_parent(
        tmp_path / "father",
        org_id="father01",
        island_id="second_order",
        genes=["diagonal preconditioning", "curvature damping", "gradient clipping"],
    )
    design_response_text = (
        "## CORE_GENES\n"
        "- adaptive momentum\n"
        "- warmup schedule\n"
        "- diagonal preconditioning\n\n"
        "## INTERACTION_NOTES\n"
        "Preserve maternal schedule while injecting diagonal preconditioning.\n\n"
        "## COMPUTE_NOTES\n"
        "No extra closures; keep the controller cheap.\n\n"
        "## CHANGE_DESCRIPTION\n"
        "Kept the maternal schedule and introduced paternal diagonal preconditioning.\n"
    )
    implementation_response_text = _implementation_code("CrossoverChild")
    generator = _FakeCanonicalGenerator(
        design_response_text=design_response_text,
        implementation_response_text=implementation_response_text,
    )
    operator = CrossbreedingOperator(p=1.0, seed=3)
    org_dir = tmp_path / "child_crossover"
    org_dir.mkdir(parents=True, exist_ok=True)

    child = operator.produce(
        mother=mother,
        father=father,
        organism_id="child02",
        generation=1,
        org_dir=org_dir,
        generator=generator,
    )

    _, _, user_prompt, _ = generator.calls[0]
    assert "=== NOVELTY REJECTION FEEDBACK ===" in user_prompt
    assert "MOTHER (primary parent" in user_prompt
    assert "FATHER (secondary parent" in user_prompt
    assert "Mother implementation code" not in user_prompt
    assert "Father implementation code" not in user_prompt
    assert generator.calls[1][0] == "novelty_check"
    assert "=== CANDIDATE CHILD GENETIC CODE ===" in generator.calls[1][2]
    assert child.island_id == mother.island_id
    assert child.mother_id == mother.organism_id
    assert child.father_id == father.organism_id
    latest = read_lineage(org_dir / "lineage.json")[-1]
    assert latest["cross_island"] is True
    assert latest["father_island_id"] == "second_order"
    assert latest["change_description"] == "Kept the maternal schedule and introduced paternal diagonal preconditioning."


def test_mutation_operator_passes_novelty_feedback_into_regeneration_prompt(tmp_path: Path) -> None:
    parent = _make_parent(
        tmp_path,
        org_id="parent02",
        island_id="gradient_methods",
        genes=["adaptive momentum", "warmup schedule", "gradient clipping"],
    )
    design_response_text = (
        "## CORE_GENES\n"
        "- adaptive momentum\n"
        "- warmup schedule\n"
        "- trust ratio scaling\n\n"
        "## INTERACTION_NOTES\n"
        "Retain stable momentum and warmup while adding trust-ratio scaling.\n\n"
        "## COMPUTE_NOTES\n"
        "Keep step_fn unused and update compute light.\n\n"
        "## CHANGE_DESCRIPTION\n"
        "Dropped clipping and replaced it with trust-ratio scaling.\n"
    )
    generator = _FakeCanonicalGenerator(
        design_response_text=design_response_text,
        implementation_response_text=_implementation_code("MutationRetryChild"),
        novelty_response_texts=[
            (
                "## NOVELTY_VERDICT\n"
                "NOVELTY_REJECTED\n\n"
                "## REJECTION_REASON\n"
                "This only paraphrases the parent momentum idea.\n"
            ),
            "## NOVELTY_VERDICT\nNOVELTY_ACCEPTED\n",
        ],
    )
    operator = MutationOperator(q=0.6, seed=1)
    org_dir = tmp_path / "child_mutation_retry"
    org_dir.mkdir(parents=True, exist_ok=True)

    operator.produce(
        parent=parent,
        organism_id="child03",
        generation=1,
        org_dir=org_dir,
        generator=generator,
    )

    retry_design_prompt = generator.calls[2][2]
    assert generator.calls[2][0] == "design"
    assert "This only paraphrases the parent momentum idea." in retry_design_prompt


def test_mutation_operator_passes_parent_as_patch_compilation_base(tmp_path: Path) -> None:
    parent = _make_parent(
        tmp_path,
        org_id="patch_parent",
        island_id="gradient_methods",
        genes=["adaptive momentum", "warmup schedule", "gradient clipping"],
    )
    generator = _FakePatchCompilationGenerator(
        design_response_text=(
            "## CORE_GENES\n"
            "- adaptive momentum\n"
            "- warmup schedule\n"
            "- trust ratio scaling\n\n"
            "## INTERACTION_NOTES\n"
            "Retain stable momentum and warmup while adding trust-ratio scaling.\n\n"
            "## COMPUTE_NOTES\n"
            "Keep step_fn unused and update compute light.\n\n"
            "## CHANGE_DESCRIPTION\n"
            "Dropped clipping and replaced it with trust-ratio scaling.\n"
        ),
        implementation_response_text=_implementation_code("PatchMutationChild"),
    )
    org_dir = tmp_path / "child_patch_mutation"
    org_dir.mkdir(parents=True, exist_ok=True)

    MutationOperator(q=0.6, seed=1).produce(
        parent=parent,
        organism_id="patch_child01",
        generation=1,
        org_dir=org_dir,
        generator=generator,
    )

    assert generator.implementation_base_parent is parent


def test_crossover_operator_passes_mother_as_patch_compilation_base(tmp_path: Path) -> None:
    mother = _make_parent(
        tmp_path / "patch_mother",
        org_id="patch_mother",
        island_id="gradient_methods",
        genes=["adaptive momentum", "warmup schedule", "gradient clipping"],
    )
    father = _make_parent(
        tmp_path / "patch_father",
        org_id="patch_father",
        island_id="second_order",
        genes=["diagonal preconditioning", "curvature damping", "gradient clipping"],
    )
    generator = _FakePatchCompilationGenerator(
        design_response_text=(
            "## CORE_GENES\n"
            "- adaptive momentum\n"
            "- warmup schedule\n"
            "- diagonal preconditioning\n\n"
            "## INTERACTION_NOTES\n"
            "Preserve maternal schedule while injecting diagonal preconditioning.\n\n"
            "## COMPUTE_NOTES\n"
            "No extra closures; keep the controller cheap.\n\n"
            "## CHANGE_DESCRIPTION\n"
            "Kept the maternal schedule and introduced paternal diagonal preconditioning.\n"
        ),
        implementation_response_text=_implementation_code("PatchCrossoverChild"),
    )
    org_dir = tmp_path / "child_patch_crossover"
    org_dir.mkdir(parents=True, exist_ok=True)

    CrossbreedingOperator(p=1.0, seed=3).produce(
        mother=mother,
        father=father,
        organism_id="patch_child02",
        generation=1,
        org_dir=org_dir,
        generator=generator,
    )

    assert generator.implementation_base_parent is mother
    assert generator.implementation_base_parent is not father
