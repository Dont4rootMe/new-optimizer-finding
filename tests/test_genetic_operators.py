"""Direct canonical operator tests with deterministic fake LLM responses."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from src.evolve.prompt_utils import load_prompt_bundle
from src.evolve.storage import sha1_text, write_json
from src.evolve.template_parser import parse_llm_response, render_template
from src.evolve.types import OrganismMeta
from src.organisms.crossbreeding import CrossbreedingOperator
from src.organisms.mutation import MutationOperator
from src.organisms.organism import build_implementation_prompt_from_design, save_organism_artifacts


def _cfg():
    return OmegaConf.create(
        {
            "seed": 42,
            "evolver": {
                "prompts": {
                    "project_context": "conf/prompts/shared/project_context.txt",
                    "seed_system": "conf/prompts/seed/system.txt",
                    "seed_user": "conf/prompts/seed/user.txt",
                    "mutation_system": "conf/prompts/mutation/system.txt",
                    "mutation_user": "conf/prompts/mutation/user.txt",
                    "crossover_system": "conf/prompts/crossover/system.txt",
                    "crossover_user": "conf/prompts/crossover/user.txt",
                    "implementation_system": "conf/prompts/implementation/system.txt",
                    "implementation_user": "conf/prompts/implementation/user.txt",
                    "implementation_template": "conf/prompts/implementation/template.txt",
                },
                "llm": {
                    "provider": "mock",
                    "model": "mock-model",
                    "temperature": 0.0,
                    "max_output_tokens": 512,
                    "reasoning_effort": None,
                    "seed": 123,
                    "fallback_to_chat_completions": True,
                },
            },
        }
    )


def _optimizer_code() -> str:
    return render_template(
        {
            "IMPORTS": "import math",
            "INIT_BODY": "        self.model = model\n        self.max_steps = max_steps",
            "STEP_BODY": "        del weights, grads, activations, step_fn",
            "ZERO_GRAD_BODY": "        pass",
        },
        optimizer_name="ParentOpt",
        class_name="ParentOpt",
    )


def _make_parent(
    tmp_path: Path,
    org_id: str,
    island_id: str,
    genes: list[str],
) -> OrganismMeta:
    org_dir_path = tmp_path / f"org_{org_id}"
    org_dir_path.mkdir(parents=True, exist_ok=True)
    (org_dir_path / "optimizer.py").write_text(_optimizer_code(), encoding="utf-8")

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
        optimizer_path=str(org_dir_path / "optimizer.py"),
        simple_reward=0.5,
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
    save_organism_artifacts(parent)
    return parent


class _FakeCanonicalGenerator:
    def __init__(self, *, design_response_text: str, implementation_response_text: str) -> None:
        self.prompt_bundle = load_prompt_bundle(_cfg())
        self.model_name = "mock-model"
        self.seed = 123
        self.design_response_text = design_response_text
        self.implementation_response_text = implementation_response_text
        self.calls: list[tuple[str, str, str, Path]] = []

    def run_creation_stages(
        self,
        *,
        design_system_prompt: str,
        design_user_prompt: str,
        org_dir: Path,
        organism_id: str,
        generation: int,
    ) -> tuple[dict[str, str], str, str]:
        del generation
        self.calls.append(("design", design_system_prompt, design_user_prompt, org_dir))
        parsed = parse_llm_response(self.design_response_text)
        implementation_system_prompt, implementation_user_prompt = build_implementation_prompt_from_design(
            parsed,
            self.prompt_bundle,
        )
        self.calls.append(("implementation", implementation_system_prompt, implementation_user_prompt, org_dir))
        write_json(
            org_dir / "llm_request.json",
            {
                "design": {
                    "system_prompt": design_system_prompt,
                    "user_prompt": design_user_prompt,
                    "request": {"provider": "test", "stage": "design"},
                },
                "implementation": {
                    "system_prompt": implementation_system_prompt,
                    "user_prompt": implementation_user_prompt,
                    "request": {"provider": "test", "stage": "implementation"},
                },
            },
        )
        write_json(
            org_dir / "llm_response.json",
            {
                "design": {
                    "text": self.design_response_text,
                    "response": {"provider": "test", "stage": "design"},
                },
                "implementation": {
                    "text": self.implementation_response_text,
                    "response": {"provider": "test", "stage": "implementation"},
                },
            },
        )
        prompt_hash = sha1_text(
            "\n".join(
                (
                    design_system_prompt,
                    design_user_prompt,
                    implementation_system_prompt,
                    implementation_user_prompt,
                )
            )
        )
        return parsed, self.implementation_response_text, prompt_hash


def test_mutation_operator_produce_persists_artifacts_and_semantic_lineage(tmp_path: Path) -> None:
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
    implementation_response_text = render_template(
        {
            "IMPORTS": "import math",
            "INIT_BODY": "        self.model = model\n        self.max_steps = max_steps",
            "STEP_BODY": "        del weights, grads, activations, step_fn",
            "ZERO_GRAD_BODY": "        pass",
        },
        optimizer_name="MutationChild",
        class_name="MutationChild",
        template_text=load_prompt_bundle(_cfg()).implementation_template,
    )
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
    assert generator.calls[1][0] == "implementation"
    assert "=== FIXED OPTIMIZER TEMPLATE ===" in generator.calls[1][2]
    assert child.island_id == parent.island_id
    assert child.mother_id == parent.organism_id
    assert (org_dir / "genetic_code.md").exists()
    assert (org_dir / "lineage.json").exists()
    assert (org_dir / "optimizer.py").read_text(encoding="utf-8") == implementation_response_text
    lineage = json.loads((org_dir / "lineage.json").read_text(encoding="utf-8"))
    latest = lineage[-1]
    assert latest["operator"] == "mutation"
    assert latest["change_description"] == "Dropped clipping and replaced it with trust-ratio scaling."
    assert "gene_diff_summary" not in latest
    assert "aggregate_score" not in latest
    llm_request = json.loads((org_dir / "llm_request.json").read_text(encoding="utf-8"))
    llm_response = json.loads((org_dir / "llm_response.json").read_text(encoding="utf-8"))
    assert set(llm_request.keys()) == {"design", "implementation"}
    assert set(llm_response.keys()) == {"design", "implementation"}


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
    implementation_response_text = render_template(
        {
            "IMPORTS": "import math",
            "INIT_BODY": "        self.model = model\n        self.max_steps = max_steps",
            "STEP_BODY": "        del weights, grads, activations, step_fn",
            "ZERO_GRAD_BODY": "        pass",
        },
        optimizer_name="CrossoverChild",
        class_name="CrossoverChild",
        template_text=load_prompt_bundle(_cfg()).implementation_template,
    )
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
    assert "=== INHERITED GENE POOL ===" in user_prompt
    assert "MOTHER (primary parent" in user_prompt
    assert "FATHER (secondary parent" in user_prompt
    assert generator.calls[1][0] == "implementation"
    assert child.island_id == mother.island_id
    assert child.mother_id == mother.organism_id
    assert child.father_id == father.organism_id
    assert len(child.lineage) == len(mother.lineage) + 1
    latest = child.lineage[-1]
    assert latest["cross_island"] is True
    assert latest["father_island_id"] == "second_order"
    assert latest["change_description"] == "Kept the maternal schedule and introduced paternal diagonal preconditioning."
    assert "gene_diff_summary" not in latest
