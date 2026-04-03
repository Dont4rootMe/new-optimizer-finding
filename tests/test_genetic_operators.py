"""Direct canonical operator tests with deterministic fake LLM responses."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from src.evolve.prompt_utils import load_prompt_bundle
from src.evolve.storage import write_json
from src.evolve.types import OrganismMeta
from src.organisms.crossbreeding import CrossbreedingOperator
from src.organisms.mutation import MutationOperator
from src.organisms.organism import save_organism_artifacts


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
    return (
        "# === FIXED: DO NOT MODIFY ===\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "# === END FIXED ===\n"
        "\n"
        "# === EDITABLE: IMPORTS ===\n"
        "import math\n"
        "# === END EDITABLE ===\n"
        "\n"
        "# === FIXED ===\n"
        "OPTIMIZER_NAME = \"ParentOpt\"\n"
        "\n"
        "class ParentOpt:\n"
        "    def __init__(self, model: nn.Module, max_steps: int) -> None:\n"
        "# === END FIXED ===\n"
        "        # === EDITABLE: INIT_BODY ===\n"
        "        self.model = model\n"
        "        # === END EDITABLE ===\n"
        "\n"
        "# === FIXED ===\n"
        "    def step(self, weights, grads, activations, step_fn) -> None:\n"
        "# === END FIXED ===\n"
        "        # === EDITABLE: STEP_BODY ===\n"
        "        del weights, grads, activations, step_fn\n"
        "        # === END EDITABLE ===\n"
        "\n"
        "# === FIXED ===\n"
        "    def zero_grad(self, set_to_none: bool = True) -> None:\n"
        "# === END FIXED ===\n"
        "        # === EDITABLE: ZERO_GRAD_BODY ===\n"
        "        pass\n"
        "        # === END EDITABLE ===\n"
        "\n"
        "# === FIXED ===\n"
        "def build_optimizer(model: nn.Module, max_steps: int):\n"
        "    return ParentOpt(model, max_steps)\n"
        "# === END FIXED ===\n"
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
        selection_reward=0.5,
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
                "gene_diff_summary": "; ".join(genes),
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
    def __init__(self, response_text: str) -> None:
        self.prompt_bundle = load_prompt_bundle(_cfg())
        self.model_name = "mock-model"
        self.seed = 123
        self.response_text = response_text
        self.calls: list[tuple[str, str, Path]] = []

    def call_llm(self, system_prompt: str, user_prompt: str, org_dir: Path) -> str:
        self.calls.append((system_prompt, user_prompt, org_dir))
        write_json(org_dir / "llm_request.json", {"provider": "test"})
        write_json(org_dir / "llm_response.json", {"provider": "test", "text": self.response_text})
        return self.response_text


def test_mutation_operator_produce_persists_artifacts_and_semantic_lineage(tmp_path: Path) -> None:
    parent = _make_parent(
        tmp_path,
        org_id="parent01",
        island_id="gradient_methods",
        genes=["adaptive momentum", "warmup schedule", "gradient clipping"],
    )
    response_text = (
        "## CORE_GENES\n"
        "- adaptive momentum\n"
        "- warmup schedule\n"
        "- trust ratio scaling\n\n"
        "## INTERACTION_NOTES\n"
        "Retain stable momentum and warmup while adding trust-ratio scaling.\n\n"
        "## COMPUTE_NOTES\n"
        "Keep step_fn unused and update compute light.\n\n"
        "## CHANGE_DESCRIPTION\n"
        "Dropped clipping and replaced it with trust-ratio scaling.\n\n"
        "## IMPORTS\n"
        "import math\n\n"
        "## INIT_BODY\n"
        "        self.model = model\n"
        "        self.max_steps = max_steps\n\n"
        "## STEP_BODY\n"
        "        del weights, grads, activations, step_fn\n\n"
        "## ZERO_GRAD_BODY\n"
        "        pass\n"
    )
    generator = _FakeCanonicalGenerator(response_text)
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

    _, user_prompt, _ = generator.calls[0]
    assert "Inherited gene pool" in user_prompt
    assert "Removed genes" in user_prompt
    assert child.island_id == parent.island_id
    assert child.mother_id == parent.organism_id
    assert (org_dir / "genetic_code.md").exists()
    assert (org_dir / "lineage.json").exists()
    lineage = json.loads((org_dir / "lineage.json").read_text(encoding="utf-8"))
    latest = lineage[-1]
    assert latest["operator"] == "mutation"
    assert "gradient clipping" in latest["gene_diff_summary"]
    assert "trust ratio scaling" in latest["gene_diff_summary"]
    assert "aggregate_score" not in latest


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
    response_text = (
        "## CORE_GENES\n"
        "- adaptive momentum\n"
        "- warmup schedule\n"
        "- diagonal preconditioning\n\n"
        "## INTERACTION_NOTES\n"
        "Preserve maternal schedule while injecting diagonal preconditioning.\n\n"
        "## COMPUTE_NOTES\n"
        "No extra closures; keep the controller cheap.\n\n"
        "## CHANGE_DESCRIPTION\n"
        "Kept the maternal schedule and introduced paternal diagonal preconditioning.\n\n"
        "## IMPORTS\n"
        "import math\n\n"
        "## INIT_BODY\n"
        "        self.model = model\n"
        "        self.max_steps = max_steps\n\n"
        "## STEP_BODY\n"
        "        del weights, grads, activations, step_fn\n\n"
        "## ZERO_GRAD_BODY\n"
        "        pass\n"
    )
    generator = _FakeCanonicalGenerator(response_text)
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

    _, user_prompt, _ = generator.calls[0]
    assert "Inherited maternal-biased gene pool" in user_prompt
    assert child.island_id == mother.island_id
    assert child.mother_id == mother.organism_id
    assert child.father_id == father.organism_id
    assert len(child.lineage) == len(mother.lineage) + 1
    latest = child.lineage[-1]
    assert latest["cross_island"] is True
    assert latest["father_island_id"] == "second_order"
    assert "adaptive momentum" in latest["gene_diff_summary"]
    assert "diagonal preconditioning" in latest["gene_diff_summary"]
