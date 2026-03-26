"""Tests for genetic operators with mock LLM."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.evolve.generator import OptimizerGenerator
from src.evolve.operators import SeedOperator, MutationOperator, CrossoverOperator
from src.evolve.storage import organism_dir
from src.evolve.types import OrganismMeta


def _cfg():
    return OmegaConf.create(
        {
            "seed": 42,
            "evolver": {
                "max_generation_attempts": 2,
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


def _make_parent(tmp_path: Path, org_id: str = "parent01") -> OrganismMeta:
    """Create a parent organism with a valid optimizer.py on disk."""
    org_dir_path = tmp_path / f"org_{org_id}"
    org_dir_path.mkdir(parents=True, exist_ok=True)

    # Write a simple optimizer.py with editable sections
    code = (
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
        "        pass\n"
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
    (org_dir_path / "optimizer.py").write_text(code, encoding="utf-8")

    return OrganismMeta(
        organism_id=org_id,
        generation=0,
        timestamp="2026-01-01T00:00:00Z",
        parent_ids=[],
        operator="seed",
        idea_dna=["simple SGD", "constant learning rate"],
        evolution_log=[
            {"generation": 0, "change_description": "Initial creation", "score": 0.5, "parent_ids": []}
        ],
        model_name="mock",
        prompt_hash="abc",
        seed=123,
        organism_dir=str(org_dir_path),
        optimizer_path=str(org_dir_path / "optimizer.py"),
        score=0.5,
        simple_score=0.5,
    )


def test_seed_operator_generates_organism(tmp_path: Path) -> None:
    generator = OptimizerGenerator(_cfg())
    seed_op = SeedOperator()
    org_dir_path = tmp_path / "gen_0" / "org_test_seed"
    org_dir_path.mkdir(parents=True, exist_ok=True)

    org = generator.generate_organism(
        operator=seed_op,
        parents=[],
        organism_id="test_seed",
        generation=0,
        organism_dir=org_dir_path,
    )

    assert org.organism_id == "test_seed"
    assert org.operator == "seed"
    assert org.generation == 0
    assert len(org.idea_dna) > 0
    assert (org_dir_path / "optimizer.py").exists()
    assert (org_dir_path / "organism.json").exists()


def test_mutation_operator_generates_organism(tmp_path: Path) -> None:
    parent = _make_parent(tmp_path)
    generator = OptimizerGenerator(_cfg())
    mutation_op = MutationOperator()
    org_dir_path = tmp_path / "gen_1" / "org_mutant"
    org_dir_path.mkdir(parents=True, exist_ok=True)

    org = generator.generate_organism(
        operator=mutation_op,
        parents=[parent],
        organism_id="mutant",
        generation=1,
        organism_dir=org_dir_path,
    )

    assert org.organism_id == "mutant"
    assert org.operator == "mutation"
    assert org.generation == 1
    assert parent.organism_id in org.parent_ids
    assert (org_dir_path / "optimizer.py").exists()


def test_crossover_operator_generates_organism(tmp_path: Path) -> None:
    parent_a = _make_parent(tmp_path / "parents", "parent_a")
    parent_b = _make_parent(tmp_path / "parents", "parent_b")
    generator = OptimizerGenerator(_cfg())
    crossover_op = CrossoverOperator()
    org_dir_path = tmp_path / "gen_1" / "org_cross"
    org_dir_path.mkdir(parents=True, exist_ok=True)

    org = generator.generate_organism(
        operator=crossover_op,
        parents=[parent_a, parent_b],
        organism_id="cross",
        generation=1,
        organism_dir=org_dir_path,
    )

    assert org.organism_id == "cross"
    assert org.operator == "crossover"
    assert len(org.parent_ids) == 2
    assert (org_dir_path / "optimizer.py").exists()
