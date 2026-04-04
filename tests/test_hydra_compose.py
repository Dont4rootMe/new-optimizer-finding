"""Tests for Hydra config composition."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]


def test_hydra_compose_config_and_experiments() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config")

    assert "experiments" in cfg
    assert "evolver" in cfg
    expected = {
        "cifar_convnet",
        "audio_transformer",
        "minigpt_wikitext2",
        "ddpm_cifar10",
        "lora_sft",
        "synthetic_logreg",
        "mnist_mlp",
        "poly_regression",
        "rosenbrock_net",
        "xor_mlp",
        "sin_regression",
        "matrix_factorization",
        "tiny_autoencoder",
        "two_spirals",
        "linear_denoiser",
        "conv1d_classify",
        "quadratic_bowl",
    }
    assert set(cfg.experiments.keys()) == expected

    for exp_name in expected:
        assert "enabled" in cfg.experiments[exp_name]
        assert "run_validation" in cfg.experiments[exp_name]
        assert "normalization" in cfg.experiments[exp_name]
        assert "quality_ref" not in cfg.experiments[exp_name].normalization
        assert "steps_ref" not in cfg.experiments[exp_name].normalization

    assert "selection_strategy" not in cfg.evolver
    assert "eval_experiments" not in cfg.evolver
    assert "evaluation" not in cfg.evolver
    assert "organisms_per_island" not in cfg.evolver.islands
    assert "inter_island_crossover_rate" not in cfg.evolver.islands
    assert "probability" not in cfg.evolver.operators.mutation
    assert "parent_sampling" not in cfg.evolver.operators.mutation
    assert "parent_sampling" not in cfg.evolver.operators.crossover
    assert cfg.evolver.max_generations == 100
    assert cfg.evolver.islands.dir == "conf/prompts/islands"
    assert cfg.evolver.islands.seed_organisms_per_island == 5
    assert cfg.evolver.islands.max_organisms_per_island == 5
    assert cfg.evolver.reproduction.offspring_per_generation == 10
    assert cfg.evolver.reproduction.operator_selection_strategy == "deterministic"
    assert cfg.evolver.prompts.implementation_system == "conf/prompts/implementation/system.txt"
    assert cfg.evolver.prompts.implementation_user == "conf/prompts/implementation/user.txt"
    assert cfg.evolver.prompts.implementation_template == "conf/prompts/implementation/template.txt"
    assert cfg.evolver.reproduction.operator_weights.within_island_crossover == 1.0
    assert cfg.evolver.operators.mutation.gene_removal_probability == 0.2
    assert cfg.evolver.operators.crossover.primary_parent_gene_inheritance_probability == 0.7
    assert cfg.evolver.phases.simple.eval_mode == "smoke"
    assert cfg.evolver.phases.simple.timeout_sec_per_eval == 7200
    assert cfg.evolver.phases.great_filter.eval_mode == "full"
    assert cfg.evolver.phases.great_filter.timeout_sec_per_eval == 7200
