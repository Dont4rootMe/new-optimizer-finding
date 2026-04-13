"""Tests for Hydra config composition."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


ROOT = Path(__file__).resolve().parents[1]


def test_hydra_compose_config_and_experiments() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config_optimization_survey")

    assert "experiments" in cfg
    assert "evolver" in cfg
    assert "api_platforms" in cfg
    assert "safety" not in cfg
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
        assert "need_cuda" in cfg.experiments[exp_name]
        assert "run_validation" in cfg.experiments[exp_name]
        assert "normalization" in cfg.experiments[exp_name]
        assert "quality_ref" not in cfg.experiments[exp_name].normalization
        assert "steps_ref" not in cfg.experiments[exp_name].normalization
        assert (
            cfg.experiments[exp_name]._target_
            == "experiments.optimization_survey._runtime.runner.OptimizationSurveyExperimentEvaluator"
        )
        assert "experiment_target" in cfg.experiments[exp_name]
        assert (
            str(cfg.experiments[exp_name].baseline.profile_path).endswith(
                f"/optimization_survey/{exp_name}/baseline.json"
            )
        )

    assert "selection_strategy" not in cfg.evolver
    assert "eval_experiments" not in cfg.evolver
    assert "evaluation" not in cfg.evolver
    assert "provider" not in cfg.evolver.llm
    assert "model" not in cfg.evolver.llm
    assert "organisms_per_island" not in cfg.evolver.islands
    assert "inter_island_crossover_rate" not in cfg.evolver.islands
    assert "probability" not in cfg.evolver.operators.mutation
    assert "parent_sampling" not in cfg.evolver.operators.mutation
    assert "parent_sampling" not in cfg.evolver.operators.crossover
    assert "max_evaluation_jobs" not in cfg.evolver
    assert cfg.evolver.max_generations == 100
    assert cfg.mode == "evolve"
    assert "organism_dir" not in cfg
    assert cfg.resources.evaluation.gpu_ranks == [0]
    assert cfg.resources.evaluation.cpu_parallel_jobs == 4
    assert cfg.evolver.islands.dir == "conf/experiments/optimization_survey/prompts/islands"
    assert cfg.evolver.islands.seed_organisms_per_island == 5
    assert cfg.evolver.islands.max_organisms_per_island == 5
    assert cfg.evolver.reproduction.offspring_per_generation == 10
    assert cfg.evolver.reproduction.operator_selection_strategy == "deterministic"
    assert cfg.evolver.creation.max_attempts_to_create_organism == 3
    assert cfg.evolver.creation.max_attempts_to_repair_organism_after_error == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection == 2
    assert cfg.evolver.creation.max_parallel_organisms == 4
    assert cfg.evolver.llm.selection_strategy == "random"
    assert cfg.evolver.llm.route_weights.mock == 1.0
    assert cfg.api_platforms.mock._target_ == "api_platforms.mock.platform.build_platform"
    assert (
        cfg.evolver.prompts.implementation_system
        == "conf/experiments/optimization_survey/prompts/implementation/system.txt"
    )
    assert (
        cfg.evolver.prompts.implementation_user
        == "conf/experiments/optimization_survey/prompts/implementation/user.txt"
    )
    assert (
        cfg.evolver.prompts.implementation_template
        == "conf/experiments/optimization_survey/prompts/implementation/template.txt"
    )
    assert cfg.evolver.reproduction.operator_weights.within_island_crossover == 1.0
    assert cfg.evolver.operators.mutation.gene_removal_probability == 0.2
    assert cfg.evolver.operators.crossover.primary_parent_gene_inheritance_probability == 0.7
    assert cfg.evolver.phases.simple.eval_mode == "smoke"
    assert cfg.evolver.phases.simple.timeout_sec_per_eval == 7200
    assert cfg.evolver.phases.great_filter.eval_mode == "full"
    assert cfg.evolver.phases.great_filter.timeout_sec_per_eval == 7200


def test_optimization_survey_canonical_preset_accepts_standalone_validation_overrides() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config_optimization_survey",
            overrides=["mode=run", "+organism_dir=/tmp/organism"],
        )

    assert cfg.mode == "run"
    assert cfg.organism_dir == "/tmp/organism"
    assert cfg.evolver.creation.max_attempts_to_create_organism == 3
    assert cfg.evolver.creation.max_attempts_to_repair_organism_after_error == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection == 2
    assert cfg.evolver.creation.max_parallel_organisms == 4


def test_all_shipped_api_platform_route_configs_instantiate() -> None:
    conf_dir = ROOT / "conf"
    route_names = {
        "mock",
        "gpt_5_4",
        "gpt_5_4_mini",
        "gpt_5_4_nano",
        "claude_opus_46",
        "claude_sonnet_46",
        "claude_haiku_45",
        "ollama_gemma4_26b",
        "ollama_qwen35_27b",
        "gemma_4_26b_a4b_it",
        "gemma_4_31b_it",
        "qwen35_27b_claude46_opus_distilled",
        "qwen35_27b",
        "qwen35_35b_a3b",
    }
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        for route_name in route_names:
            overrides = [] if route_name == "mock" else [f"+api_platforms@api_platforms.{route_name}={route_name}"]
            cfg = compose(config_name="config_optimization_survey", overrides=overrides)
            route_cfg = instantiate(cfg.api_platforms[route_name], _recursive_=False)
            assert route_cfg.route_id == route_name


def test_circle_packing_shinka_config_composes() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config_circle_packing_shinka")

    assert set(cfg.experiments.keys()) == {"unit_square_26"}
    assert "safety" not in cfg
    assert set(cfg.api_platforms.keys()) == {"ollama_qwen35_27b", "ollama_gemma4_26b"}
    assert cfg.experiments.unit_square_26._target_ == "experiments.circle_packing_shinka.unit_square_26.UnitSquare26CirclePackingExperiment"
    assert cfg.experiments.unit_square_26.need_cuda is False
    assert cfg.evolver.prompts.project_context == "conf/experiments/circle_packing_shinka/prompts/shared/project_context.txt"
    assert cfg.evolver.phases.simple.experiments == ["unit_square_26"]
    assert cfg.evolver.phases.great_filter.enabled is True
    assert cfg.resources.evaluation.gpu_ranks == []
    assert cfg.resources.evaluation.cpu_parallel_jobs == 5
    assert cfg.paths.ollama_cache_root == "./ollama_cache"
    assert cfg.evolver.max_generations == 150
    assert cfg.mode == "evolve"
    assert "organism_dir" not in cfg
    assert cfg.evolver.creation.max_attempts_to_create_organism == 3
    assert cfg.evolver.creation.max_attempts_to_repair_organism_after_error == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection == 2
    assert cfg.evolver.creation.max_parallel_organisms == 5
    assert cfg.evolver.islands.seed_organisms_per_island == 3
    assert cfg.evolver.islands.max_organisms_per_island == 10
    assert "top_k_per_island" not in cfg.evolver.phases.simple
    assert cfg.evolver.phases.great_filter.top_h_per_island == 5
    assert cfg.evolver.reproduction.offspring_per_generation == 5
    assert set(cfg.evolver.llm.route_weights.keys()) == {"ollama_qwen35_27b", "ollama_gemma4_26b"}
    assert cfg.evolver.llm.route_weights.ollama_qwen35_27b == 1.0
    assert cfg.evolver.llm.route_weights.ollama_gemma4_26b == 1.0
    assert cfg.api_platforms.ollama_qwen35_27b.max_concurrency == 5
    assert cfg.api_platforms.ollama_gemma4_26b.max_concurrency == 5

    qwen_route = instantiate(cfg.api_platforms.ollama_qwen35_27b, _recursive_=False)
    gemma_route = instantiate(cfg.api_platforms.ollama_gemma4_26b, _recursive_=False)
    assert qwen_route.route_id == "ollama_qwen35_27b"
    assert qwen_route.provider_model_id == "qwen3.5:27b"
    assert qwen_route.base_url == "http://127.0.0.1:12435/api"
    assert qwen_route.gpu_ranks == [1]
    assert qwen_route.stage_options["design"]["think"] == "medium"
    assert qwen_route.stage_options["implementation"]["max_output_tokens"] == 4096
    assert qwen_route.stage_options["novelty_check"]["temperature"] == 0.1
    assert qwen_route.request_options["num_ctx"] == 65536
    assert gemma_route.route_id == "ollama_gemma4_26b"
    assert gemma_route.provider_model_id == "gemma4:26b"
    assert gemma_route.base_url == "http://127.0.0.1:12434/api"
    assert gemma_route.gpu_ranks == [0]
    assert gemma_route.stage_options["implementation"]["think"] == "medium"
    assert gemma_route.stage_options["repair"]["top_k"] == 64
    assert gemma_route.stage_options["novelty_check"]["top_k"] == 64


def test_circle_packing_canonical_preset_accepts_standalone_validation_overrides() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config_circle_packing_shinka",
            overrides=["mode=run", "+organism_dir=/tmp/organism"],
        )

    assert cfg.mode == "run"
    assert cfg.organism_dir == "/tmp/organism"
    assert cfg.evolver.creation.max_attempts_to_create_organism == 3
    assert cfg.evolver.creation.max_attempts_to_repair_organism_after_error == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection == 2
    assert cfg.evolver.creation.max_parallel_organisms == 5
