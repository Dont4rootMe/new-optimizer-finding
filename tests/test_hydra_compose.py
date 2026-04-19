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
    assert cfg.evolver.reproduction.species_sampling.strategy == "weighted_rule"
    assert cfg.evolver.reproduction.species_sampling.weighted_rule_lambda == 1.0
    assert cfg.evolver.reproduction.species_sampling.mutation_softmax_temperature == 1.0
    assert cfg.evolver.reproduction.species_sampling.within_island_crossover_softmax_temperature == 1.0
    assert cfg.evolver.reproduction.species_sampling.inter_island_crossover_softmax_temperature == 1.0
    assert cfg.evolver.creation.max_attempts_to_create_organism == 3
    assert cfg.evolver.creation.max_attempts_to_repair_organism_after_error == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_compatibility_rejection == 3
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
        == "conf/experiments/optimization_survey/prompts/shared/template.txt"
    )
    assert cfg.evolver.prompts.genome_schema == "conf/experiments/optimization_survey/prompts/shared/genome_schema.txt"
    assert cfg.evolver.prompts.compatibility_seed_system.endswith("/compatibility/seed/system.txt")
    assert cfg.evolver.prompts.compatibility_mutation_user.endswith("/compatibility/mutation/user.txt")
    assert cfg.evolver.prompts.compatibility_crossover_system.endswith("/compatibility/crossover/system.txt")
    assert cfg.evolver.reproduction.selection_score.mode == "weighted_sum"
    assert cfg.evolver.reproduction.selection_score.normalize_weights is True
    assert cfg.evolver.reproduction.selection_score.weights.simple_score == 1.0
    assert cfg.evolver.reproduction.selection_score.weights.inheritance_fitness == 0.0
    assert cfg.evolver.reproduction.operator_weights.within_island_crossover == 1.0
    assert cfg.evolver.operators.mutation.gene_removal_probability == 0.2
    assert "parent_selection_softmax_temperature" not in cfg.evolver.operators.mutation
    assert cfg.evolver.operators.crossover.primary_parent_gene_inheritance_probability == 0.7
    assert "parent_selection_softmax_temperature" not in cfg.evolver.operators.crossover
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
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_compatibility_rejection == 3
    assert cfg.evolver.creation.max_parallel_organisms == 4


def test_all_shipped_api_platform_route_configs_instantiate() -> None:
    conf_dir = ROOT / "conf"
    route_expectations = {
        "mock": "mock-model",
        "gpt_5_4": "gpt-5.4",
        "gpt_5_4_mini": "gpt-5.4-mini",
        "gpt_5_4_nano": "gpt-5.4-nano",
        "claude_opus_46": "claude-opus-4.6",
        "claude_sonnet_46": "claude-sonnet-4.6",
        "claude_haiku_45": "claude-haiku-4.5",
        "ollama_gemma4_26b": "gemma4:26b",
        "ollama_gemma4_31b": "gemma4:31b",
        "ollama_nemotron_cascade_2_30b": "nemotron-cascade-2:30b",
        "ollama_qwen35_27b": "qwen3.5:27b",
        "ollama_qwen35_35b": "qwen3.5:35b",
        "ollama_qwen35_122b": "qwen3.5:122b",
        "gemma_4_26b_a4b_it": "google/gemma-4-26B-A4B-it",
        "gemma_4_31b_it": "google/gemma-4-31B-it",
        "qwen35_27b_claude46_opus_distilled": "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
        "qwen35_27b": "Qwen/Qwen3.5-27B",
        "qwen35_35b_a3b": "Qwen/Qwen3.5-35B-A3B",
    }
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        for route_name, provider_model_id in route_expectations.items():
            overrides = [] if route_name == "mock" else [f"+api_platforms@api_platforms.{route_name}={route_name}"]
            cfg = compose(config_name="config_optimization_survey", overrides=overrides)
            route_cfg = instantiate(cfg.api_platforms[route_name], _recursive_=False)
            assert route_cfg.route_id == route_name
            assert route_cfg.provider_model_id == provider_model_id
            if route_name.startswith("ollama_"):
                assert route_cfg.max_output_tokens == 12288
                assert route_cfg.think == "low"
                assert route_cfg.stage_options["design"]["think"] == "medium"
                assert route_cfg.stage_options["design"]["max_output_tokens"] == 12288
                assert route_cfg.stage_options["implementation"]["think"] == "low"
                assert route_cfg.stage_options["implementation"]["max_output_tokens"] == 12288
                assert route_cfg.stage_options["repair"]["think"] == "low"
                assert route_cfg.stage_options["repair"]["max_output_tokens"] == 12288
                assert route_cfg.stage_options["novelty_check"]["think"] == "low"
                assert route_cfg.stage_options["novelty_check"]["max_output_tokens"] == 12288


def test_circle_packing_shinka_config_composes() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config_circle_packing_shinka")

    assert set(cfg.experiments.keys()) == {"unit_square_26"}
    assert "safety" not in cfg
    assert "ollama_qwen35_122b" in cfg.api_platforms
    assert cfg.experiments.unit_square_26._target_ == "experiments.circle_packing_shinka.unit_square_26.UnitSquare26CirclePackingExperiment"
    assert cfg.experiments.unit_square_26.need_cuda is False
    assert cfg.evolver.prompts.project_context == "conf/experiments/circle_packing_shinka/prompts/shared/project_context.txt"
    assert cfg.evolver.phases.simple.experiments == ["unit_square_26"]
    assert cfg.evolver.phases.great_filter.enabled is False
    assert cfg.resources.evaluation.gpu_ranks == []
    assert cfg.resources.evaluation.cpu_parallel_jobs == 20
    assert cfg.paths.ollama_cache_root == "./ollama_cache"
    assert cfg.evolver.max_generations == 50
    assert cfg.mode == "evolve"
    assert "organism_dir" not in cfg
    assert cfg.evolver.creation.max_attempts_to_create_organism == 3
    assert cfg.evolver.creation.max_attempts_to_repair_organism_after_error == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_compatibility_rejection == 3
    assert cfg.evolver.creation.max_parallel_organisms == 20
    assert cfg.evolver.prompts.genome_schema == "conf/experiments/circle_packing_shinka/prompts/shared/genome_schema.txt"
    assert cfg.evolver.prompts.compatibility_seed_system.endswith("/compatibility/seed/system.txt")
    assert cfg.evolver.prompts.compatibility_mutation_user.endswith("/compatibility/mutation/user.txt")
    assert cfg.evolver.prompts.compatibility_crossover_system.endswith("/compatibility/crossover/system.txt")
    assert cfg.evolver.reproduction.selection_score.mode == "weighted_sum"
    assert cfg.evolver.reproduction.selection_score.normalize_weights is True
    assert cfg.evolver.reproduction.selection_score.weights.simple_score == 1.0
    assert cfg.evolver.reproduction.selection_score.weights.inheritance_fitness == 0.0
    assert cfg.evolver.islands.seed_organisms_per_island == 8
    assert cfg.evolver.islands.max_organisms_per_island == 8
    assert cfg.evolver.reproduction.species_sampling.strategy == "weighted_rule"
    assert cfg.evolver.reproduction.species_sampling.weighted_rule_lambda == 1.0
    assert cfg.evolver.reproduction.species_sampling.mutation_softmax_temperature == 1.0
    assert cfg.evolver.reproduction.species_sampling.within_island_crossover_softmax_temperature == 1.0
    assert cfg.evolver.reproduction.species_sampling.inter_island_crossover_softmax_temperature == 1.0
    assert "top_k_per_island" not in cfg.evolver.phases.simple
    assert cfg.evolver.phases.great_filter.top_h_per_island == 5
    assert cfg.evolver.reproduction.offspring_per_generation == 10
    assert set(cfg.evolver.llm.route_weights.keys()) == {"ollama_qwen35_122b"}
    assert cfg.evolver.llm.route_weights.ollama_qwen35_122b == 1.0
    assert cfg.api_platforms.ollama_qwen35_122b.max_concurrency == 4

    qwen_route = instantiate(cfg.api_platforms.ollama_qwen35_122b, _recursive_=False)
    assert qwen_route.route_id == "ollama_qwen35_122b"
    assert qwen_route.provider_model_id == "qwen3.5:122b"
    assert qwen_route.base_url == "http://127.0.0.1:11434/api"
    assert qwen_route.gpu_ranks == []
    assert qwen_route.gpu_rank_groups == []
    assert qwen_route.max_concurrency == 4
    assert qwen_route.max_output_tokens == 12288
    assert qwen_route.stage_options["design"]["think"] == "medium"
    assert qwen_route.stage_options["design"]["temperature"] == 1.0
    assert qwen_route.stage_options["design"]["max_output_tokens"] == 12288
    assert qwen_route.stage_options["implementation"]["think"] == "low"
    assert qwen_route.stage_options["implementation"]["temperature"] == 0.4
    assert qwen_route.stage_options["implementation"]["max_output_tokens"] == 12288
    assert qwen_route.stage_options["repair"]["think"] == "low"
    assert qwen_route.stage_options["repair"]["max_output_tokens"] == 12288
    assert qwen_route.stage_options["novelty_check"]["think"] == "low"
    assert qwen_route.stage_options["novelty_check"]["temperature"] == 0.1
    assert qwen_route.stage_options["novelty_check"]["max_output_tokens"] == 12288
    assert qwen_route.request_options["num_ctx"] == 65536


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
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_compatibility_rejection == 3
    assert cfg.evolver.creation.max_parallel_organisms == 20


def test_awtf2025_heuristic_config_composes() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config_awtf2025_heuristic")

    assert set(cfg.experiments.keys()) == {"group_commands_and_wall_planning"}
    assert "safety" not in cfg
    assert set(cfg.api_platforms.keys()) == {
        "ollama_nemotron_cascade_2_30b",
        "ollama_qwen35_35b",
        "ollama_gemma4_31b",
    }
    assert (
        cfg.experiments.group_commands_and_wall_planning._target_
        == "experiments.awtf2025_heuristic.group_commands_and_wall_planning.GroupCommandsAndWallPlanningExperiment"
    )
    assert cfg.experiments.group_commands_and_wall_planning.need_cuda is False
    assert cfg.evolver.prompts.project_context == "conf/experiments/awtf2025_heuristic/prompts/shared/project_context.txt"
    assert cfg.evolver.phases.simple.experiments == ["group_commands_and_wall_planning"]
    assert cfg.evolver.phases.great_filter.enabled is False
    assert cfg.resources.evaluation.gpu_ranks == []
    assert cfg.resources.evaluation.cpu_parallel_jobs == 25
    assert cfg.paths.ollama_cache_root == "./ollama_cache"
    assert cfg.evolver.max_generations == 150
    assert cfg.mode == "evolve"
    assert "organism_dir" not in cfg
    assert cfg.evolver.creation.max_attempts_to_create_organism == 3
    assert cfg.evolver.creation.max_attempts_to_repair_organism_after_error == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_compatibility_rejection == 3
    assert cfg.evolver.creation.max_parallel_organisms == 15
    assert cfg.evolver.islands.seed_organisms_per_island == 10
    assert cfg.evolver.islands.max_organisms_per_island == 30
    assert cfg.evolver.prompts.genome_schema == "conf/experiments/awtf2025_heuristic/prompts/shared/genome_schema.txt"
    assert cfg.evolver.prompts.compatibility_seed_system.endswith("/compatibility/seed/system.txt")
    assert cfg.evolver.prompts.compatibility_mutation_user.endswith("/compatibility/mutation/user.txt")
    assert cfg.evolver.prompts.compatibility_crossover_system.endswith("/compatibility/crossover/system.txt")
    assert cfg.evolver.reproduction.species_sampling.strategy == "weighted_rule"
    assert cfg.evolver.reproduction.species_sampling.weighted_rule_lambda == 1.0
    assert cfg.evolver.reproduction.species_sampling.mutation_softmax_temperature == 1.0
    assert cfg.evolver.reproduction.species_sampling.within_island_crossover_softmax_temperature == 1.0
    assert cfg.evolver.reproduction.species_sampling.inter_island_crossover_softmax_temperature == 1.0
    assert cfg.evolver.reproduction.selection_score.mode == "weighted_sum"
    assert cfg.evolver.reproduction.selection_score.normalize_weights is True
    assert cfg.evolver.reproduction.selection_score.weights.simple_score == 1.0
    assert cfg.evolver.reproduction.selection_score.weights.inheritance_fitness == 0.0
    assert cfg.evolver.phases.simple.eval_mode == "full"
    assert "top_k_per_island" not in cfg.evolver.phases.simple
    assert cfg.evolver.phases.great_filter.eval_mode == "full"
    assert cfg.evolver.phases.great_filter.top_h_per_island == 5
    assert cfg.evolver.reproduction.offspring_per_generation == 15
    assert set(cfg.evolver.llm.route_weights.keys()) == {
        "ollama_nemotron_cascade_2_30b",
        "ollama_qwen35_35b",
        "ollama_gemma4_31b",
    }
    assert cfg.evolver.llm.route_weights.ollama_nemotron_cascade_2_30b == 1.0
    assert cfg.evolver.llm.route_weights.ollama_qwen35_35b == 1.0
    assert cfg.evolver.llm.route_weights.ollama_gemma4_31b == 1.0
    assert cfg.api_platforms.ollama_nemotron_cascade_2_30b.max_concurrency == 3
    assert cfg.api_platforms.ollama_qwen35_35b.max_concurrency == 3
    assert cfg.api_platforms.ollama_gemma4_31b.max_concurrency == 3
    assert list(cfg.experiments.group_commands_and_wall_planning.validation.smoke_case_ids) == [0, 1, 2, 3, 4]
    assert len(cfg.experiments.group_commands_and_wall_planning.validation.full_case_ids) == 100
    assert cfg.experiments.group_commands_and_wall_planning.validation.aggregate == "mean"

    nemotron_route = instantiate(cfg.api_platforms.ollama_nemotron_cascade_2_30b, _recursive_=False)
    qwen_route = instantiate(cfg.api_platforms.ollama_qwen35_35b, _recursive_=False)
    gemma_route = instantiate(cfg.api_platforms.ollama_gemma4_31b, _recursive_=False)
    assert nemotron_route.route_id == "ollama_nemotron_cascade_2_30b"
    assert nemotron_route.provider_model_id == "nemotron-cascade-2:30b"
    assert nemotron_route.base_url == "http://127.0.0.1:12439/api"
    assert nemotron_route.gpu_ranks == [5]
    assert nemotron_route.max_output_tokens == 12288
    assert nemotron_route.stage_options["design"]["think"] is False
    assert nemotron_route.stage_options["design"]["max_output_tokens"] == 6000
    assert nemotron_route.stage_options["implementation"]["think"] is False
    assert nemotron_route.stage_options["implementation"]["max_output_tokens"] == 6000
    assert nemotron_route.stage_options["implementation"]["temperature"] == 0.4
    assert nemotron_route.stage_options["repair"]["think"] is False
    assert nemotron_route.stage_options["repair"]["max_output_tokens"] == 6000
    assert nemotron_route.stage_options["novelty_check"]["think"] is False
    assert nemotron_route.stage_options["novelty_check"]["max_output_tokens"] == 6000
    assert nemotron_route.request_options["num_ctx"] == 65536
    assert qwen_route.route_id == "ollama_qwen35_35b"
    assert qwen_route.provider_model_id == "qwen3.5:35b"
    assert qwen_route.base_url == "http://127.0.0.1:12438/api"
    assert qwen_route.gpu_ranks == [4]
    assert qwen_route.max_output_tokens == 12288
    assert qwen_route.stage_options["design"]["think"] is False
    assert qwen_route.stage_options["design"]["max_output_tokens"] == 6000
    assert qwen_route.stage_options["implementation"]["max_output_tokens"] == 6000
    assert qwen_route.stage_options["implementation"]["think"] is False
    assert qwen_route.stage_options["implementation"]["temperature"] == 0.4
    assert qwen_route.stage_options["repair"]["think"] is False
    assert qwen_route.stage_options["repair"]["max_output_tokens"] == 6000
    assert qwen_route.stage_options["novelty_check"]["think"] is False
    assert qwen_route.stage_options["novelty_check"]["temperature"] == 0.1
    assert qwen_route.stage_options["novelty_check"]["max_output_tokens"] == 6000
    assert qwen_route.request_options["num_ctx"] == 65536
    assert gemma_route.route_id == "ollama_gemma4_31b"
    assert gemma_route.provider_model_id == "gemma4:31b"
    assert gemma_route.base_url == "http://127.0.0.1:12437/api"
    assert gemma_route.gpu_ranks == [3]
    assert gemma_route.max_output_tokens == 12288
    assert gemma_route.stage_options["design"]["think"] is False
    assert gemma_route.stage_options["design"]["max_output_tokens"] == 6000
    assert gemma_route.stage_options["implementation"]["think"] is False
    assert gemma_route.stage_options["implementation"]["max_output_tokens"] == 6000
    assert gemma_route.stage_options["implementation"]["temperature"] == 0.4
    assert gemma_route.stage_options["repair"]["think"] is False
    assert gemma_route.stage_options["repair"]["max_output_tokens"] == 6000
    assert gemma_route.stage_options["novelty_check"]["think"] is False
    assert gemma_route.stage_options["novelty_check"]["max_output_tokens"] == 6000
    assert gemma_route.stage_options["repair"]["top_k"] == 64
    assert gemma_route.stage_options["novelty_check"]["top_k"] == 64


def test_awtf2025_canonical_preset_accepts_standalone_validation_overrides() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config_awtf2025_heuristic",
            overrides=["mode=run", "+organism_dir=/tmp/organism"],
        )

    assert cfg.mode == "run"
    assert cfg.organism_dir == "/tmp/organism"
    assert cfg.evolver.creation.max_attempts_to_create_organism == 3
    assert cfg.evolver.creation.max_attempts_to_repair_organism_after_error == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_compatibility_rejection == 3
    assert cfg.evolver.creation.max_parallel_organisms == 15
