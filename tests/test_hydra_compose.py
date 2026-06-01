"""Tests for Hydra config composition."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


ROOT = Path(__file__).resolve().parents[1]

# CO-Bench is driven by ONE preset (config_co-bench); the task is selected by
# the field experiments.co_bench.CO_BENCH_TASK. Map each UPPER-CASE identifier
# to its (dataset task string, lowercase slug).
CO_BENCH_TASKS = {
    "TSP": ("Travelling salesman problem", "tsp"),
    "BIN_PACKING_1D": ("Bin packing - one-dimensional", "bin_packing_1d"),
    "MULTI_KNAPSACK": ("Multidimensional knapsack problem", "multi_knapsack"),
    "SET_COVERING": ("Set covering", "set_covering"),
    "GRAPH_COLOURING": ("Graph colouring", "graph_colouring"),
    "JOB_SHOP": ("Job shop scheduling", "job_shop"),
}


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
    assert cfg.evolver.islands.mode == "from_seed"
    assert cfg.evolver.islands.seeds_per_island == 5
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
                # PHASE A speedup (2026-05-21): qwen35_122b has its design
                # stage tuned down to think=low / 6144 tokens (was medium /
                # 12288) because design was 51% of wall-time. The yaml uses
                # design as the &qwen_stage_defaults anchor so repair and
                # novelty_check inherit max_output_tokens=6144 too — fine,
                # neither needs more headroom. Other ollama routes keep the
                # canonical medium-thinking design budget.
                if route_name == "ollama_qwen35_122b":
                    assert route_cfg.stage_options["design"]["think"] == "low"
                    assert route_cfg.stage_options["design"]["max_output_tokens"] == 6144
                    assert route_cfg.stage_options["implementation"]["think"] is False
                    assert route_cfg.stage_options["implementation"]["max_output_tokens"] == 8192
                    assert route_cfg.stage_options["repair"]["think"] == "low"
                    assert route_cfg.stage_options["repair"]["max_output_tokens"] == 6144
                    assert route_cfg.stage_options["novelty_check"]["think"] == "low"
                    assert route_cfg.stage_options["novelty_check"]["max_output_tokens"] == 6144
                else:
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
    assert set(cfg.api_platforms.keys()) == {
        "ollama_gemma4_31b",
        "ollama_qwen35_122b",
    }
    assert cfg.experiments.unit_square_26._target_ == "experiments.circle_packing_shinka.unit_square_26.UnitSquare26CirclePackingExperiment"
    assert cfg.experiments.unit_square_26.need_cuda is False
    assert cfg.evolver.prompts.project_context == "conf/experiments/circle_packing_shinka/prompts/shared/project_context.txt"
    assert cfg.evolver.phases.simple.experiments == ["unit_square_26"]
    assert cfg.evolver.phases.great_filter.enabled is False
    assert cfg.resources.evaluation.gpu_ranks == []
    assert cfg.resources.evaluation.cpu_parallel_jobs == 20
    assert cfg.paths.ollama_cache_root == "./ollama_cache"
    assert cfg.evolver.max_generations == 150
    assert cfg.evolver.max_organism_creations is False
    # Optional per-model token-budget stop: present and disabled by default.
    assert set(cfg.evolver.max_tokens_per_model.keys()) == {
        "ollama_gemma4_31b",
        "ollama_qwen35_122b",
    }
    assert cfg.evolver.max_tokens_per_model.ollama_gemma4_31b is False
    assert cfg.evolver.max_tokens_per_model.ollama_qwen35_122b is False
    assert cfg.mode == "evolve"
    assert "organism_dir" not in cfg
    # PHASE A retry tightening (mirrors awtf2025_heuristic): retries cut
    # to 1 / 2 / 0 to stop cascading retries from burning wall-time.
    assert cfg.evolver.creation.max_attempts_to_create_organism == 1
    assert cfg.evolver.creation.max_attempts_to_repair_organism_after_error == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection == 0
    assert cfg.evolver.creation.max_parallel_organisms == 20
    assert cfg.evolver.prompts.genome_schema == "conf/experiments/circle_packing_shinka/prompts/shared/genome_schema.txt"
    assert cfg.evolver.reproduction.selection_score.mode == "weighted_sum"
    assert cfg.evolver.reproduction.selection_score.normalize_weights is True
    assert cfg.evolver.reproduction.selection_score.weights.simple_score == 1.0
    assert cfg.evolver.reproduction.selection_score.weights.inheritance_fitness == 0.0
    assert cfg.evolver.islands.seeds_per_island == 5
    assert cfg.evolver.islands.max_organisms_per_island == 10
    assert cfg.evolver.reproduction.species_sampling.strategy == "weighted_rule"
    assert cfg.evolver.reproduction.species_sampling.weighted_rule_lambda == 1.0
    assert cfg.evolver.reproduction.species_sampling.mutation_softmax_temperature == 1.0
    assert cfg.evolver.reproduction.species_sampling.within_island_crossover_softmax_temperature == 1.0
    assert cfg.evolver.reproduction.species_sampling.inter_island_crossover_softmax_temperature == 1.0
    assert "top_k_per_island" not in cfg.evolver.phases.simple
    assert cfg.evolver.phases.great_filter.top_h_per_island == 5
    assert cfg.evolver.reproduction.offspring_per_generation == 6
    # Model setup mirrors awtf2025_heuristic (atcoder): qwen 122B drives the
    # creative stages, gemma 31B the validator stages; pipelines drive route
    # selection so only gemma remains in route_weights.
    assert set(cfg.evolver.llm.route_weights.keys()) == {"ollama_gemma4_31b"}
    assert cfg.evolver.llm.route_weights.ollama_gemma4_31b == 1.0
    assert cfg.api_platforms.ollama_gemma4_31b.max_concurrency == 3
    assert cfg.api_platforms.ollama_qwen35_122b.max_concurrency == 4
    assert [p.id for p in cfg.evolver.llm.pipelines] == ["qwen_creative_gemma_check"]

    qwen122_route = instantiate(cfg.api_platforms.ollama_qwen35_122b, _recursive_=False)
    gemma_route = instantiate(cfg.api_platforms.ollama_gemma4_31b, _recursive_=False)
    assert qwen122_route.route_id == "ollama_qwen35_122b"
    assert qwen122_route.provider_model_id == "qwen3.5:122b"
    assert qwen122_route.gpu_ranks == [0, 1, 2, 3, 4, 5]
    assert qwen122_route.gpu_rank_groups == [[0, 1, 2], [3, 4, 5]]
    assert qwen122_route.max_output_tokens == 12288
    assert gemma_route.route_id == "ollama_gemma4_31b"
    assert gemma_route.provider_model_id == "gemma4:31b"
    assert gemma_route.gpu_ranks == [6, 7]
    assert gemma_route.gpu_rank_groups == [[6], [7]]
    assert gemma_route.stage_options["design"]["think"] is False
    assert gemma_route.stage_options["design"]["max_output_tokens"] == 9000
    assert gemma_route.stage_options["novelty_check"]["think"] is False
    assert gemma_route.stage_options["novelty_check"]["max_output_tokens"] == 9000


def test_circle_packing_canonical_preset_accepts_standalone_validation_overrides() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config_circle_packing_shinka",
            overrides=["mode=run", "+organism_dir=/tmp/organism"],
        )

    assert cfg.mode == "run"
    assert cfg.organism_dir == "/tmp/organism"
    assert cfg.evolver.creation.max_attempts_to_create_organism == 1
    assert cfg.evolver.creation.max_attempts_to_repair_organism_after_error == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection == 0
    assert cfg.evolver.creation.max_parallel_organisms == 20


def test_awtf2025_heuristic_config_composes() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config_awtf2025_heuristic")

    assert set(cfg.experiments.keys()) == {"group_commands_and_wall_planning"}
    assert "safety" not in cfg
    assert set(cfg.api_platforms.keys()) == {
        "ollama_gemma4_31b",
        "ollama_qwen35_122b",
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
    # Optional per-model token-budget stop: present and disabled by default.
    assert set(cfg.evolver.max_tokens_per_model.keys()) == {
        "ollama_gemma4_31b",
        "ollama_qwen35_122b",
    }
    assert cfg.evolver.max_tokens_per_model.ollama_gemma4_31b is False
    assert cfg.evolver.max_tokens_per_model.ollama_qwen35_122b is False
    assert cfg.mode == "evolve"
    assert "organism_dir" not in cfg
    # PHASE A speedup (2026-05-21): retry budgets tightened to (1, 2, 0)
    # because cascading retries cost wall-time without lifting per-organism
    # success rate. See plan file users-artemon-downloads-awtf2025-heuris-merry-stonebraker.md
    assert cfg.evolver.creation.max_attempts_to_create_organism == 1
    assert cfg.evolver.creation.max_attempts_to_repair_organism_after_error == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection == 0
    assert cfg.evolver.creation.max_parallel_organisms == 8
    assert cfg.evolver.islands.seeds_per_island == 5
    assert cfg.evolver.islands.max_organisms_per_island == 10
    assert cfg.evolver.prompts.genome_schema == "conf/experiments/awtf2025_heuristic/prompts/shared/genome_schema.txt"
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
    # PHASE A speedup (2026-05-21): offspring 10 → 6 to match the new
    # max_parallel_organisms=8 broker ceiling so every offspring runs in
    # parallel instead of queueing.
    assert cfg.evolver.reproduction.offspring_per_generation == 6
    # The active route bank for awtf2025_heuristic is qwen 122B (creative
    # stages) + gemma 31B (validator stages). The legacy nemotron /
    # qwen35_35b routes were retired when we switched to the two-platform
    # pipeline bandit; only the gemma route remains in route_weights
    # because pipelines now drive route selection, not route_weights.
    assert set(cfg.evolver.llm.route_weights.keys()) == {"ollama_gemma4_31b"}
    assert cfg.evolver.llm.route_weights.ollama_gemma4_31b == 1.0
    assert cfg.api_platforms.ollama_gemma4_31b.max_concurrency == 3
    assert cfg.api_platforms.ollama_qwen35_122b.max_concurrency == 4
    assert list(cfg.experiments.group_commands_and_wall_planning.validation.smoke_case_ids) == [0, 1, 2, 3, 4]
    assert len(cfg.experiments.group_commands_and_wall_planning.validation.full_case_ids) == 100
    assert cfg.experiments.group_commands_and_wall_planning.validation.aggregate == "mean"

    qwen122_route = instantiate(cfg.api_platforms.ollama_qwen35_122b, _recursive_=False)
    gemma_route = instantiate(cfg.api_platforms.ollama_gemma4_31b, _recursive_=False)
    assert qwen122_route.route_id == "ollama_qwen35_122b"
    assert qwen122_route.provider_model_id == "qwen3.5:122b"
    assert qwen122_route.gpu_ranks == [0, 1, 2, 3, 4, 5]
    assert qwen122_route.gpu_rank_groups == [[0, 1, 2], [3, 4, 5]]
    assert qwen122_route.max_output_tokens == 12288
    # PHASE A: design tuned to think=low / 6144 tokens.
    assert qwen122_route.stage_options["design"]["think"] == "low"
    assert qwen122_route.stage_options["design"]["max_output_tokens"] == 6144
    assert qwen122_route.stage_options["implementation"]["think"] is False
    assert qwen122_route.stage_options["implementation"]["max_output_tokens"] == 8192
    assert gemma_route.route_id == "ollama_gemma4_31b"
    assert gemma_route.provider_model_id == "gemma4:31b"
    assert gemma_route.gpu_ranks == [6, 7]
    assert gemma_route.gpu_rank_groups == [[6], [7]]
    assert gemma_route.max_output_tokens == 12288
    assert gemma_route.stage_options["design"]["think"] is False
    assert gemma_route.stage_options["design"]["max_output_tokens"] == 9000
    assert gemma_route.stage_options["implementation"]["think"] is False
    assert gemma_route.stage_options["implementation"]["max_output_tokens"] == 9000
    assert gemma_route.stage_options["repair"]["think"] is False
    assert gemma_route.stage_options["repair"]["max_output_tokens"] == 9000
    assert gemma_route.stage_options["novelty_check"]["think"] is False
    assert gemma_route.stage_options["novelty_check"]["max_output_tokens"] == 9000


def test_awtf2025_canonical_preset_accepts_standalone_validation_overrides() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config_awtf2025_heuristic",
            overrides=["mode=run", "+organism_dir=/tmp/organism"],
        )

    assert cfg.mode == "run"
    assert cfg.organism_dir == "/tmp/organism"
    # PHASE A speedup (2026-05-21): retry budgets tightened further to (1, 2, 0)
    # after measuring that the prior (2, 2, 1) still spent 30%+ of wall-time on
    # retries with no measurable score lift. max_parallel_organisms 15 → 8 to
    # match the actual broker concurrency ceiling (qwen 122B 2 instances ×
    # 4 max_concurrency = 8). See plan file
    # users-artemon-downloads-awtf2025-heuris-merry-stonebraker.md.
    assert cfg.evolver.creation.max_attempts_to_create_organism == 1
    assert cfg.evolver.creation.max_attempts_to_repair_organism_after_error == 2
    assert cfg.evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection == 0
    assert cfg.evolver.creation.max_parallel_organisms == 8


def test_co_bench_config_defaults_to_tsp() -> None:
    # ONE preset, no per-task presets: composing config_co-bench with no
    # override selects TSP (the default CO_BENCH_TASK).
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config_co-bench")

    assert set(cfg.experiments.keys()) == {"co_bench"}
    assert cfg.experiments.co_bench.CO_BENCH_TASK == "TSP"
    assert cfg.experiments.co_bench.co_bench_task == "Travelling salesman problem"
    assert cfg.experiments.co_bench.name == "tsp"
    assert str(cfg.evolver.islands.seed_program_path).endswith(
        "co_bench/_baselines/shinka/tsp/initial_program.py"
    )
    assert str(cfg.evolver.prompts.project_context).endswith(
        "co-bench/prompts/tsp/project_context.txt"
    )


@pytest.mark.parametrize("identifier", sorted(CO_BENCH_TASKS))
def test_co_bench_config_composes(identifier: str) -> None:
    # ONE top-level preset selects ANY task via the CO_BENCH_TASK field; there
    # are NO per-task presets and NO conf/runs/ tree.
    dataset_task, slug = CO_BENCH_TASKS[identifier]
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config_co-bench",
            overrides=[f"experiments.co_bench.CO_BENCH_TASK={identifier}"],
        )

    # Exactly one experiment, keyed by the static experiment key ``co_bench``.
    assert set(cfg.experiments.keys()) == {"co_bench"}
    exp = cfg.experiments.co_bench
    assert exp._target_ == "experiments.co_bench._runtime.runner.CoBenchExperimentEvaluator"
    # co_bench_task / name resolve via node-relative interpolation from the
    # registry and survive per-experiment config detachment.
    assert exp.co_bench_task == dataset_task
    assert exp.name == slug
    assert exp.need_cuda is False

    # Seed + per-task project context are addressed by the selected slug.
    assert str(cfg.evolver.islands.seed_program_path).endswith(
        f"co_bench/_baselines/shinka/{slug}/initial_program.py"
    )
    assert str(cfg.evolver.prompts.project_context).endswith(
        f"co-bench/prompts/{slug}/project_context.txt"
    )

    # All OTHER prompts live under the SHARED co-bench prompt tree (not per-slug).
    shared_prefix = "conf/experiments/co-bench/prompts/"
    assert cfg.evolver.prompts.genome_schema == f"{shared_prefix}shared/genome_schema.txt"
    assert cfg.evolver.prompts.implementation_template == f"{shared_prefix}shared/template.txt"
    for key, value in cfg.evolver.prompts.items():
        if key == "project_context" or not isinstance(value, str):
            continue
        assert value.startswith(shared_prefix), f"prompt {key} not under shared tree: {value}"
        # Only project_context is per-slug; no other prompt path embeds the slug.
        assert f"/prompts/{slug}/" not in value, f"prompt {key} unexpectedly slug-scoped: {value}"

    # Phases reference the static experiment KEY, not the slug.
    assert cfg.evolver.phases.simple.experiments == ["co_bench"]
