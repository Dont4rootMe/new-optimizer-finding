"""Tests for the pipeline-config feature (per-organism stage->route bundles)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.evolve.pipeline import (
    PIPELINE_STAGES,
    PipelineConfig,
    canonical_pipeline_stage,
    parse_pipelines,
    validate_pipeline_routes,
)


def test_canonical_stage_collapses_aliases() -> None:
    """Concrete stages (``design_attempt`` etc.) collapse onto the small
    canonical taxonomy that pipeline configs use."""

    assert canonical_pipeline_stage("design_rationalization") == "rationalization"
    assert canonical_pipeline_stage("design") == "design"
    assert canonical_pipeline_stage("design_attempt") == "design"
    assert canonical_pipeline_stage("implementation") == "implementation"
    assert canonical_pipeline_stage("implementation_attempt") == "implementation"
    assert canonical_pipeline_stage("implementation_template") == "implementation"
    assert canonical_pipeline_stage("compatibility_check") == "compatibility"
    assert canonical_pipeline_stage("novelty_check") == "novelty"
    assert canonical_pipeline_stage("repair") == "repair"
    assert canonical_pipeline_stage("repair_attempt") == "repair"


def test_canonical_stage_unknown_passes_through() -> None:
    """An unknown stage name is returned unchanged so test fixtures and
    future stages can still look up routes by exact key without forcing
    every new stage into the alias table."""

    assert canonical_pipeline_stage("custom_stage_xyz") == "custom_stage_xyz"


def test_parse_pipelines_accepts_omegaconf_list() -> None:
    raw = OmegaConf.create(
        [
            {
                "id": "gemma_only",
                "stages": {
                    "rationalization": "ollama_gemma4_31b",
                    "design": "ollama_gemma4_31b",
                    "implementation": "ollama_gemma4_31b",
                    "compatibility": "ollama_gemma4_31b",
                    "novelty": "ollama_gemma4_31b",
                    "repair": "ollama_gemma4_31b",
                },
            },
            {
                "id": "qwen_creative_gemma_check",
                "stages": {
                    "rationalization": "ollama_qwen_122b",
                    "design": "ollama_qwen_122b",
                    "implementation": "ollama_qwen_122b",
                    "compatibility": "ollama_gemma4_31b",
                    "novelty": "ollama_gemma4_31b",
                    "repair": "ollama_gemma4_31b",
                },
            },
        ]
    )

    pipelines = parse_pipelines(raw)

    assert [p.id for p in pipelines] == ["gemma_only", "qwen_creative_gemma_check"]
    creative = pipelines[1]
    assert creative.route_for("design") == "ollama_qwen_122b"
    assert creative.route_for("design_attempt") == "ollama_qwen_122b"
    assert creative.route_for("compatibility_check") == "ollama_gemma4_31b"
    assert creative.route_for("novelty_check") == "ollama_gemma4_31b"
    assert creative.route_for("repair") == "ollama_gemma4_31b"


def test_parse_pipelines_accepts_python_list() -> None:
    raw = [
        {
            "id": "p",
            "stages": {stage: "r" for stage in PIPELINE_STAGES},
        }
    ]
    pipelines = parse_pipelines(raw)
    assert pipelines[0].id == "p"
    for stage in PIPELINE_STAGES:
        assert pipelines[0].route_for(stage) == "r"


def test_parse_pipelines_treats_missing_block_as_empty() -> None:
    assert parse_pipelines(None) == []
    assert parse_pipelines([]) == []


def test_parse_pipelines_rejects_missing_canonical_stage() -> None:
    """A pipeline that omits any canonical stage must fail at parse time —
    otherwise the omission would only surface when that stage was actually
    invoked, potentially many organisms into a run."""

    raw = [
        {
            "id": "missing_repair",
            "stages": {
                "rationalization": "a",
                "design": "a",
                "implementation": "a",
                "compatibility": "a",
                "novelty": "a",
                # repair intentionally missing
            },
        }
    ]

    with pytest.raises(ValueError, match="missing required stages.*repair"):
        parse_pipelines(raw)


def test_parse_pipelines_rejects_duplicate_pipeline_ids() -> None:
    raw = [
        {"id": "p", "stages": {stage: "r" for stage in PIPELINE_STAGES}},
        {"id": "p", "stages": {stage: "r" for stage in PIPELINE_STAGES}},
    ]
    with pytest.raises(ValueError, match="Duplicate pipeline id"):
        parse_pipelines(raw)


def test_parse_pipelines_rejects_blank_id() -> None:
    raw = [{"id": "", "stages": {stage: "r" for stage in PIPELINE_STAGES}}]
    with pytest.raises(ValueError, match="missing 'id'"):
        parse_pipelines(raw)


def test_parse_pipelines_rejects_non_mapping_stages() -> None:
    raw = [{"id": "p", "stages": []}]
    with pytest.raises(ValueError, match="non-empty 'stages' mapping"):
        parse_pipelines(raw)


def test_pipeline_route_for_uses_canonical_lookup() -> None:
    """``PipelineConfig.route_for`` accepts concrete or canonical stage
    names; both reach the same configured route."""

    pipeline = PipelineConfig(
        id="p",
        stages={stage: f"route-{stage}" for stage in PIPELINE_STAGES},
    )
    assert pipeline.route_for("design") == "route-design"
    assert pipeline.route_for("design_attempt") == "route-design"
    assert pipeline.route_for("implementation_template") == "route-implementation"


def test_pipeline_route_for_raises_on_unknown_stage() -> None:
    pipeline = PipelineConfig(
        id="p",
        stages={stage: "r" for stage in PIPELINE_STAGES},
    )
    with pytest.raises(KeyError, match="no route for stage"):
        pipeline.route_for("totally_unrelated_stage")


def test_validate_pipeline_routes_rejects_unknown_route() -> None:
    pipelines = [
        PipelineConfig(
            id="bad",
            stages={
                **{stage: "ollama_gemma4_31b" for stage in PIPELINE_STAGES},
                "design": "ollama_nonexistent",
            },
        )
    ]
    with pytest.raises(ValueError, match="references unknown route 'ollama_nonexistent'"):
        validate_pipeline_routes(pipelines, ["ollama_gemma4_31b"])


def test_validate_pipeline_routes_accepts_known_routes() -> None:
    pipelines = [
        PipelineConfig(
            id="ok",
            stages={stage: "ollama_gemma4_31b" for stage in PIPELINE_STAGES},
        )
    ]
    validate_pipeline_routes(pipelines, ["ollama_gemma4_31b", "ollama_qwen35_35b"])


class _StubRegistry:
    """Minimal stand-in for ``ApiPlatformRegistry`` for generator tests.

    The real registry parses provider configs, builds brokers, and manages
    process pools — none of that is needed to test pipeline routing
    decisions, which only need ``available_route_ids`` and a no-op
    weight validator.
    """

    def __init__(self, route_ids: list[str]) -> None:
        self.available_route_ids = list(route_ids)

    def validate_route_weights(self, weights: dict[str, float]) -> None:
        unknown = [r for r in weights if r not in self.available_route_ids]
        if unknown:
            raise ValueError(f"unknown routes: {unknown}")


def _generator_cfg_with_pipelines(pipelines: list[dict] | None) -> object:
    """Build the minimal OmegaConf payload that ``BaseLlmGenerator`` reads."""

    llm: dict[str, object] = {
        "seed": 123,
        "route_weights": {"ollama_gemma4_31b": 1.0},
    }
    if pipelines is not None:
        llm["pipelines"] = pipelines
    return OmegaConf.create({"evolver": {"llm": llm}})


def test_generator_sample_route_id_routes_through_pipeline() -> None:
    """When pipelines are configured, ``sample_route_id(organism_id, stage)``
    resolves through the pipeline's stage->route map and ignores the legacy
    ``route_weights`` block."""

    from src.evolve.llm_generator_base import BaseLlmGenerator

    cfg = _generator_cfg_with_pipelines(
        [
            {
                "id": "creative_qwen_check_gemma",
                "stages": {
                    "rationalization": "ollama_qwen_122b",
                    "design": "ollama_qwen_122b",
                    "implementation": "ollama_qwen_122b",
                    "compatibility": "ollama_gemma4_31b",
                    "novelty": "ollama_gemma4_31b",
                    "repair": "ollama_gemma4_31b",
                },
            }
        ]
    )
    registry = _StubRegistry(["ollama_gemma4_31b", "ollama_qwen_122b"])
    generator = BaseLlmGenerator(cfg, registry)

    assert generator.sample_route_id(organism_id="o1", stage="design") == "ollama_qwen_122b"
    assert generator.sample_route_id(organism_id="o1", stage="implementation") == "ollama_qwen_122b"
    assert generator.sample_route_id(organism_id="o1", stage="compatibility_check") == "ollama_gemma4_31b"
    assert generator.sample_route_id(organism_id="o1", stage="novelty_check") == "ollama_gemma4_31b"
    assert generator.sample_route_id(organism_id="o1", stage="repair") == "ollama_gemma4_31b"
    # design_attempt should alias to design.
    assert generator.sample_route_id(organism_id="o1", stage="design_attempt") == "ollama_qwen_122b"


def test_generator_sticky_pipeline_per_organism() -> None:
    """All stages of one organism must share one pipeline. When two
    pipelines exist, organisms should be partitioned across them
    (deterministically per organism_id) and an organism's pipeline
    should be cached for the duration of its lifecycle."""

    from src.evolve.llm_generator_base import BaseLlmGenerator

    cfg = _generator_cfg_with_pipelines(
        [
            {
                "id": "all_gemma",
                "stages": {stage: "ollama_gemma4_31b" for stage in PIPELINE_STAGES},
            },
            {
                "id": "all_qwen",
                "stages": {stage: "ollama_qwen35_35b" for stage in PIPELINE_STAGES},
            },
        ]
    )
    registry = _StubRegistry(["ollama_gemma4_31b", "ollama_qwen35_35b"])
    generator = BaseLlmGenerator(cfg, registry)

    # Sample many organisms; each must pick exactly one pipeline and
    # stick with it across every stage.
    for organism_id in [f"org_{i}" for i in range(20)]:
        design_route = generator.sample_route_id(organism_id=organism_id, stage="design")
        impl_route = generator.sample_route_id(organism_id=organism_id, stage="implementation")
        compat_route = generator.sample_route_id(
            organism_id=organism_id, stage="compatibility_check"
        )
        assert design_route == impl_route == compat_route
        # And the cached pipeline_id matches what we'd resolve from the route.
        cached_pipeline = generator.pipeline_id_for_organism(organism_id)
        assert cached_pipeline in {"all_gemma", "all_qwen"}
        if cached_pipeline == "all_gemma":
            assert design_route == "ollama_gemma4_31b"
        else:
            assert design_route == "ollama_qwen35_35b"


def test_generator_empty_pipelines_falls_back_to_route_weights() -> None:
    """When ``evolver.llm.pipelines`` is omitted, the legacy per-stage
    route-weight sampling path stays in effect — passing ``stage`` is
    accepted but doesn't change the resolution rule."""

    from src.evolve.llm_generator_base import BaseLlmGenerator

    cfg = _generator_cfg_with_pipelines(None)  # no pipelines
    registry = _StubRegistry(["ollama_gemma4_31b"])
    generator = BaseLlmGenerator(cfg, registry)

    route = generator.sample_route_id(organism_id="o1", stage="design")
    assert route == "ollama_gemma4_31b"
    # No pipeline assigned because pipelines are disabled.
    assert generator.pipeline_id_for_organism("o1") is None


def test_generator_observe_pipeline_reward_no_op_when_disabled() -> None:
    """``observe_pipeline_reward`` must be safe to call regardless of
    whether pipelines are configured. The evolution loop calls both
    ``observe_route_reward`` and ``observe_pipeline_reward``
    unconditionally; one of them is expected to be a no-op depending on
    which mode is active."""

    from src.evolve.llm_generator_base import BaseLlmGenerator

    cfg = _generator_cfg_with_pipelines(None)
    registry = _StubRegistry(["ollama_gemma4_31b"])
    generator = BaseLlmGenerator(cfg, registry)

    # Should not raise.
    assert generator.observe_pipeline_reward("nonexistent", simple_score=-1.0) == 0.0
