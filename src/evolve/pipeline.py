"""Per-organism pipeline configurations and bandit sampling over them.

Background
----------

The earlier route-sampling design picked an LLM route independently for
every stage of every organism. That made sense when only one model was
available, but stopped being expressive enough as soon as we wanted to
say things like "use the heavy reasoning-capable model only for the
creative stages (rationalization + design + implementation) and the
cheap fast model for the validators (compatibility + novelty + repair)."
With per-stage independent routing the bandit had to learn six small
single-arm problems instead of one joint policy, and the operator had
no way to express "these stages move together."

A pipeline is a named bundle that assigns one route to each canonical
stage. A run defines a list of candidate pipelines; the bandit picks
one per organism, and every stage of that organism runs against the
pipeline's stage→route mapping. Reward feedback (organism fitness)
flows to the pipeline that produced it.

When ``evolver.llm.pipelines`` is empty or absent, the legacy per-stage
``route_weights`` path stays in effect — this module is a non-breaking
addition.

Stage canonicalization
----------------------

Concrete stage strings used at call sites are richer than the six
pipeline categories (e.g. ``design_attempt`` vs ``design``, or
``compatibility_check`` vs ``compatibility``).
:func:`canonical_pipeline_stage` collapses each concrete stage into one
of :data:`PIPELINE_STAGES`. Operators write pipelines in the canonical
form; callers look up routes by passing the concrete stage and the
canonical form is resolved internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from omegaconf import DictConfig, ListConfig, OmegaConf

# The canonical stage taxonomy a pipeline must cover. Adding a new
# canonical stage means: (a) extending this tuple, (b) extending the
# alias map below, (c) updating every pipeline config to fill the new
# slot. We keep the list deliberately small.
PIPELINE_STAGES: tuple[str, ...] = (
    "rationalization",
    "design",
    "implementation",
    "compatibility",
    "novelty",
    "repair",
)


# Maps concrete stage strings emitted at LLM call sites to the canonical
# stage the pipeline configures. Anything not in this map is passed
# through unchanged — that lets unforeseen stages (or test fixtures)
# still resolve via direct key lookup on the pipeline.
_STAGE_ALIASES: dict[str, str] = {
    "design_rationalization": "rationalization",
    "design": "design",
    "design_attempt": "design",
    "implementation": "implementation",
    "implementation_attempt": "implementation",
    "implementation_template": "implementation",
    "compatibility_check": "compatibility",
    "novelty_check": "novelty",
    "repair": "repair",
    "repair_attempt": "repair",
}


def canonical_pipeline_stage(stage: str) -> str:
    """Collapse a concrete stage name into one of :data:`PIPELINE_STAGES`."""

    return _STAGE_ALIASES.get(stage, stage)


@dataclass(frozen=True)
class PipelineConfig:
    """Named bundle of stage→route assignments for one organism's lifecycle.

    ``stages`` is keyed by canonical stage name; values are route_ids
    from the api_platforms registry.
    """

    id: str
    stages: dict[str, str]

    def route_for(self, stage: str) -> str:
        """Return the configured route for the concrete ``stage``."""

        canonical = canonical_pipeline_stage(stage)
        route = self.stages.get(canonical)
        if route is None:
            available = ", ".join(sorted(self.stages))
            raise KeyError(
                f"Pipeline {self.id!r} has no route for stage {stage!r} "
                f"(canonical={canonical!r}); available stages: [{available}]"
            )
        return route


def parse_pipelines(cfg: Any) -> list[PipelineConfig]:
    """Parse the ``evolver.llm.pipelines`` block into ``PipelineConfig``s.

    Accepts an OmegaConf list, a Python list, or ``None``/missing. Each
    entry must be a mapping with at least ``id`` and ``stages`` keys.
    Every canonical stage in :data:`PIPELINE_STAGES` must be present —
    a missing slot is a configuration bug rather than a default we
    silently fill, because the resulting behavior would be invisible at
    runtime and would only surface when the missing stage is invoked.
    """

    if cfg is None:
        return []
    if isinstance(cfg, (DictConfig, ListConfig)):
        raw = OmegaConf.to_container(cfg, resolve=True)
    else:
        raw = cfg
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            "evolver.llm.pipelines must be a list of {id, stages} entries; "
            f"got {type(raw).__name__}"
        )

    pipelines: list[PipelineConfig] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Pipeline entry #{index} must be a mapping; got {type(entry).__name__}"
            )
        pid = str(entry.get("id", "")).strip()
        if not pid:
            raise ValueError(f"Pipeline entry #{index} is missing 'id'")
        if pid in seen_ids:
            raise ValueError(f"Duplicate pipeline id {pid!r}")
        stages_raw = entry.get("stages")
        if not isinstance(stages_raw, dict) or not stages_raw:
            raise ValueError(
                f"Pipeline {pid!r} must define a non-empty 'stages' mapping"
            )
        normalized: dict[str, str] = {}
        for key, value in stages_raw.items():
            canonical = canonical_pipeline_stage(str(key))
            normalized[canonical] = str(value)
        missing = [stage for stage in PIPELINE_STAGES if stage not in normalized]
        if missing:
            raise ValueError(
                f"Pipeline {pid!r} is missing required stages: {missing}; "
                f"each pipeline must define every canonical stage in "
                f"{list(PIPELINE_STAGES)}"
            )
        pipelines.append(PipelineConfig(id=pid, stages=normalized))
        seen_ids.add(pid)
    return pipelines


def validate_pipeline_routes(
    pipelines: Iterable[PipelineConfig], available_routes: Iterable[str]
) -> None:
    """Fail fast when a pipeline references an unregistered route.

    Misnaming a route in a pipeline config would otherwise only surface
    when the relevant stage was actually invoked — potentially many
    organisms into a run. Validation at startup keeps the error cheap.
    """

    available = set(available_routes)
    for pipeline in pipelines:
        for stage, route in pipeline.stages.items():
            if route not in available:
                raise ValueError(
                    f"Pipeline {pipeline.id!r} stage {stage!r} references "
                    f"unknown route {route!r}; available routes: "
                    f"{sorted(available)}"
                )
