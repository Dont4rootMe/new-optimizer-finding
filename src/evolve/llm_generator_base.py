"""Generic LLM routing helpers for canonical organism generators."""

from __future__ import annotations

import ast
import hashlib
import logging
import random
import re
import threading

from omegaconf import DictConfig, OmegaConf

from api_platforms import ApiPlatformRegistry
from src.evolve.bandit import AdaptiveSampler, cfg_from_omega
from src.evolve.pipeline import (
    PipelineConfig,
    parse_pipelines,
    validate_pipeline_routes,
)

LOGGER = logging.getLogger(__name__)

_FULL_MODE_PREAMBLE_RE = re.compile(
    r"^\s*##\s*COMPILATION_MODE\s*\n\s*FULL\s*(?:\n+|$)",
    re.IGNORECASE,
)


def _strip_legacy_full_mode_preamble(text: str) -> str:
    """Remove a stray FULL-mode patch header from an otherwise full Python file."""

    source = str(text).strip()
    if not source:
        return ""
    return _FULL_MODE_PREAMBLE_RE.sub("", source, count=1).lstrip()


class BaseLlmGenerator:
    """Hold route-weight sampling and code extraction for creation stages."""

    def __init__(self, cfg: DictConfig, llm_registry: ApiPlatformRegistry) -> None:
        self.cfg = cfg
        self.evolver_cfg = cfg.evolver
        self.llm_cfg = cfg.evolver.llm
        self.seed = int(self.llm_cfg.seed)
        self.registry = llm_registry
        configured_route_weights = {
            str(route_id): float(weight)
            for route_id, weight in dict(self.llm_cfg.get("route_weights", {})).items()
        }
        if configured_route_weights:
            self.route_weights = configured_route_weights
        else:
            self.route_weights = {route_id: 1.0 for route_id in self.registry.available_route_ids}
        self.registry.validate_route_weights(self.route_weights)
        self._rng = random.Random(self.seed)
        self._rng_lock = threading.Lock()

        # Adaptive sampler over LLM routes. Default config keeps the legacy
        # weighted-static behaviour: when `evolver.llm.route_sampling` is
        # absent we fall back to weighted_static seeded with `route_weights`.
        # When the user opts into `strategy: bandit`, the same `route_weights`
        # block is folded in as a Beta-prior bias so existing per-model
        # tuning is not thrown away on the switch.
        route_sampling_cfg = OmegaConf.select(self.cfg, "evolver.llm.route_sampling", default=None)
        bandit_cfg = cfg_from_omega(route_sampling_cfg, fallback_weights=self.route_weights)
        self.route_sampler: AdaptiveSampler = AdaptiveSampler(bandit_cfg)
        # Batch-scoped route assignment populated by the evolution loop before
        # a creation batch. Per-organism hashing alone does not guarantee that
        # a small batch is spread across all positive-weight routes — two seed
        # organisms on two equal-weight routes hit the same route 50% of the
        # time, leaving one ollama server idle. The evolution loop pre-assigns
        # a balanced mapping here so sample_route_id can honour it. Protected
        # by `_rng_lock` because multiple creation threads read it in parallel.
        self._batch_route_assignments: dict[str, str] = {}

        # ---- Pipeline mode -------------------------------------------------
        # When `evolver.llm.pipelines` is defined, route selection moves up
        # one level of abstraction: the bandit picks a *pipeline* (a named
        # bundle of canonical stage -> route_id assignments) per organism,
        # and every stage of that organism uses the pipeline's mapping. This
        # is opt-in. When the block is absent the legacy per-stage route
        # sampling above stays in effect.
        pipelines_cfg = OmegaConf.select(self.cfg, "evolver.llm.pipelines", default=None)
        self.pipelines: list[PipelineConfig] = parse_pipelines(pipelines_cfg)
        # Diagnostic — the previous run distributed 98.6% of calls to gemma
        # and 1.4% to qwen despite the awtf2025 yaml declaring a
        # ``qwen_creative_gemma_check`` pipeline. The leading hypothesis is
        # that ``pipelines`` is either not loading or is being shadowed by
        # the legacy ``route_sampling`` path. Surface the loaded list on
        # stderr so the post-mortem can confirm at-a-glance which branch is
        # actually live.
        if self.pipelines:
            LOGGER.info(
                "[pipelines] loaded %d pipeline(s): %s — pipeline-bandit path is ACTIVE",
                len(self.pipelines),
                [p.id for p in self.pipelines],
            )
        else:
            LOGGER.info(
                "[pipelines] no `evolver.llm.pipelines` configured — "
                "legacy `route_sampling` path stays in effect"
            )
        if self.pipelines:
            validate_pipeline_routes(self.pipelines, self.registry.available_route_ids)
            pipeline_sampling_cfg = OmegaConf.select(
                self.cfg, "evolver.llm.pipeline_sampling", default=None
            )
            pipeline_fallback = {p.id: 1.0 for p in self.pipelines}
            pipeline_bandit_cfg = cfg_from_omega(
                pipeline_sampling_cfg, fallback_weights=pipeline_fallback
            )
            self.pipeline_sampler: AdaptiveSampler | None = AdaptiveSampler(
                pipeline_bandit_cfg
            )
            self._pipelines_by_id: dict[str, PipelineConfig] = {
                p.id: p for p in self.pipelines
            }
        else:
            self.pipeline_sampler = None
            self._pipelines_by_id = {}
        # Per-organism pipeline assignment cache: an organism's pipeline is
        # sampled the first time *any* stage asks for its route, and held
        # for the rest of the organism's stages. Protected by `_rng_lock`.
        self._organism_pipeline_ids: dict[str, str] = {}

    def _extract_python(self, text: str, *, preserve_trailing_newline: bool = True) -> str:
        """Extract Python code from LLM response, stripping markdown fences.

        Strategy:
        1. Find ALL markdown code blocks and pick the longest one (the main code).
        2. If no fences found, use the whole response.
        3. Strip any remaining stray backtick lines.
        4. Strip a stray legacy FULL-mode preamble if the model emitted one.
        5. Validate syntax — raise ValueError if the code is not parseable Python.
        """

        # Find all code blocks — pick the longest to avoid grabbing a short snippet
        pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            code = max(matches, key=len).strip()
        else:
            code = text.strip()

        # Strip any remaining stray backtick lines that the regex didn't catch
        lines = code.split("\n")
        cleaned_lines = [line for line in lines if not line.strip().startswith("```")]
        code = "\n".join(cleaned_lines).strip()
        if not code.strip():
            code = text.strip()

        code = _strip_legacy_full_mode_preamble(code)
        if preserve_trailing_newline and code and not code.endswith("\n"):
            code += "\n"

        # Validate Python syntax — fail fast instead of wasting an eval slot
        try:
            ast.parse(code)
        except SyntaxError as exc:
            LOGGER.warning("Extracted code has syntax error: %s (line %s)", exc.msg, exc.lineno)
            raise ValueError(
                f"LLM returned syntactically invalid Python: {exc.msg} at line {exc.lineno}"
            ) from exc

        return code

    def set_batch_route_assignments(self, assignments: dict[str, str]) -> None:
        """Install a batch-scoped organism_id -> route_id mapping.

        The evolution loop calls this right before launching a creation batch.
        `sample_route_id` will honour the mapping whenever the organism_id is
        present, falling back to per-organism hashing otherwise. This is what
        guarantees that a small seed batch actually spreads across every
        positive-weight route instead of statistically colliding on one.
        """

        with self._rng_lock:
            self._batch_route_assignments = dict(assignments)

    def clear_batch_route_assignments(self) -> None:
        with self._rng_lock:
            self._batch_route_assignments = {}

    def sample_route_id(
        self,
        *,
        organism_id: str | None = None,
        stage: str | None = None,
    ) -> str:
        """Sample a route id, honouring pipeline mode when configured.

        Resolution order:
          1. **Pipeline mode** (``evolver.llm.pipelines`` non-empty): the
             pipeline assigned to ``organism_id`` resolves ``stage`` to its
             configured route. The assignment is made on first lookup and
             cached for the rest of the organism's stages so every stage
             of one organism uses the same pipeline.
          2. **Batch-scoped legacy assignment**: ``set_batch_route_assignments``
             may have pinned a route for this organism for load-balancing
             across providers.
          3. **Per-organism deterministic sampling**: a fresh RNG seeded
             from ``(self.seed, organism_id)`` asks ``route_sampler`` for
             a route weighted by ``route_weights``.
          4. **Generator-level RNG fallback** when ``organism_id`` is None
             (manual pipeline / one-off callers).

        ``stage`` is required in pipeline mode (the pipeline only resolves
        when it knows which canonical stage to look up). Legacy callers
        that pass only ``organism_id`` continue to work when pipelines are
        empty.
        """

        # Pipeline mode short-circuit. When pipelines are configured, the
        # legacy `route_weights` path is not consulted — pipelines own
        # routing for the whole organism.
        if self.pipelines and organism_id is not None and stage is not None:
            pipeline = self.pipeline_for_organism(organism_id)
            return pipeline.route_for(stage)

        available = [route_id for route_id, weight in self.route_weights.items() if weight > 0]
        if not available:
            raise ValueError("evolver.llm.route_weights must contain at least one positive-weight route.")

        if organism_id is not None:
            with self._rng_lock:
                assigned = self._batch_route_assignments.get(organism_id)
            if assigned is not None and assigned in available:
                return assigned

            # Per-organism RNG so the route is reproducible per organism and
            # is independent of concurrent thread interleaving. The bandit
            # sampler is asked through this RNG so even Thompson sampling
            # decisions are reproducible per (seed, organism_id).
            digest = hashlib.sha1(
                f"{self.seed}:{organism_id}".encode("utf-8"), usedforsecurity=False
            ).digest()
            call_seed = int.from_bytes(digest[:8], "big", signed=False)
            call_rng = random.Random(call_seed)
            return self.route_sampler.select(available, rng=call_rng)

        with self._rng_lock:
            return self.route_sampler.select(available, rng=self._rng)

    def pipeline_for_organism(self, organism_id: str) -> PipelineConfig:
        """Return the pipeline assigned to ``organism_id``, sampling lazily.

        Caller must have verified ``self.pipelines`` is non-empty. The
        pipeline_id is cached so subsequent stages of the same organism
        see the same pipeline.
        """

        if not self.pipelines or self.pipeline_sampler is None:
            raise RuntimeError(
                "pipeline_for_organism called but evolver.llm.pipelines is empty"
            )

        with self._rng_lock:
            cached = self._organism_pipeline_ids.get(organism_id)
        if cached is not None:
            return self._pipelines_by_id[cached]

        # Deterministic per-organism RNG mirrors the route-sampling path:
        # two near-simultaneous creations on the same organism_id must land
        # on the same pipeline regardless of thread interleaving.
        digest = hashlib.sha1(
            f"{self.seed}:pipeline:{organism_id}".encode("utf-8"),
            usedforsecurity=False,
        ).digest()
        call_seed = int.from_bytes(digest[:8], "big", signed=False)
        call_rng = random.Random(call_seed)
        available_ids = [p.id for p in self.pipelines]
        pipeline_id = self.pipeline_sampler.select(available_ids, rng=call_rng)

        with self._rng_lock:
            self._organism_pipeline_ids[organism_id] = pipeline_id
        # One-line diagnostic so the post-mortem can confirm the bandit
        # is actually splitting work across pipelines when more than one
        # is configured. Logged once per organism (first lookup wins;
        # subsequent stages of the same organism short-circuit on the
        # cache above).
        LOGGER.info(
            "[pipelines] organism=%s assigned pipeline=%s (from %s)",
            organism_id,
            pipeline_id,
            available_ids,
        )
        return self._pipelines_by_id[pipeline_id]

    def pipeline_id_for_organism(self, organism_id: str) -> str | None:
        """Return the cached pipeline_id for an organism, or None when no
        pipeline has been sampled for it yet (or when pipelines are disabled).

        Used by the evolution loop to persist ``llm_pipeline_id`` on the
        organism and to feed reward back to the right pipeline bandit arm.
        """

        if not self.pipelines:
            return None
        with self._rng_lock:
            return self._organism_pipeline_ids.get(organism_id)

    def observe_route_reward(self, route_id: str, *, simple_score: float | None) -> float:
        """Feed a route's outcome back into the adaptive sampler.

        Called by the evolution loop right after an organism finishes its
        simple evaluation (or fails to). For non-bandit strategies the
        underlying ``AdaptiveSampler`` performs a no-op, so this is safe to
        call unconditionally.
        """

        return self.route_sampler.observe(route_id, simple_score=simple_score)

    def observe_pipeline_reward(
        self, pipeline_id: str, *, simple_score: float | None
    ) -> float:
        """Feed a pipeline's outcome back into the pipeline-level sampler.

        No-op when pipelines are disabled. Mirrors ``observe_route_reward``
        so the evolution loop can call both unconditionally and let the
        unused one short-circuit.
        """

        if self.pipeline_sampler is None:
            return 0.0
        return self.pipeline_sampler.observe(pipeline_id, simple_score=simple_score)
