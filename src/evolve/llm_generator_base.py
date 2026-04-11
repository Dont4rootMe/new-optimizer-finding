"""Generic LLM routing helpers for canonical organism generators."""

from __future__ import annotations

import hashlib
import random
import re
import threading

from omegaconf import DictConfig

from api_platforms import ApiPlatformRegistry


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

    def _extract_python(self, text: str) -> str:
        pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(1).strip() + "\n"
        return text.strip() + "\n"

    def sample_route_id(self, *, organism_id: str | None = None) -> str:
        """Sample a route id by `route_weights`, optionally per-organism-deterministic.

        When `organism_id` is provided we derive a fresh per-call RNG seeded
        from `(self.seed, organism_id)`. This makes route selection
        reproducible per organism AND removes the coupling between the order
        in which concurrent creation threads grab `_rng_lock` and the route
        distribution. Without this, two near-simultaneous seed creations
        routinely land on the same route, which defeats the whole purpose of
        having multiple `route_weights` entries and saturates one provider
        while the others idle.

        When no `organism_id` is provided (legacy callers) we fall back to
        the shared generator-level RNG.
        """

        available = [route_id for route_id, weight in self.route_weights.items() if weight > 0]
        if not available:
            raise ValueError("evolver.llm.route_weights must contain at least one positive-weight route.")
        weights = [self.route_weights[route_id] for route_id in available]

        if organism_id is not None:
            digest = hashlib.sha1(
                f"{self.seed}:{organism_id}".encode("utf-8"), usedforsecurity=False
            ).digest()
            call_seed = int.from_bytes(digest[:8], "big", signed=False)
            call_rng = random.Random(call_seed)
            return call_rng.choices(available, weights=weights, k=1)[0]

        with self._rng_lock:
            return self._rng.choices(available, weights=weights, k=1)[0]
