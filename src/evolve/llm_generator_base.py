"""Generic LLM routing helpers for canonical organism generators."""

from __future__ import annotations

import ast
import hashlib
import logging
import random
import re
import threading

from omegaconf import DictConfig

from api_platforms import ApiPlatformRegistry

LOGGER = logging.getLogger(__name__)


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
        # Batch-scoped route assignment populated by the evolution loop before
        # a creation batch. Per-organism hashing alone does not guarantee that
        # a small batch is spread across all positive-weight routes — two seed
        # organisms on two equal-weight routes hit the same route 50% of the
        # time, leaving one ollama server idle. The evolution loop pre-assigns
        # a balanced mapping here so sample_route_id can honour it. Protected
        # by `_rng_lock` because multiple creation threads read it in parallel.
        self._batch_route_assignments: dict[str, str] = {}

    def _extract_python(self, text: str) -> str:
        """Extract Python code from LLM response, stripping markdown fences.

        Strategy:
        1. Find ALL markdown code blocks and pick the longest one (the main code).
        2. If no fences found, use the whole response.
        3. Strip any remaining stray backtick lines.
        4. Validate syntax — raise ValueError if the code is not parseable Python.
        """

        # Find all code blocks — pick the longest to avoid grabbing a short snippet
        pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            code = max(matches, key=len).strip() + "\n"
        else:
            code = text.strip() + "\n"

        # Strip any remaining stray backtick lines that the regex didn't catch
        lines = code.split("\n")
        cleaned_lines = [line for line in lines if not line.strip().startswith("```")]
        code = "\n".join(cleaned_lines)
        if not code.strip():
            code = text.strip() + "\n"

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

    def sample_route_id(self, *, organism_id: str | None = None) -> str:
        """Sample a route id by `route_weights`, optionally per-organism-deterministic.

        Precedence, when `organism_id` is provided:
          1. `self._batch_route_assignments[organism_id]` — the evolution loop's
             pre-computed balanced assignment for the current batch.
          2. A fresh per-call RNG seeded from `(self.seed, organism_id)`.

        The deterministic fallback makes route selection reproducible per
        organism AND removes the coupling between the order in which
        concurrent creation threads grab `_rng_lock` and the route
        distribution. Without the batch mapping, two near-simultaneous seed
        creations routinely land on the same route, defeating the purpose of
        having multiple `route_weights` entries and saturating one provider
        while the others idle. The batch mapping fixes the remaining
        statistical variance for small batches.

        When no `organism_id` is provided (legacy callers) we fall back to
        the shared generator-level RNG.
        """

        available = [route_id for route_id, weight in self.route_weights.items() if weight > 0]
        if not available:
            raise ValueError("evolver.llm.route_weights must contain at least one positive-weight route.")
        weights = [self.route_weights[route_id] for route_id in available]

        if organism_id is not None:
            with self._rng_lock:
                assigned = self._batch_route_assignments.get(organism_id)
            if assigned is not None and assigned in available:
                return assigned

            digest = hashlib.sha1(
                f"{self.seed}:{organism_id}".encode("utf-8"), usedforsecurity=False
            ).digest()
            call_seed = int.from_bytes(digest[:8], "big", signed=False)
            call_rng = random.Random(call_seed)
            return call_rng.choices(available, weights=weights, k=1)[0]

        with self._rng_lock:
            return self._rng.choices(available, weights=weights, k=1)[0]
