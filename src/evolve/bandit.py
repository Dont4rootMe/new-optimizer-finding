"""Multi-armed bandit samplers for adaptive non-stationary selection.

The evolution loop needs to pick three things per organism creation:

* an LLM route (model) — some routes may stably produce weaker organisms but
  catch up later as the population evolves into regimes that suit them
* a parent island — analogously, an island may be a temporary underdog
* a partner island for cross-island crossover — conditional on the origin
  island, some pairings work better than others

Uniform random sampling wastes effort on bad arms; pure greedy locks onto
early winners and never recovers from a regime shift. The classic remedy
is a Bayesian multi-armed bandit; for non-stationary rewards we use a
*discounted* Thompson sampler — every arm has a Beta(α, β) posterior and
both α, β decay by a discount factor γ before each update, so old
observations fade out and the bandit stays exploration-friendly.

This module is **stateful** but pickle-clean: a sampler exposes
``state_dict()`` / ``load_state(...)`` so the evolution loop can persist
the bandit across resumes via ``population_state.json``.
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------


_VALID_REWARD_MODES = ("survival", "score_quantile", "hybrid")


@dataclass
class RewardComputerConfig:
    mode: str = "score_quantile"
    window: int = 50
    survival_weight: float = 0.3  # only used in 'hybrid' mode

    def validate(self) -> None:
        if self.mode not in _VALID_REWARD_MODES:
            raise ValueError(
                f"Unsupported reward mode {self.mode!r}; expected one of {_VALID_REWARD_MODES}."
            )
        if self.window < 1:
            raise ValueError("Reward window must be >= 1.")
        if not 0.0 <= self.survival_weight <= 1.0:
            raise ValueError("survival_weight must be in [0, 1].")


class RewardComputer:
    """Maps `(simple_score, survived)` outcomes to a reward in ``[0, 1]``.

    The score window is deliberately small — we want the bandit to track
    *recent* score regimes, not the global all-time distribution, because the
    population's score distribution drifts as evolution proceeds.
    """

    def __init__(self, cfg: RewardComputerConfig) -> None:
        cfg.validate()
        self._cfg = cfg
        self._scores_window: deque[float] = deque(maxlen=cfg.window)

    def compute(self, *, simple_score: float | None) -> float:
        survived = simple_score is not None
        if survived:
            self._scores_window.append(float(simple_score))

        if self._cfg.mode == "survival":
            return 1.0 if survived else 0.0

        if self._cfg.mode == "score_quantile":
            if not survived:
                return 0.0
            return self._score_quantile(float(simple_score))

        # 'hybrid'
        survival_part = self._cfg.survival_weight * (1.0 if survived else 0.0)
        score_part = (1.0 - self._cfg.survival_weight) * (
            self._score_quantile(float(simple_score)) if survived else 0.0
        )
        return survival_part + score_part

    def _score_quantile(self, score: float) -> float:
        observed = list(self._scores_window)
        if len(observed) <= 1:
            return 0.5  # too little signal yet — neutral reward
        # rank from below; ties counted as half
        below = sum(1 for value in observed if value < score)
        ties = sum(1 for value in observed if value == score)
        rank = (below + 0.5 * ties) / len(observed)
        return max(0.0, min(1.0, rank))

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self._cfg.mode,
            "window": self._cfg.window,
            "survival_weight": self._cfg.survival_weight,
            "scores_window": list(self._scores_window),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        scores = state.get("scores_window")
        if isinstance(scores, list):
            self._scores_window.clear()
            for value in scores[-self._cfg.window :]:
                try:
                    self._scores_window.append(float(value))
                except (TypeError, ValueError):
                    continue


# ---------------------------------------------------------------------------
# Bandit samplers
# ---------------------------------------------------------------------------


class BanditSampler:
    """Abstract bandit interface.

    Implementations must be deterministic given an injected ``random.Random``
    so that runs are reproducible. The ``select`` call may be issued for any
    subset of the configured arms (``available``); arms not yet observed are
    initialized lazily from the prior.
    """

    name: str = "bandit"

    def select(self, available: list[str], *, rng: random.Random) -> str:
        raise NotImplementedError

    def update(self, arm: str, reward: float) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name}

    def load_state(self, state: dict[str, Any]) -> None:  # noqa: D401
        """Default: ignore — stateless samplers don't need restoration."""

    @staticmethod
    def _validate_available(available: list[str]) -> None:
        if not available:
            raise ValueError("BanditSampler.select called with no available arms.")


class UniformSampler(BanditSampler):
    """Uniform random over the currently-available arms."""

    name = "uniform"

    def select(self, available: list[str], *, rng: random.Random) -> str:
        self._validate_available(available)
        return rng.choice(available)

    def update(self, arm: str, reward: float) -> None:
        # Stateless — no learning.
        pass


class WeightedStaticSampler(BanditSampler):
    """Static-weight categorical sampler (matches the legacy `route_weights`).

    Arms not in ``weights`` get ``default_weight`` (1.0). All weights must be
    non-negative; arms with weight 0 are dropped from the available set.
    """

    name = "weighted_static"

    def __init__(self, weights: dict[str, float], *, default_weight: float = 1.0) -> None:
        if any(value < 0 for value in weights.values()):
            raise ValueError("WeightedStaticSampler weights must be non-negative.")
        self._weights = dict(weights)
        self._default_weight = float(default_weight)

    def select(self, available: list[str], *, rng: random.Random) -> str:
        self._validate_available(available)
        weights = [self._weight_for(arm) for arm in available]
        if sum(weights) <= 0:
            raise ValueError("WeightedStaticSampler: all available arms have zero weight.")
        return rng.choices(available, weights=weights, k=1)[0]

    def update(self, arm: str, reward: float) -> None:
        # Stateless.
        pass

    def _weight_for(self, arm: str) -> float:
        return float(self._weights.get(arm, self._default_weight))


class DiscountedThompsonSampler(BanditSampler):
    """Discounted Thompson sampler with Beta(α, β) posteriors.

    On each ``update(arm, reward)``:
      α[arm] = α[arm] * γ + reward
      β[arm] = β[arm] * γ + (1 - reward)

    The discount factor γ < 1 makes old observations fade — the effective
    sample size stabilizes around 1 / (1 - γ), so γ=0.97 gives ~33 effective
    observations of recent history per arm. Other arms are unaffected by an
    update — they keep their full posterior, which is correct because we want
    "no information about this arm right now" rather than "this arm got
    worse just because we didn't pick it".
    """

    name = "discounted_thompson"

    def __init__(
        self,
        *,
        discount: float,
        prior_alpha: float,
        prior_beta: float,
        prior_bias: dict[str, float] | None = None,
    ) -> None:
        if not 0.0 < discount <= 1.0:
            raise ValueError("Discount must be in (0, 1].")
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Prior alpha/beta must be positive (Beta distribution requirement).")
        self._discount = float(discount)
        self._prior_alpha = float(prior_alpha)
        self._prior_beta = float(prior_beta)
        self._prior_bias = dict(prior_bias or {})
        # Posterior parameters per arm. Lazily filled on first update / select.
        self._alpha: dict[str, float] = {}
        self._beta: dict[str, float] = {}

    def select(self, available: list[str], *, rng: random.Random) -> str:
        self._validate_available(available)
        best_arm = available[0]
        best_sample = -math.inf
        for arm in available:
            self._ensure_arm(arm)
            sample = _beta_variate(rng, self._alpha[arm], self._beta[arm])
            if sample > best_sample:
                best_sample = sample
                best_arm = arm
        return best_arm

    def update(self, arm: str, reward: float) -> None:
        if not 0.0 <= reward <= 1.0:
            # Clamp instead of raising — reward computers should keep [0, 1] but
            # numerical drift shouldn't break the loop.
            reward = max(0.0, min(1.0, reward))
        self._ensure_arm(arm)
        self._alpha[arm] = self._alpha[arm] * self._discount + reward
        self._beta[arm] = self._beta[arm] * self._discount + (1.0 - reward)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "discount": self._discount,
            "prior_alpha": self._prior_alpha,
            "prior_beta": self._prior_beta,
            "prior_bias": dict(self._prior_bias),
            "alpha": dict(self._alpha),
            "beta": dict(self._beta),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if state.get("name") != self.name:
            LOGGER.warning(
                "DiscountedThompsonSampler.load_state: name mismatch %r (expected %r); ignoring.",
                state.get("name"),
                self.name,
            )
            return
        alpha = state.get("alpha")
        beta = state.get("beta")
        if isinstance(alpha, dict) and isinstance(beta, dict):
            self._alpha = {str(arm): float(value) for arm, value in alpha.items() if _is_positive_finite(value)}
            self._beta = {str(arm): float(value) for arm, value in beta.items() if _is_positive_finite(value)}

    def posterior_mean(self, arm: str) -> float | None:
        """Expose the posterior mean for diagnostics / logging."""

        if arm not in self._alpha:
            return None
        a, b = self._alpha[arm], self._beta[arm]
        denom = a + b
        return a / denom if denom > 0 else None

    def _ensure_arm(self, arm: str) -> None:
        if arm in self._alpha:
            return
        bias = max(0.0, float(self._prior_bias.get(arm, 1.0)))
        # Scale the prior by bias: arms with higher bias get a stronger prior in
        # favour (more α relative to β); arms with bias 0 still receive the
        # uninformative Beta(prior_alpha, prior_beta) so they aren't disabled.
        scale = bias if bias > 0 else 1.0
        self._alpha[arm] = self._prior_alpha * scale
        self._beta[arm] = self._prior_beta


def _beta_variate(rng: random.Random, alpha: float, beta: float) -> float:
    """Wrap `random.betavariate` so we get something in (0, 1) reliably."""

    if alpha <= 0:
        alpha = 1e-3
    if beta <= 0:
        beta = 1e-3
    return rng.betavariate(alpha, beta)


def _is_positive_finite(value: Any) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


# ---------------------------------------------------------------------------
# Factory + reward-aware wrapper
# ---------------------------------------------------------------------------


@dataclass
class BanditConfig:
    """Hydra-side schema for a bandit slot.

    Attributes mirror the YAML keys directly. ``strategy`` selects between
    ``uniform`` / ``weighted_static`` / ``bandit``. The ``bandit`` block is
    only used when ``strategy == "bandit"``.
    """

    strategy: str = "uniform"
    algorithm: str = "discounted_thompson"
    discount: float = 0.97
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    reward_mode: str = "score_quantile"
    reward_window: int = 50
    survival_weight: float = 0.3
    weights: dict[str, float] | None = None  # for weighted_static
    prior_bias: dict[str, float] | None = None  # for bandit (alpha-bias seed)

    def validate(self) -> None:
        if self.strategy not in ("uniform", "weighted_static", "bandit"):
            raise ValueError(
                f"Unsupported sampling strategy {self.strategy!r}. "
                "Expected one of: uniform, weighted_static, bandit."
            )
        if self.strategy == "bandit":
            if self.algorithm not in ("discounted_thompson",):
                raise ValueError(
                    f"Unsupported bandit algorithm {self.algorithm!r}; expected discounted_thompson."
                )
            RewardComputerConfig(
                mode=self.reward_mode,
                window=self.reward_window,
                survival_weight=self.survival_weight,
            ).validate()


def build_sampler(cfg: BanditConfig) -> BanditSampler:
    cfg.validate()
    if cfg.strategy == "uniform":
        return UniformSampler()
    if cfg.strategy == "weighted_static":
        return WeightedStaticSampler(cfg.weights or {})
    # bandit
    return DiscountedThompsonSampler(
        discount=cfg.discount,
        prior_alpha=cfg.prior_alpha,
        prior_beta=cfg.prior_beta,
        prior_bias=cfg.prior_bias,
    )


def build_reward_computer(cfg: BanditConfig) -> RewardComputer:
    return RewardComputer(
        RewardComputerConfig(
            mode=cfg.reward_mode,
            window=cfg.reward_window,
            survival_weight=cfg.survival_weight,
        )
    )


def cfg_from_omega(payload: Any, *, fallback_weights: dict[str, float] | None = None) -> BanditConfig:
    """Build a `BanditConfig` from an OmegaConf node (or plain dict / None).

    Missing keys fall back to the dataclass defaults. ``fallback_weights`` is
    used as the default ``weights`` when ``strategy == "weighted_static"`` and
    no explicit weights were provided (this is how the existing
    ``llm.route_weights`` block flows in without a separate
    ``llm.route_sampling.bandit.weights`` knob).
    """

    if payload is None:
        cfg = BanditConfig()
        if fallback_weights:
            cfg.weights = dict(fallback_weights)
            cfg.strategy = "weighted_static"
        return cfg

    container = _to_dict(payload)
    bandit_block = _to_dict(container.get("bandit"))
    cfg = BanditConfig(
        strategy=str(container.get("strategy", "uniform")).strip().lower() or "uniform",
        algorithm=str(bandit_block.get("algorithm", "discounted_thompson")),
        discount=float(bandit_block.get("discount", 0.97)),
        prior_alpha=float(bandit_block.get("prior_alpha", 1.0)),
        prior_beta=float(bandit_block.get("prior_beta", 1.0)),
        reward_mode=str(bandit_block.get("reward_mode", "score_quantile")),
        reward_window=int(bandit_block.get("reward_window", 50)),
        survival_weight=float(bandit_block.get("survival_weight", 0.3)),
    )
    raw_weights = container.get("weights") or bandit_block.get("weights")
    if raw_weights is not None:
        cfg.weights = {str(arm): float(weight) for arm, weight in _to_dict(raw_weights).items()}
    elif fallback_weights and cfg.strategy == "weighted_static":
        cfg.weights = dict(fallback_weights)
    raw_bias = bandit_block.get("prior_bias")
    if raw_bias is not None:
        cfg.prior_bias = {str(arm): float(weight) for arm, weight in _to_dict(raw_bias).items()}
    elif fallback_weights and cfg.strategy == "bandit":
        # Fold the legacy `route_weights` (or analogous) into the prior so
        # users keep their existing tuning when switching to bandit mode.
        cfg.prior_bias = dict(fallback_weights)
    return cfg


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        from omegaconf import OmegaConf  # type: ignore[import-not-found]
    except ImportError:
        return {}
    try:
        resolved = OmegaConf.to_container(value, resolve=True)
    except Exception:  # noqa: BLE001
        return {}
    if isinstance(resolved, dict):
        return resolved
    return {}


# ---------------------------------------------------------------------------
# Adaptive sampler — a sampler + a reward computer + state persistence
# ---------------------------------------------------------------------------


class AdaptiveSampler:
    """Sampler + reward computer pair, persisted as a single state blob.

    This is the object the evolution loop and the LLM generator actually use.
    For non-bandit strategies the reward computer is still wired up but is
    ignored (the underlying sampler's ``update`` is a no-op).
    """

    def __init__(self, cfg: BanditConfig) -> None:
        cfg.validate()
        self._cfg = cfg
        self._sampler = build_sampler(cfg)
        self._reward = build_reward_computer(cfg)

    @property
    def strategy(self) -> str:
        return self._cfg.strategy

    @property
    def is_adaptive(self) -> bool:
        return self._cfg.strategy == "bandit"

    @property
    def sampler(self) -> BanditSampler:
        return self._sampler

    def select(self, available: Iterable[str], *, rng: random.Random) -> str:
        return self._sampler.select(list(available), rng=rng)

    def observe(self, arm: str, *, simple_score: float | None) -> float:
        """Compute reward, update the sampler, return the reward (for logging)."""

        reward = self._reward.compute(simple_score=simple_score)
        self._sampler.update(arm, reward)
        return reward

    def state_dict(self) -> dict[str, Any]:
        return {
            "strategy": self._cfg.strategy,
            "sampler": self._sampler.state_dict(),
            "reward": self._reward.state_dict(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if state.get("strategy") != self._cfg.strategy:
            # config change between runs — drop saved state silently
            return
        sampler_state = state.get("sampler")
        if isinstance(sampler_state, dict):
            self._sampler.load_state(sampler_state)
        reward_state = state.get("reward")
        if isinstance(reward_state, dict):
            self._reward.load_state(reward_state)


class ConditionalAdaptiveSampler:
    """One AdaptiveSampler per *condition* (e.g. one bandit per origin island).

    Used for cross-island partner selection: each origin island owns its own
    bandit over the partner-island arm set, so the bandit can learn that
    "from island A, partner B is good" without forcing "from island B,
    partner A is also good".
    """

    def __init__(self, cfg: BanditConfig) -> None:
        cfg.validate()
        self._cfg = cfg
        self._samplers: dict[str, AdaptiveSampler] = {}

    @property
    def strategy(self) -> str:
        return self._cfg.strategy

    @property
    def is_adaptive(self) -> bool:
        return self._cfg.strategy == "bandit"

    def select(self, condition: str, available: Iterable[str], *, rng: random.Random) -> str:
        return self._for(condition).select(available, rng=rng)

    def observe(self, condition: str, arm: str, *, simple_score: float | None) -> float:
        return self._for(condition).observe(arm, simple_score=simple_score)

    def _for(self, condition: str) -> AdaptiveSampler:
        sampler = self._samplers.get(condition)
        if sampler is None:
            sampler = AdaptiveSampler(self._cfg)
            self._samplers[condition] = sampler
        return sampler

    def state_dict(self) -> dict[str, Any]:
        return {
            "strategy": self._cfg.strategy,
            "samplers": {key: sampler.state_dict() for key, sampler in self._samplers.items()},
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if state.get("strategy") != self._cfg.strategy:
            return
        samplers = state.get("samplers")
        if not isinstance(samplers, dict):
            return
        for key, blob in samplers.items():
            if not isinstance(blob, dict):
                continue
            sampler = self._for(str(key))
            sampler.load_state(blob)
