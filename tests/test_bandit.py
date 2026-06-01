"""Tests for bandit samplers, reward computers, and the adaptive wrapper."""

from __future__ import annotations

import random
from collections import Counter

import pytest
from omegaconf import OmegaConf

from src.evolve.bandit import (
    AdaptiveSampler,
    BanditConfig,
    ConditionalAdaptiveSampler,
    DiscountedThompsonSampler,
    RewardComputer,
    RewardComputerConfig,
    UniformSampler,
    WeightedStaticSampler,
    cfg_from_omega,
)


# ---------------------------------------------------------------------------
# Reward computer
# ---------------------------------------------------------------------------


def test_reward_computer_survival_mode_returns_binary() -> None:
    rc = RewardComputer(RewardComputerConfig(mode="survival"))
    assert rc.compute(simple_score=0.5) == 1.0
    assert rc.compute(simple_score=None) == 0.0


def test_reward_computer_score_quantile_warms_up_to_neutral() -> None:
    """With <2 observations the quantile is undefined; we return 0.5 (neutral)."""

    rc = RewardComputer(RewardComputerConfig(mode="score_quantile"))
    first = rc.compute(simple_score=1.0)
    assert first == pytest.approx(0.5)


def test_reward_computer_score_quantile_ranks_recent_window() -> None:
    rc = RewardComputer(RewardComputerConfig(mode="score_quantile", window=10))
    # warm up the window with a spread of scores
    for value in (0.1, 0.2, 0.3, 0.4, 0.5):
        rc.compute(simple_score=value)
    # Now a high score should land near rank=1.0
    high = rc.compute(simple_score=10.0)
    assert high > 0.5
    # And a low score should land near rank=0.0
    low = rc.compute(simple_score=-1.0)
    assert low < 0.5


def test_reward_computer_dead_organism_is_zero_in_score_quantile() -> None:
    rc = RewardComputer(RewardComputerConfig(mode="score_quantile"))
    rc.compute(simple_score=1.0)
    rc.compute(simple_score=2.0)
    assert rc.compute(simple_score=None) == 0.0


def test_reward_computer_hybrid_combines_survival_and_quality() -> None:
    rc = RewardComputer(RewardComputerConfig(mode="hybrid", survival_weight=0.4, window=10))
    # Warm up so quantile is meaningful
    for value in (0.0, 0.1, 0.2, 0.3, 0.4):
        rc.compute(simple_score=value)
    survived = rc.compute(simple_score=10.0)  # high quantile + survival bonus
    died = rc.compute(simple_score=None)
    # Survival bonus alone is 0.4; quantile ≈ 1.0 → expect close to 1.0 for survived
    assert survived > 0.6
    assert died == 0.0


def test_reward_computer_state_dict_round_trips() -> None:
    rc = RewardComputer(RewardComputerConfig(mode="score_quantile", window=5))
    for value in (0.1, 0.2, 0.3):
        rc.compute(simple_score=value)
    state = rc.state_dict()

    rc2 = RewardComputer(RewardComputerConfig(mode="score_quantile", window=5))
    rc2.load_state(state)
    assert rc2.state_dict()["scores_window"] == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Uniform / weighted static
# ---------------------------------------------------------------------------


def test_uniform_sampler_distributes_evenly_in_the_long_run() -> None:
    sampler = UniformSampler()
    rng = random.Random(0)
    counts = Counter(sampler.select(["a", "b", "c"], rng=rng) for _ in range(3000))
    # Allow ±3% slack — UniformSampler is exactly random.choice
    for arm in ("a", "b", "c"):
        assert 950 < counts[arm] < 1050


def test_weighted_static_sampler_tracks_weights() -> None:
    sampler = WeightedStaticSampler({"a": 9.0, "b": 1.0})
    rng = random.Random(0)
    counts = Counter(sampler.select(["a", "b"], rng=rng) for _ in range(3000))
    # 9:1 ratio → roughly 2700:300
    assert counts["a"] > 2400
    assert counts["b"] < 600


def test_weighted_static_sampler_rejects_negative_weights() -> None:
    with pytest.raises(ValueError):
        WeightedStaticSampler({"a": -1.0, "b": 1.0})


# ---------------------------------------------------------------------------
# Discounted Thompson sampler
# ---------------------------------------------------------------------------


def test_discounted_thompson_lazily_initializes_arms_with_prior() -> None:
    sampler = DiscountedThompsonSampler(discount=0.9, prior_alpha=2.0, prior_beta=3.0)
    rng = random.Random(7)
    sampler.select(["a", "b", "c"], rng=rng)
    state = sampler.state_dict()
    # All three arms get the prior on first touch
    assert set(state["alpha"]) == {"a", "b", "c"}
    for arm in ("a", "b", "c"):
        assert state["alpha"][arm] == pytest.approx(2.0)
        assert state["beta"][arm] == pytest.approx(3.0)


def test_discounted_thompson_update_decays_then_increments() -> None:
    sampler = DiscountedThompsonSampler(discount=0.5, prior_alpha=1.0, prior_beta=1.0)
    sampler.update("a", reward=1.0)
    state = sampler.state_dict()
    # α' = 1.0 * 0.5 + 1.0 = 1.5, β' = 1.0 * 0.5 + 0.0 = 0.5
    assert state["alpha"]["a"] == pytest.approx(1.5)
    assert state["beta"]["a"] == pytest.approx(0.5)


def test_discounted_thompson_clamps_out_of_range_rewards() -> None:
    sampler = DiscountedThompsonSampler(discount=1.0, prior_alpha=1.0, prior_beta=1.0)
    sampler.update("a", reward=2.5)  # clamped to 1.0
    sampler.update("a", reward=-1.0)  # clamped to 0.0
    state = sampler.state_dict()
    # α += 1 + 0 = 2, β += 0 + 1 = 1 (plus the priors)
    assert state["alpha"]["a"] == pytest.approx(2.0)
    assert state["beta"]["a"] == pytest.approx(2.0)


def test_discounted_thompson_prefers_arm_with_better_track_record() -> None:
    """The sampler should mostly pick the higher-reward arm after enough updates."""

    sampler = DiscountedThompsonSampler(discount=0.97, prior_alpha=1.0, prior_beta=1.0)
    rng = random.Random(42)
    # Train: arm "good" wins almost always, arm "bad" loses almost always
    for _ in range(150):
        sampler.update("good", reward=1.0)
        sampler.update("bad", reward=0.0)
    counts = Counter(sampler.select(["good", "bad"], rng=rng) for _ in range(500))
    assert counts["good"] > counts["bad"] * 5


def test_discounted_thompson_recovers_when_underdog_starts_winning() -> None:
    """If 'bad' starts producing rewards, the bandit should shift toward it.

    Key non-stationary property: the discount factor lets the bandit forget the
    earlier 'good' wins so a regime change in rewards isn't permanently locked
    out by the prior history.
    """

    sampler = DiscountedThompsonSampler(discount=0.85, prior_alpha=1.0, prior_beta=1.0)
    rng = random.Random(7)
    # Phase 1: 'good' is winning
    for _ in range(80):
        sampler.update("good", reward=1.0)
        sampler.update("bad", reward=0.0)
    # Phase 2: regime flips — 'bad' starts winning
    for _ in range(80):
        sampler.update("good", reward=0.0)
        sampler.update("bad", reward=1.0)
    counts = Counter(sampler.select(["good", "bad"], rng=rng) for _ in range(500))
    # After the flip the bandit should overwhelmingly prefer 'bad' (the new winner).
    assert counts["bad"] > counts["good"]


def test_discounted_thompson_state_dict_round_trips() -> None:
    sampler = DiscountedThompsonSampler(discount=0.9, prior_alpha=1.0, prior_beta=1.0)
    sampler.update("a", reward=1.0)
    sampler.update("b", reward=0.0)
    state = sampler.state_dict()

    sampler2 = DiscountedThompsonSampler(discount=0.9, prior_alpha=1.0, prior_beta=1.0)
    sampler2.load_state(state)
    assert sampler2.state_dict()["alpha"] == state["alpha"]
    assert sampler2.state_dict()["beta"] == state["beta"]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def test_cfg_from_omega_returns_default_when_payload_missing() -> None:
    cfg = cfg_from_omega(None)
    assert cfg.strategy == "uniform"


def test_cfg_from_omega_promotes_to_weighted_static_with_fallback_weights() -> None:
    cfg = cfg_from_omega(None, fallback_weights={"a": 1.0, "b": 0.5})
    assert cfg.strategy == "weighted_static"
    assert cfg.weights == {"a": 1.0, "b": 0.5}


def test_cfg_from_omega_folds_route_weights_into_bandit_prior_bias() -> None:
    payload = OmegaConf.create({"strategy": "bandit"})
    cfg = cfg_from_omega(payload, fallback_weights={"a": 2.0, "b": 1.0})
    assert cfg.strategy == "bandit"
    # When switching to bandit we keep existing per-arm tuning as a Beta-prior bias.
    assert cfg.prior_bias == {"a": 2.0, "b": 1.0}


def test_cfg_from_omega_reads_full_bandit_block() -> None:
    payload = OmegaConf.create(
        {
            "strategy": "bandit",
            "bandit": {
                "algorithm": "discounted_thompson",
                "discount": 0.91,
                "prior_alpha": 2.0,
                "prior_beta": 3.0,
                "reward_mode": "hybrid",
                "reward_window": 25,
                "survival_weight": 0.4,
            },
        }
    )
    cfg = cfg_from_omega(payload)
    assert cfg.discount == pytest.approx(0.91)
    assert cfg.prior_alpha == pytest.approx(2.0)
    assert cfg.prior_beta == pytest.approx(3.0)
    assert cfg.reward_mode == "hybrid"
    assert cfg.reward_window == 25
    assert cfg.survival_weight == pytest.approx(0.4)


def test_bandit_config_validates_reward_mode() -> None:
    cfg = BanditConfig(strategy="bandit", reward_mode="invalid")
    with pytest.raises(ValueError):
        cfg.validate()


# ---------------------------------------------------------------------------
# AdaptiveSampler / ConditionalAdaptiveSampler
# ---------------------------------------------------------------------------


def test_adaptive_sampler_uniform_is_no_op_on_observe() -> None:
    sampler = AdaptiveSampler(BanditConfig(strategy="uniform", reward_mode="survival"))
    # observe must not raise even though the underlying sampler is stateless;
    # survival mode → 1.0 when the organism reached a score, 0.0 otherwise.
    assert sampler.observe("any", simple_score=0.5) == 1.0
    assert sampler.observe("any", simple_score=None) == 0.0
    assert sampler.is_adaptive is False


def test_adaptive_sampler_bandit_round_trips_state() -> None:
    cfg = BanditConfig(
        strategy="bandit",
        algorithm="discounted_thompson",
        discount=0.9,
        prior_alpha=1.0,
        prior_beta=1.0,
        reward_mode="survival",
    )
    sampler = AdaptiveSampler(cfg)
    sampler.observe("a", simple_score=1.0)
    sampler.observe("b", simple_score=None)
    state = sampler.state_dict()

    sampler2 = AdaptiveSampler(cfg)
    sampler2.load_state(state)
    assert sampler2.state_dict()["sampler"]["alpha"] == state["sampler"]["alpha"]


def test_conditional_adaptive_sampler_separates_per_origin() -> None:
    cfg = BanditConfig(
        strategy="bandit",
        algorithm="discounted_thompson",
        discount=1.0,
        prior_alpha=1.0,
        prior_beta=1.0,
        reward_mode="survival",
    )
    sampler = ConditionalAdaptiveSampler(cfg)
    # From origin "A", arm "X" wins; from origin "B", arm "X" loses.
    sampler.observe("A", "X", simple_score=1.0)
    sampler.observe("B", "X", simple_score=None)
    state = sampler.state_dict()
    # Both origins now have a sampler tracking arm X with different posteriors.
    assert "A" in state["samplers"] and "B" in state["samplers"]
    a_alpha = state["samplers"]["A"]["sampler"]["alpha"]["X"]
    b_alpha = state["samplers"]["B"]["sampler"]["alpha"]["X"]
    assert a_alpha > b_alpha
