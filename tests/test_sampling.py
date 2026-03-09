"""Tests for src/orchestrator/sampling.py.

Covers:
- Acceptance-rate properties (identical vs. divergent distributions)
- Residual distribution validity
- Statistical correctness: output distribution ≈ target (chi-squared / KL)
- Temperature=0 gives greedy (argmax)
- top_p nucleus filtering
- apply_temperature softmax properties
- Numerical edge cases: uniform, one-hot, very peaked, NaN/Inf inputs,
  zero-sum distributions, unnormalised inputs
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from src.orchestrator.sampling import (
    apply_temperature,
    compute_residual_distribution,
    rejection_sample,
    sample_from_distribution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 50  # small vocabulary for fast tests
SEED = 42


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _uniform(vocab: int = VOCAB) -> np.ndarray:
    return np.full(vocab, 1.0 / vocab)


def _one_hot(idx: int, vocab: int = VOCAB) -> np.ndarray:
    arr = np.zeros(vocab)
    arr[idx] = 1.0
    return arr


def _peaked(peak_idx: int, vocab: int = VOCAB, sharpness: float = 20.0) -> np.ndarray:
    """Very peaked distribution with most mass on one token."""
    logits = np.zeros(vocab)
    logits[peak_idx] = sharpness
    logits -= logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def _random_dist(vocab: int = VOCAB, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale=1.0, size=vocab)
    return raw / raw.sum()


# ---------------------------------------------------------------------------
# rejection_sample — acceptance rate
# ---------------------------------------------------------------------------


class TestRejectionSampleAcceptanceRate:
    def test_identical_distributions_high_acceptance(self) -> None:
        """When draft == target, acceptance rate should be ~100%."""
        dist = _random_dist()
        rng = _rng()
        n_trials = 1_000
        accepted_count = 0

        for _ in range(n_trials):
            # Sample a draft token from the distribution itself.
            token = int(rng.choice(VOCAB, p=dist))
            accepted, _ = rejection_sample(dist, dist, token, rng=rng)
            if accepted:
                accepted_count += 1

        rate = accepted_count / n_trials
        # Should be essentially 1.0; allow tiny slack for float rounding.
        assert rate >= 0.98, f"Expected ≥98% acceptance, got {rate:.1%}"

    def test_divergent_distributions_low_acceptance(self) -> None:
        """When draft and target strongly disagree, acceptance rate is low."""
        # Target: peaked on token 0.  Draft: peaked on token VOCAB-1.
        target = _peaked(0)
        draft = _peaked(VOCAB - 1)
        rng = _rng()
        n_trials = 1_000
        accepted_count = 0
        draft_token = VOCAB - 1  # draft always proposes this token

        for _ in range(n_trials):
            accepted, _ = rejection_sample(target, draft, draft_token, rng=rng)
            if accepted:
                accepted_count += 1

        rate = accepted_count / n_trials
        # target(VOCAB-1) ≈ 0, so acceptance ratio ≈ 0.
        assert rate <= 0.05, f"Expected ≤5% acceptance, got {rate:.1%}"

    def test_partial_overlap_intermediate_acceptance(self) -> None:
        """Moderate overlap → intermediate acceptance."""
        rng = _rng()
        target = _random_dist(seed=1)
        draft = _random_dist(seed=2)

        n_trials = 2_000
        accepted_count = 0
        for _ in range(n_trials):
            token = int(rng.choice(VOCAB, p=draft))
            accepted, _ = rejection_sample(target, draft, token, rng=rng)
            if accepted:
                accepted_count += 1

        rate = accepted_count / n_trials
        # Acceptance rate = sum_x min(p(x), q(x)) / sum_x q(x) — should be
        # strictly between the two extremes.
        assert 0.05 < rate < 0.95, f"Expected intermediate acceptance, got {rate:.1%}"

    def test_rejected_token_is_from_residual(self) -> None:
        """On rejection the returned token must be valid (in vocab range)."""
        target = _peaked(0)
        draft = _peaked(VOCAB - 1)
        rng = _rng()

        # Force enough trials to get rejections.
        for _ in range(200):
            accepted, token = rejection_sample(target, draft, VOCAB - 1, rng=rng)
            assert 0 <= token < VOCAB

    def test_accepted_token_equals_draft_token(self) -> None:
        """When accepted, the returned token must equal the draft token."""
        dist = _random_dist()
        rng = _rng()

        for _ in range(500):
            draft_token = int(rng.choice(VOCAB, p=dist))
            accepted, token = rejection_sample(dist, dist, draft_token, rng=rng)
            if accepted:
                assert token == draft_token


# ---------------------------------------------------------------------------
# rejection_sample — statistical correctness
# ---------------------------------------------------------------------------


class TestRejectionSampleStatisticalCorrectness:
    """Over many samples the output distribution should match the target.

    This is the core correctness guarantee of speculative decoding.
    """

    def test_output_matches_target_distribution(self) -> None:
        """Output token distribution ≈ target (chi-squared, α=0.001)."""
        target = _random_dist(seed=7)
        draft = _random_dist(seed=8)
        rng = _rng(seed=0)

        n_samples = 50_000
        counts = np.zeros(VOCAB, dtype=np.int64)

        for _ in range(n_samples):
            # Sample a draft token from the draft distribution.
            draft_token = int(rng.choice(VOCAB, p=draft))
            _, token = rejection_sample(target, draft, draft_token, rng=rng)
            counts[token] += 1

        expected = target * n_samples
        # Chi-squared goodness-of-fit. Use a very permissive threshold
        # since this is a statistical test that can flake.
        chi2, p_value = scipy_stats.chisquare(counts, f_exp=expected)
        assert p_value > 0.0001, (
            f"Output distribution diverges from target: chi2={chi2:.1f}, "
            f"p={p_value:.6f}"
        )

    def test_output_matches_target_kl_divergence(self) -> None:
        """KL divergence between empirical and target should be small."""
        target = _random_dist(seed=9)
        draft = _random_dist(seed=10)
        rng = _rng(seed=1)

        n_samples = 10_000
        counts = np.zeros(VOCAB, dtype=np.int64)

        for _ in range(n_samples):
            draft_token = int(rng.choice(VOCAB, p=draft))
            _, token = rejection_sample(target, draft, draft_token, rng=rng)
            counts[token] += 1

        empirical = (counts + 1e-9) / (counts.sum() + 1e-9 * VOCAB)
        kl = float(np.sum(target * np.log((target + 1e-10) / (empirical + 1e-10))))
        assert kl < 0.02, f"KL divergence too large: {kl:.4f}"


# ---------------------------------------------------------------------------
# compute_residual_distribution
# ---------------------------------------------------------------------------


class TestComputeResidualDistribution:
    def test_output_is_valid_distribution(self) -> None:
        """Residual must be non-negative and sum to 1."""
        target = _random_dist(seed=3)
        draft = _random_dist(seed=4)
        residual = compute_residual_distribution(target, draft)

        assert np.all(residual >= 0.0), "Residual has negative entries"
        assert math.isclose(float(residual.sum()), 1.0, abs_tol=1e-9)

    def test_identical_dists_fallback_to_target(self) -> None:
        """When target == draft, residual is all zeros → fall back to target."""
        dist = _random_dist()
        residual = compute_residual_distribution(dist, dist)
        np.testing.assert_allclose(residual, dist, atol=1e-9)

    def test_residual_mass_proportional_to_excess(self) -> None:
        """Token with p > q gets positive residual; token with p < q gets 0."""
        # target: all mass on token 0; draft: all mass on token 1.
        target = _one_hot(0)
        draft = _one_hot(1)
        residual = compute_residual_distribution(target, draft)
        assert residual[0] == pytest.approx(1.0)
        assert residual[1] == pytest.approx(0.0)

    def test_residual_concentrates_on_undersampled_tokens(self) -> None:
        """Residual should weight tokens where target > draft."""
        target = _peaked(5)
        draft = _peaked(10)
        residual = compute_residual_distribution(target, draft)
        # Token 5 has most of target mass and little in draft → high residual.
        assert residual[5] > residual[10]

    def test_uniform_target_peaked_draft(self) -> None:
        """Residual is well-defined when draft is peaked, target is flat."""
        target = _uniform()
        draft = _peaked(0, sharpness=100.0)
        residual = compute_residual_distribution(target, draft)
        assert np.all(residual >= 0.0)
        assert math.isclose(float(residual.sum()), 1.0, abs_tol=1e-9)
        # Token 0: draft >> target → residual[0] should be 0 or very small.
        assert residual[0] < 0.1

    def test_shape_preserved(self) -> None:
        target = _random_dist()
        draft = _random_dist(seed=5)
        residual = compute_residual_distribution(target, draft)
        assert residual.shape == (VOCAB,)


# ---------------------------------------------------------------------------
# sample_from_distribution — temperature and top-p
# ---------------------------------------------------------------------------


class TestSampleFromDistribution:
    def test_temperature_zero_is_greedy(self) -> None:
        """temperature=0 must always return argmax."""
        rng = _rng()
        probs = _random_dist()
        expected = int(np.argmax(probs))
        for _ in range(100):
            assert sample_from_distribution(probs, temperature=0.0, rng=rng) == expected

    def test_temperature_zero_one_hot(self) -> None:
        probs = _one_hot(7)
        assert sample_from_distribution(probs, temperature=0.0) == 7

    def test_temperature_one_samples_from_dist(self) -> None:
        """At temp=1 the empirical distribution should match the input."""
        probs = _random_dist()
        rng = _rng()
        n = 20_000
        counts = np.bincount(
            [sample_from_distribution(probs, temperature=1.0, rng=rng) for _ in range(n)],
            minlength=VOCAB,
        )
        empirical = counts / n
        np.testing.assert_allclose(empirical, probs, atol=0.03)

    def test_high_temperature_flattens(self) -> None:
        """Higher temperature → more uniform; measure entropy."""
        probs = _peaked(0)
        rng = _rng()
        n = 5_000

        def entropy(p: np.ndarray) -> float:
            p = p + 1e-10
            p = p / p.sum()
            return float(-np.sum(p * np.log(p)))

        counts_low = np.bincount(
            [sample_from_distribution(probs, temperature=0.1, rng=rng) for _ in range(n)],
            minlength=VOCAB,
        )
        counts_high = np.bincount(
            [sample_from_distribution(probs, temperature=5.0, rng=rng) for _ in range(n)],
            minlength=VOCAB,
        )
        assert entropy(counts_low / n) < entropy(counts_high / n)

    def test_top_p_one_no_filtering(self) -> None:
        """top_p=1.0 must not alter the distribution."""
        probs = _random_dist()
        rng = _rng()
        n = 10_000
        counts = np.bincount(
            [sample_from_distribution(probs, top_p=1.0, rng=rng) for _ in range(n)],
            minlength=VOCAB,
        )
        np.testing.assert_allclose(counts / n, probs, atol=0.03)

    def test_top_p_small_concentrates_on_high_prob_tokens(self) -> None:
        """With small top_p, only highest-probability tokens are sampled."""
        # Give tokens 0..4 most mass, rest tiny.
        probs = np.full(VOCAB, 1e-4)
        probs[:5] = 0.18  # 5 × 0.18 = 0.9 → 90% mass
        probs /= probs.sum()
        top_5_idx = set(range(5))

        rng = _rng()
        n = 1_000
        samples = {sample_from_distribution(probs, top_p=0.91, rng=rng) for _ in range(n)}
        # All samples should come from the top-5 tokens.
        assert samples.issubset(top_5_idx), f"Got tokens outside top-5: {samples - top_5_idx}"

    def test_top_p_always_returns_at_least_one_token(self) -> None:
        """Even with top_p=0.0 we must not error out."""
        probs = _random_dist()
        token = sample_from_distribution(probs, top_p=0.0)
        assert 0 <= token < VOCAB

    def test_returns_valid_index(self) -> None:
        probs = _random_dist()
        rng = _rng()
        for _ in range(200):
            t = sample_from_distribution(probs, rng=rng)
            assert 0 <= t < VOCAB


# ---------------------------------------------------------------------------
# apply_temperature
# ---------------------------------------------------------------------------


class TestApplyTemperature:
    def test_output_is_valid_distribution(self) -> None:
        logits = np.random.default_rng(0).standard_normal(VOCAB)
        probs = apply_temperature(logits, temperature=1.0)
        assert np.all(probs >= 0.0)
        assert math.isclose(float(probs.sum()), 1.0, abs_tol=1e-9)

    def test_temperature_zero_is_one_hot(self) -> None:
        logits = np.array([1.0, 5.0, 2.0, 0.5])
        probs = apply_temperature(logits, temperature=0.0)
        assert probs[1] == pytest.approx(1.0)
        assert probs.sum() == pytest.approx(1.0)

    def test_temperature_one_is_softmax(self) -> None:
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        probs = apply_temperature(logits, temperature=1.0)
        # Reference softmax.
        exp_l = np.exp(logits - logits.max())
        expected = exp_l / exp_l.sum()
        np.testing.assert_allclose(probs, expected, atol=1e-12)

    def test_high_temperature_approaches_uniform(self) -> None:
        logits = np.array([10.0, 0.0, 0.0, 0.0])
        probs = apply_temperature(logits, temperature=1_000.0)
        np.testing.assert_allclose(probs, [0.25, 0.25, 0.25, 0.25], atol=0.01)

    def test_low_temperature_sharpens(self) -> None:
        logits = np.array([1.0, 1.1, 0.9])
        probs_low = apply_temperature(logits, temperature=0.01)
        probs_high = apply_temperature(logits, temperature=10.0)
        # Low temp → highest logit dominates.
        assert probs_low[1] > probs_high[1]
        assert probs_low.max() > probs_high.max()

    def test_shape_preserved(self) -> None:
        logits = np.zeros(VOCAB)
        probs = apply_temperature(logits, temperature=1.0)
        assert probs.shape == (VOCAB,)

    def test_large_logit_values_no_overflow(self) -> None:
        """Very large logits must not produce NaN via exp overflow."""
        logits = np.array([1e6, 0.0, -1e6])
        probs = apply_temperature(logits, temperature=1.0)
        assert not np.any(np.isnan(probs))
        assert math.isclose(float(probs.sum()), 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Edge cases and numerical robustness
# ---------------------------------------------------------------------------


class TestNumericalEdgeCases:
    def test_uniform_target_and_draft(self) -> None:
        """Uniform distributions — everything should work without error."""
        u = _uniform()
        rng = _rng()
        accepted, token = rejection_sample(u, u, 0, rng=rng)
        assert accepted  # p/q = 1 → always accept
        residual = compute_residual_distribution(u, u)
        np.testing.assert_allclose(residual, u, atol=1e-9)

    def test_one_hot_target_and_draft_same_token(self) -> None:
        target = _one_hot(3)
        draft = _one_hot(3)
        rng = _rng()
        for _ in range(50):
            accepted, token = rejection_sample(target, draft, 3, rng=rng)
            assert accepted
            assert token == 3

    def test_one_hot_target_draft_different_token(self) -> None:
        """Draft proposes token with zero target probability → always reject."""
        target = _one_hot(3)
        draft = _one_hot(5)
        rng = _rng()
        for _ in range(100):
            accepted, token = rejection_sample(target, draft, 5, rng=rng)
            assert not accepted
            assert token == 3  # correction must come from target (token 3)

    def test_very_peaked_distribution(self) -> None:
        probs = _peaked(0, sharpness=50.0)
        rng = _rng()
        n = 500
        tokens = [sample_from_distribution(probs, rng=rng) for _ in range(n)]
        # Almost all tokens should be 0.
        assert tokens.count(0) / n > 0.95

    def test_nan_input_handled_gracefully(self) -> None:
        """NaN values in input should be replaced and not crash."""
        probs = _random_dist()
        probs_with_nan = probs.copy()
        probs_with_nan[0] = float("nan")
        # Should not raise; result may differ from clean input but must be valid.
        token = sample_from_distribution(probs_with_nan, rng=_rng())
        assert 0 <= token < VOCAB

    def test_inf_logits_handled_gracefully(self) -> None:
        logits = np.array([1.0, float("inf"), -float("inf")])
        probs = apply_temperature(logits, temperature=1.0)
        # After handling inf, result must still be a valid distribution.
        assert not np.any(np.isnan(probs))
        assert math.isclose(float(probs.sum()), 1.0, abs_tol=1e-9)

    def test_zero_draft_probability_at_draft_token(self) -> None:
        """Draft assigns zero prob to proposed token → acceptance ratio = 1."""
        # Build target and draft such that draft_token has provably zero mass
        # in the draft.  Use a one-hot draft pointed at a different token so
        # the zeroing is unambiguous and no normalisation heuristic can restore
        # mass to draft_token.
        target = _random_dist()
        draft_token = 3
        # Draft is a one-hot at token 5 (≠ draft_token=3) → q(3) == 0 exactly.
        draft = _one_hot(5)
        rng = _rng()
        accepted, token = rejection_sample(target, draft, draft_token, rng=rng)
        # q(draft_token) == 0 → acceptance_prob == 1.0 → must accept.
        assert accepted
        assert token == draft_token

    def test_unnormalised_probs_accepted(self) -> None:
        """sample_from_distribution should handle unnormalised inputs."""
        probs = np.array([2.0, 4.0, 6.0])  # sums to 12, not 1
        rng = _rng()
        for _ in range(100):
            t = sample_from_distribution(probs, rng=rng)
            assert 0 <= t < 3

    def test_single_token_vocab(self) -> None:
        """Vocab size 1 — only one possible token, must always return 0."""
        probs = np.array([1.0])
        assert sample_from_distribution(probs) == 0
        assert apply_temperature(np.array([0.0]), temperature=1.0)[0] == pytest.approx(1.0)

    def test_reproducibility_with_same_seed(self) -> None:
        """Same seed → same sequence of tokens."""
        probs = _random_dist()
        rng1 = _rng(seed=99)
        rng2 = _rng(seed=99)
        tokens1 = [sample_from_distribution(probs, rng=rng1) for _ in range(50)]
        tokens2 = [sample_from_distribution(probs, rng=rng2) for _ in range(50)]
        assert tokens1 == tokens2

    def test_different_seeds_different_sequences(self) -> None:
        probs = _random_dist()
        rng1 = _rng(seed=1)
        rng2 = _rng(seed=2)
        tokens1 = [sample_from_distribution(probs, rng=rng1) for _ in range(50)]
        tokens2 = [sample_from_distribution(probs, rng=rng2) for _ in range(50)]
        # Astronomically unlikely to collide.
        assert tokens1 != tokens2
