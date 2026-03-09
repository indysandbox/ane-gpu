"""Tests for the speculative decoding scheduler.

All tests use mock draft and target engines — no real models are loaded.
The mocks are designed to exercise deterministic accept/reject behavior by
controlling the draft and target probability distributions directly.

Key insight for determinism: rejection_sample accepts a draft token with
probability min(1, p_target(x) / p_draft(x)). Setting target == draft
guarantees acceptance (ratio = 1). Setting target probability at the draft
token to zero guarantees rejection.

Quick run (all tests, no model download):
    pytest tests/test_scheduler.py -v
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from src.orchestrator.scheduler import (
    GenerationSummary,
    SpeculationRoundStats,
    SpeculativeScheduler,
)
from src.target.engine import VerificationResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 50  # Small enough for readable tests


# ---------------------------------------------------------------------------
# Helpers — probability distribution constructors
# ---------------------------------------------------------------------------


def _uniform(vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Uniform distribution over vocab."""
    return np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64)


def _one_hot(token_id: int, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Distribution with all mass on token_id (greedy certainty)."""
    dist = np.zeros(vocab_size, dtype=np.float64)
    dist[token_id] = 1.0
    return dist


def _zero_at(token_id: int, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Distribution with zero probability at token_id — guarantees rejection.

    Distributes the remaining mass uniformly over all other tokens.
    """
    dist = np.ones(vocab_size, dtype=np.float64)
    dist[token_id] = 0.0
    dist = dist / dist.sum()
    return dist


# ---------------------------------------------------------------------------
# Mock engines
# ---------------------------------------------------------------------------


@dataclass
class MockDraftEngine:
    """Configurable mock draft engine.

    Attributes:
        token_sequence:
            Flat list of tokens to yield, one per propose() call slot.
            Each call to propose(context, k) pops the next k tokens.
        distributions:
            Matching list of distributions, one per token.  If not provided,
            defaults to uniform for every token.
        reset_call_count:
            Incremented each time reset() is called (for assertion).
    """

    token_sequence: list[int] = field(default_factory=list)
    distributions: list[np.ndarray] = field(default_factory=list)
    reset_call_count: int = field(default=0, init=False)
    _pos: int = field(default=0, init=False)

    def propose(
        self, context: list[int], k: int
    ) -> tuple[list[int], list[np.ndarray]]:
        tokens = []
        dists = []
        for _ in range(k):
            if self._pos < len(self.token_sequence):
                tok = self.token_sequence[self._pos]
                if self._pos < len(self.distributions):
                    dist = self.distributions[self._pos]
                else:
                    dist = _uniform()
            else:
                # Fall back to token 0 with uniform distribution if exhausted
                tok = 0
                dist = _uniform()
            tokens.append(tok)
            dists.append(dist)
            self._pos += 1
        return tokens, dists

    def reset(self) -> None:
        self.reset_call_count += 1


@dataclass
class MockCache:
    """Opaque cache object used to verify truncate_cache is called correctly."""

    length: int = 0


@dataclass
class MockTargetEngine:
    """Configurable mock target engine.

    Each call to verify() pops one VerificationResult from ``results``.
    If the list is exhausted, returns K+1 uniform distributions.
    """

    results: list[VerificationResult] = field(default_factory=list)
    truncate_call_count: int = field(default=0, init=False)
    _result_pos: int = field(default=0, init=False)

    def prefill(self, tokens: list[int]) -> MockCache:
        return MockCache(length=len(tokens))

    def verify(
        self, draft_tokens: list[int], cache: Any
    ) -> VerificationResult:
        if self._result_pos < len(self.results):
            result = self.results[self._result_pos]
            self._result_pos += 1
            return result
        # Default: K+1 uniform distributions (all-accept scenario)
        k = len(draft_tokens)
        dists = [_uniform() for _ in range(k + 1)]
        return VerificationResult(distributions=dists, verify_time_ms=1.0)

    def truncate_cache(self, cache: MockCache, length: int) -> MockCache:
        self.truncate_call_count += 1
        cache.length = length
        return cache


# ---------------------------------------------------------------------------
# Helpers — scheduler factory
# ---------------------------------------------------------------------------


def _make_scheduler(
    *,
    draft: MockDraftEngine,
    target: MockTargetEngine,
    k: int = 3,
    eos_token_id: int | None = None,
) -> SpeculativeScheduler:
    return SpeculativeScheduler(
        draft_engine=draft,
        target_engine=target,
        k=k,
        eos_token_id=eos_token_id,
    )


def _run_all(
    scheduler: SpeculativeScheduler,
    prompt: list[int] | None = None,
    max_tokens: int = 20,
    seed: int = 42,
) -> tuple[list[int], list[SpeculationRoundStats]]:
    """Collect all yielded tokens and stats from the scheduler."""
    if prompt is None:
        prompt = [1, 2, 3]
    all_tokens: list[int] = []
    all_stats: list[SpeculationRoundStats] = []
    for tokens, stats in scheduler.generate(prompt, max_tokens=max_tokens, seed=seed):
        all_tokens.extend(tokens)
        all_stats.append(stats)
    return all_tokens, all_stats


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestSchedulerInit:
    """Initialization edge cases."""

    def test_valid_k(self):
        draft = MockDraftEngine(token_sequence=[1] * 20)
        target = MockTargetEngine()
        sched = _make_scheduler(draft=draft, target=target, k=5)
        assert sched.k == 5

    def test_k_one_is_valid(self):
        draft = MockDraftEngine(token_sequence=[1] * 10)
        target = MockTargetEngine()
        sched = _make_scheduler(draft=draft, target=target, k=1)
        assert sched.k == 1

    def test_k_zero_raises(self):
        draft = MockDraftEngine()
        target = MockTargetEngine()
        with pytest.raises(ValueError, match="k must be >= 1"):
            SpeculativeScheduler(draft_engine=draft, target_engine=target, k=0)

    def test_k_negative_raises(self):
        draft = MockDraftEngine()
        target = MockTargetEngine()
        with pytest.raises(ValueError, match="k must be >= 1"):
            SpeculativeScheduler(draft_engine=draft, target_engine=target, k=-1)

    def test_empty_prompt_raises(self):
        draft = MockDraftEngine(token_sequence=[1] * 10)
        target = MockTargetEngine()
        sched = _make_scheduler(draft=draft, target=target)
        with pytest.raises(ValueError, match="prompt_tokens must be non-empty"):
            list(sched.generate([], max_tokens=5))


class TestBasicGenerationLoop:
    """The loop yields tokens and stats; basic structural invariants."""

    def test_yields_tokens(self):
        # All-accept scenario: matching distributions for all draft tokens.
        k = 3
        draft_tokens = [10, 11, 12] * 10
        # Each draft distribution puts all mass on the corresponding token (one-hot).
        draft_dists = [_one_hot(t) for t in draft_tokens]
        # Target matches draft exactly → guaranteed acceptance.
        # verify() returns K+1 distributions; the K-th is the bonus position.
        draft = MockDraftEngine(token_sequence=draft_tokens, distributions=draft_dists)

        results = []
        for i in range(0, len(draft_tokens), k):
            chunk_tokens = draft_tokens[i : i + k]
            # K distributions matching draft + 1 bonus distribution
            dists = [_one_hot(t) for t in chunk_tokens] + [_one_hot(20)]
            results.append(
                VerificationResult(distributions=dists, verify_time_ms=1.0)
            )
        target = MockTargetEngine(results=results)

        sched = _make_scheduler(draft=draft, target=target, k=k)
        all_tokens, all_stats = _run_all(sched, max_tokens=6, seed=0)

        assert len(all_tokens) > 0
        assert len(all_stats) > 0

    def test_yields_tuple_of_tokens_and_stats(self):
        draft = MockDraftEngine(token_sequence=[5] * 20, distributions=[_one_hot(5)] * 20)
        dists = [_one_hot(5)] * 4  # K=3 + 1 bonus
        target = MockTargetEngine(
            results=[VerificationResult(distributions=dists, verify_time_ms=1.0)] * 10
        )
        sched = _make_scheduler(draft=draft, target=target, k=3)

        for tokens, stats in sched.generate([1, 2, 3], max_tokens=6, seed=0):
            assert isinstance(tokens, list)
            assert isinstance(stats, SpeculationRoundStats)
            break  # Only need one round

    def test_stats_fields_are_populated(self):
        draft = MockDraftEngine(token_sequence=[5] * 20, distributions=[_one_hot(5)] * 20)
        dists = [_one_hot(5)] * 4
        target = MockTargetEngine(
            results=[VerificationResult(distributions=dists, verify_time_ms=2.5)] * 10
        )
        sched = _make_scheduler(draft=draft, target=target, k=3)

        _, stats_list = _run_all(sched, max_tokens=6, seed=0)
        s = stats_list[0]

        assert s.num_proposed >= 1
        assert s.num_accepted >= 0
        assert s.draft_time_ms >= 0.0
        assert s.verify_time_ms >= 0.0
        assert isinstance(s.bonus_token, bool)

    def test_draft_reset_called_each_round(self):
        """reset() on the draft engine is called once per completed round."""
        k = 3
        n_rounds = 2
        draft = MockDraftEngine(
            token_sequence=[5] * (k * n_rounds + 5),
            distributions=[_one_hot(5)] * (k * n_rounds + 5),
        )
        dists = [_one_hot(5)] * (k + 1)
        target = MockTargetEngine(
            results=[VerificationResult(distributions=dists, verify_time_ms=1.0)] * (n_rounds + 5)
        )
        sched = _make_scheduler(draft=draft, target=target, k=k)
        _run_all(sched, max_tokens=k * n_rounds, seed=0)

        # At least one reset per round completed
        assert draft.reset_call_count >= n_rounds

    def test_prefill_is_called_with_prompt(self):
        """prefill() should be called exactly once with the prompt tokens."""
        from unittest.mock import MagicMock

        draft = MockDraftEngine(token_sequence=[1] * 20, distributions=[_one_hot(1)] * 20)
        target = MockTargetEngine(
            results=[
                VerificationResult(
                    distributions=[_one_hot(1)] * 4, verify_time_ms=1.0
                )
            ] * 10
        )

        # Wrap target.prefill to record calls
        original_prefill = target.prefill
        prefill_calls = []

        def recording_prefill(tokens):
            prefill_calls.append(list(tokens))
            return original_prefill(tokens)

        target.prefill = recording_prefill

        prompt = [10, 20, 30]
        sched = _make_scheduler(draft=draft, target=target, k=3)
        _run_all(sched, prompt=prompt, max_tokens=3, seed=0)

        assert len(prefill_calls) == 1
        assert prefill_calls[0] == prompt


class TestAllAcceptedCase:
    """When all K draft tokens are accepted, a bonus token is produced."""

    def _build_all_accept_scheduler(self, k: int = 3, n_rounds: int = 2) -> tuple:
        """Return (scheduler, draft, target) configured for guaranteed full acceptance."""
        total_slots = k * n_rounds + 10
        draft_tok = 7  # The token both draft and target agree on

        draft = MockDraftEngine(
            token_sequence=[draft_tok] * total_slots,
            distributions=[_one_hot(draft_tok)] * total_slots,
        )
        # Target distributions match draft at each position → acceptance ratio 1.
        # K+1 distributions per round: positions 0..K-1 for draft, position K for bonus.
        bonus_tok = 42
        round_dists = [_one_hot(draft_tok)] * k + [_one_hot(bonus_tok)]
        target = MockTargetEngine(
            results=[
                VerificationResult(distributions=round_dists, verify_time_ms=1.0)
            ] * (n_rounds + 5)
        )
        sched = _make_scheduler(draft=draft, target=target, k=k)
        return sched, draft, target

    def test_bonus_token_flag_set(self):
        sched, _, _ = self._build_all_accept_scheduler(k=3)
        _, stats_list = _run_all(sched, max_tokens=4, seed=0)
        # At least one round should have bonus_token=True
        assert any(s.bonus_token for s in stats_list)

    def test_tokens_generated_is_k_plus_one(self):
        k = 3
        sched, _, _ = self._build_all_accept_scheduler(k=k)
        _, stats_list = _run_all(sched, max_tokens=k + 1, seed=0)
        # The first full round should produce K accepted + 1 bonus
        first = stats_list[0]
        assert first.tokens_generated == k + 1

    def test_num_accepted_equals_k(self):
        k = 3
        sched, _, _ = self._build_all_accept_scheduler(k=k)
        _, stats_list = _run_all(sched, max_tokens=k + 1, seed=0)
        first = stats_list[0]
        assert first.num_accepted == k

    def test_acceptance_rate_is_one(self):
        k = 4
        sched, _, _ = self._build_all_accept_scheduler(k=k)
        _, stats_list = _run_all(sched, max_tokens=k + 1, seed=0)
        first = stats_list[0]
        assert first.acceptance_rate == pytest.approx(1.0)

    def test_bonus_token_value_from_target_distribution(self):
        """The bonus token should be sampled from the K-th target distribution.

        We put all mass on a specific token (42) for the bonus position and
        verify that token 42 appears in the output.
        """
        k = 3
        draft_tok = 7
        bonus_tok = 42

        draft = MockDraftEngine(
            token_sequence=[draft_tok] * 20,
            distributions=[_one_hot(draft_tok)] * 20,
        )
        round_dists = [_one_hot(draft_tok)] * k + [_one_hot(bonus_tok)]
        target = MockTargetEngine(
            results=[
                VerificationResult(distributions=round_dists, verify_time_ms=1.0)
            ] * 5
        )
        sched = _make_scheduler(draft=draft, target=target, k=k)

        all_tokens, _ = _run_all(sched, max_tokens=k + 1, seed=0)
        assert bonus_tok in all_tokens


class TestRejectionCase:
    """Rejection at position i stops the round and emits a correction token."""

    def _build_reject_at_i_scheduler(
        self, k: int, reject_at: int
    ) -> tuple[SpeculativeScheduler, MockDraftEngine, MockTargetEngine]:
        """Configure scheduler so rejection occurs at position reject_at.

        Positions 0..reject_at-1 have matching distributions (accepted).
        Position reject_at has zero target probability at the draft token
        (guaranteed rejection).
        """
        draft_tok = 5
        other_tok = 10  # Correction token will come from non-zero positions

        draft = MockDraftEngine(
            token_sequence=[draft_tok] * (k * 5),
            distributions=[_one_hot(draft_tok)] * (k * 5),
        )

        # Build K+1 target distributions for one round
        target_dists: list[np.ndarray] = []
        for i in range(k):
            if i < reject_at:
                # Match draft → accept
                target_dists.append(_one_hot(draft_tok))
            elif i == reject_at:
                # Zero probability at draft_tok → guaranteed rejection
                target_dists.append(_zero_at(draft_tok))
            else:
                # Won't be reached (loop stops after rejection)
                target_dists.append(_uniform())
        # Bonus position (won't be sampled on rejection, but must be present)
        target_dists.append(_uniform())

        target = MockTargetEngine(
            results=[
                VerificationResult(distributions=target_dists, verify_time_ms=1.0)
            ] * 10
        )
        sched = _make_scheduler(draft=draft, target=target, k=k)
        return sched, draft, target

    def test_rejection_at_position_0_yields_one_correction_token(self):
        k = 4
        sched, _, _ = self._build_reject_at_i_scheduler(k=k, reject_at=0)
        all_tokens, stats_list = _run_all(sched, max_tokens=1, seed=0)

        s = stats_list[0]
        assert s.num_accepted == 0
        assert s.bonus_token is False
        # tokens_generated = num_accepted + bonus = 0. The correction token
        # is counted in the emitted list but not in tokens_generated (by design:
        # tokens_generated tracks "useful" accepted tokens for speedup metrics).
        # What matters is that exactly 1 token was emitted.
        assert len(all_tokens) == 1

    def test_rejection_at_position_2_yields_two_accepted_plus_correction(self):
        k = 5
        reject_at = 2
        sched, _, _ = self._build_reject_at_i_scheduler(k=k, reject_at=reject_at)
        all_tokens, stats_list = _run_all(sched, max_tokens=reject_at + 1, seed=0)

        s = stats_list[0]
        assert s.num_accepted == reject_at
        assert s.bonus_token is False
        # The emitted list has reject_at accepted tokens + 1 correction token.
        # tokens_generated property counts only num_accepted (no bonus), so = reject_at.
        assert len(all_tokens) == reject_at + 1

    def test_rejection_no_bonus_token(self):
        k = 4
        sched, _, _ = self._build_reject_at_i_scheduler(k=k, reject_at=1)
        _, stats_list = _run_all(sched, max_tokens=3, seed=0)
        assert stats_list[0].bonus_token is False

    def test_rejection_acceptance_rate(self):
        k = 4
        reject_at = 2
        sched, _, _ = self._build_reject_at_i_scheduler(k=k, reject_at=reject_at)
        # max_tokens must be >= k so K isn't clipped, otherwise actual_k != k
        _, stats_list = _run_all(sched, max_tokens=k + 1, seed=0)
        s = stats_list[0]
        expected_rate = reject_at / k
        assert s.acceptance_rate == pytest.approx(expected_rate)

    def test_correction_token_different_from_draft(self):
        """On rejection, the correction token is drawn from the residual.

        With target = _zero_at(draft_tok), the residual distribution has zero
        mass at draft_tok, so the correction token must differ from draft_tok.
        """
        k = 3
        draft_tok = 5
        draft = MockDraftEngine(
            token_sequence=[draft_tok] * 20,
            distributions=[_one_hot(draft_tok)] * 20,
        )
        # Reject at position 0: target puts all mass on token 20 (not draft_tok).
        correction_tok = 20
        target_dists = [_one_hot(correction_tok)] + [_uniform()] * k  # K+1 total
        target = MockTargetEngine(
            results=[
                VerificationResult(distributions=target_dists, verify_time_ms=1.0)
            ] * 5
        )
        sched = _make_scheduler(draft=draft, target=target, k=k)
        all_tokens, _ = _run_all(sched, max_tokens=1, seed=0)

        # The first (and only) token in this run is the correction token
        assert len(all_tokens) == 1
        assert all_tokens[0] != draft_tok

    def test_cache_truncated_after_rejection(self):
        """truncate_cache is called after rejection to roll back the KV cache."""
        k = 4
        sched, _, target = self._build_reject_at_i_scheduler(k=k, reject_at=0)
        _run_all(sched, max_tokens=1, seed=0)
        assert target.truncate_call_count >= 1


class TestEosHandling:
    """EOS token stops generation immediately."""

    EOS = 49  # Sentinel EOS token (must be < VOCAB_SIZE)

    def _eos_draft(self, tokens_before_eos: int, k: int) -> MockDraftEngine:
        """Draft engine that produces non-EOS tokens, then EOS."""
        seq = [1] * tokens_before_eos + [self.EOS] + [1] * 20
        dists = [_one_hot(t) for t in seq]
        return MockDraftEngine(token_sequence=seq, distributions=dists)

    def test_eos_in_accepted_draft_stops_generation(self):
        """If EOS is accepted at draft position i, the loop should stop."""
        k = 5
        eos_pos = 2  # EOS appears at draft position 2

        draft = MockDraftEngine(
            token_sequence=[1, 1, self.EOS, 1, 1] + [1] * 20,
            distributions=[_one_hot(1), _one_hot(1), _one_hot(self.EOS)]
            + [_one_hot(1)] * 22,
        )
        # Target matches draft at all positions → EOS gets accepted
        target_dists = (
            [_one_hot(1)] * eos_pos
            + [_one_hot(self.EOS)]
            + [_uniform()] * (k - eos_pos)
            + [_uniform()]
        )  # K+1 total
        target = MockTargetEngine(
            results=[
                VerificationResult(distributions=target_dists, verify_time_ms=1.0)
            ] * 5
        )
        sched = _make_scheduler(
            draft=draft, target=target, k=k, eos_token_id=self.EOS
        )

        all_tokens, _ = _run_all(sched, max_tokens=100, seed=0)

        # Generation must have stopped — EOS is in the output
        assert self.EOS in all_tokens
        # No tokens appear after EOS
        eos_idx = all_tokens.index(self.EOS)
        assert all_tokens[eos_idx + 1 :] == []

    def test_eos_as_bonus_token_stops_generation(self):
        """If EOS appears as the bonus token (all K accepted), loop stops."""
        k = 3
        draft_tok = 7

        draft = MockDraftEngine(
            token_sequence=[draft_tok] * 20,
            distributions=[_one_hot(draft_tok)] * 20,
        )
        # All K accepted, bonus = EOS
        round_dists = [_one_hot(draft_tok)] * k + [_one_hot(self.EOS)]
        target = MockTargetEngine(
            results=[
                VerificationResult(distributions=round_dists, verify_time_ms=1.0)
            ] * 5
        )
        sched = _make_scheduler(
            draft=draft, target=target, k=k, eos_token_id=self.EOS
        )

        all_tokens, stats_list = _run_all(sched, max_tokens=100, seed=0)

        assert self.EOS in all_tokens
        eos_idx = all_tokens.index(self.EOS)
        assert all_tokens[eos_idx + 1 :] == []
        # The round that produced EOS must have bonus_token=True
        assert any(s.bonus_token for s in stats_list)

    def test_no_eos_token_id_means_eos_not_treated_specially(self):
        """Without eos_token_id, EOS token does not stop generation."""
        k = 3
        draft = MockDraftEngine(
            token_sequence=[self.EOS] * 20,
            distributions=[_one_hot(self.EOS)] * 20,
        )
        round_dists = [_one_hot(self.EOS)] * k + [_one_hot(self.EOS)]
        target = MockTargetEngine(
            results=[
                VerificationResult(distributions=round_dists, verify_time_ms=1.0)
            ] * 10
        )
        # No eos_token_id — scheduler is unaware of EOS
        sched = _make_scheduler(draft=draft, target=target, k=k, eos_token_id=None)

        all_tokens, _ = _run_all(sched, max_tokens=k + 1, seed=0)
        # Should have generated tokens without stopping early
        assert len(all_tokens) >= k


class TestMaxTokensLimit:
    """max_tokens is respected — generation does not exceed the limit."""

    def test_max_tokens_approximate_stop(self):
        """Total generated tokens should be close to max_tokens.

        Note: the scheduler may overshoot by at most 1 token due to bonus tokens
        (a free token from the verification pass). This is intentional — discarding
        a free token would be wasteful.
        """
        k = 3
        max_tokens = 7

        draft = MockDraftEngine(
            token_sequence=[1] * 50,
            distributions=[_one_hot(1)] * 50,
        )
        round_dists = [_one_hot(1)] * (k + 1)
        target = MockTargetEngine(
            results=[
                VerificationResult(distributions=round_dists, verify_time_ms=1.0)
            ] * 20
        )
        sched = _make_scheduler(draft=draft, target=target, k=k)
        all_tokens, _ = _run_all(sched, max_tokens=max_tokens, seed=0)

        # May overshoot by 1 due to bonus token
        assert len(all_tokens) <= max_tokens + 1

    def test_max_tokens_zero_yields_nothing(self):
        draft = MockDraftEngine(token_sequence=[1] * 20, distributions=[_one_hot(1)] * 20)
        target = MockTargetEngine()
        sched = _make_scheduler(draft=draft, target=target, k=3)
        all_tokens, all_stats = _run_all(sched, max_tokens=0, seed=0)
        assert all_tokens == []
        assert all_stats == []

    def test_max_tokens_one(self):
        """When max_tokens=1, K is clipped to 1 for the first (and only) round.

        With k_clipped=1 and all-accept, the round produces 1 accepted token
        plus 1 bonus token (2 total). The loop then exits because generated >= 1.
        The emitted count may slightly exceed max_tokens when a bonus token is
        added, but only by 1 — that is acceptable scheduler behavior.
        """
        k = 5
        draft = MockDraftEngine(
            token_sequence=[1] * 20,
            distributions=[_one_hot(1)] * 20,
        )
        # Verify supplies K_clipped+1 = 2 dists
        round_dists = [_one_hot(1)] * 2  # K=1 clipped + 1 bonus
        target = MockTargetEngine(
            results=[
                VerificationResult(distributions=round_dists, verify_time_ms=1.0)
            ] * 5
        )
        sched = _make_scheduler(draft=draft, target=target, k=k)
        all_tokens, stats_list = _run_all(sched, max_tokens=1, seed=0)
        # Exactly one round runs with k_clipped=1
        assert len(stats_list) == 1
        assert stats_list[0].num_proposed == 1
        # At most 2 tokens (1 accepted + 1 bonus) from one clipped round
        assert 1 <= len(all_tokens) <= 2


class TestKClipping:
    """K is clipped when remaining budget is less than the configured K."""

    def test_k_is_clipped_near_end(self):
        """When generated + k > max_tokens, the draft engine receives a smaller k.

        We verify this by making the draft engine record how many tokens were
        requested and checking the last round used a clipped k.
        """
        k = 5
        max_tokens = 7  # 7 / 5 → first round k=5, second round k=2

        proposed_counts: list[int] = []

        class RecordingDraftEngine:
            def __init__(self):
                self.reset_call_count = 0

            def propose(self, context, k_arg):
                proposed_counts.append(k_arg)
                toks = [1] * k_arg
                dists = [_one_hot(1)] * k_arg
                return toks, dists

            def reset(self):
                self.reset_call_count += 1

        draft = RecordingDraftEngine()
        # Provide enough VerificationResult objects: first round K=5 (6 dists),
        # second round K=2 (3 dists).
        target = MockTargetEngine(
            results=[
                VerificationResult(
                    distributions=[_one_hot(1)] * 6, verify_time_ms=1.0
                ),
                VerificationResult(
                    distributions=[_one_hot(1)] * 3, verify_time_ms=1.0
                ),
            ]
        )
        sched = SpeculativeScheduler(
            draft_engine=draft, target_engine=target, k=k
        )
        _run_all(sched, max_tokens=max_tokens, seed=0)

        assert k in proposed_counts, f"Expected full k={k} in first round: {proposed_counts}"
        # The last clipped round should have used fewer tokens
        last = proposed_counts[-1]
        assert last < k, f"Last round should have clipped k but got {last}"

    def test_k_clipped_to_remaining(self):
        """Scheduler clips k to (max_tokens - generated) when that is smaller."""
        k = 10
        max_tokens = 3

        proposed_counts: list[int] = []

        class RecordingDraftEngine:
            def __init__(self):
                self.reset_call_count = 0

            def propose(self, context, k_arg):
                proposed_counts.append(k_arg)
                toks = [1] * k_arg
                dists = [_one_hot(1)] * k_arg
                return toks, dists

            def reset(self):
                self.reset_call_count += 1

        draft = RecordingDraftEngine()
        target = MockTargetEngine(
            results=[
                VerificationResult(
                    distributions=[_one_hot(1)] * (max_tokens + 1), verify_time_ms=1.0
                )
            ] * 5
        )
        sched = SpeculativeScheduler(draft_engine=draft, target_engine=target, k=k)
        _run_all(sched, max_tokens=max_tokens, seed=0)

        # First (and only) round should have k clipped to max_tokens
        assert proposed_counts[0] == max_tokens, (
            f"Expected k clipped to {max_tokens}, got {proposed_counts[0]}"
        )


class TestStatsComputation:
    """SpeculationRoundStats and GenerationSummary compute correct statistics."""

    def test_stats_acceptance_rate_all_accepted(self):
        s = SpeculationRoundStats(
            num_proposed=5, num_accepted=5,
            draft_time_ms=1.0, verify_time_ms=1.0, bonus_token=True,
        )
        assert s.acceptance_rate == pytest.approx(1.0)

    def test_stats_acceptance_rate_none_accepted(self):
        s = SpeculationRoundStats(
            num_proposed=5, num_accepted=0,
            draft_time_ms=1.0, verify_time_ms=1.0, bonus_token=False,
        )
        assert s.acceptance_rate == pytest.approx(0.0)

    def test_stats_acceptance_rate_partial(self):
        s = SpeculationRoundStats(
            num_proposed=4, num_accepted=2,
            draft_time_ms=1.0, verify_time_ms=1.0, bonus_token=False,
        )
        assert s.acceptance_rate == pytest.approx(0.5)

    def test_stats_acceptance_rate_zero_proposed(self):
        s = SpeculationRoundStats(
            num_proposed=0, num_accepted=0,
            draft_time_ms=0.0, verify_time_ms=0.0, bonus_token=False,
        )
        assert s.acceptance_rate == pytest.approx(0.0)

    def test_stats_tokens_generated_with_bonus(self):
        s = SpeculationRoundStats(
            num_proposed=3, num_accepted=3,
            draft_time_ms=1.0, verify_time_ms=1.0, bonus_token=True,
        )
        assert s.tokens_generated == 4  # 3 accepted + 1 bonus

    def test_stats_tokens_generated_without_bonus(self):
        s = SpeculationRoundStats(
            num_proposed=3, num_accepted=2,
            draft_time_ms=1.0, verify_time_ms=1.0, bonus_token=False,
        )
        # tokens_generated = num_accepted + (1 if bonus_token else 0) = 2 + 0 = 2
        # The correction token is not counted here (it replaces a rejected draft token).
        assert s.tokens_generated == 2

    def test_summary_records_totals(self):
        summary = GenerationSummary()
        summary.record(
            SpeculationRoundStats(
                num_proposed=3, num_accepted=3,
                draft_time_ms=10.0, verify_time_ms=5.0, bonus_token=True,
            )
        )
        summary.record(
            SpeculationRoundStats(
                num_proposed=3, num_accepted=1,
                draft_time_ms=8.0, verify_time_ms=4.0, bonus_token=False,
            )
        )
        assert summary.total_rounds == 2
        assert summary.total_draft_ms == pytest.approx(18.0)
        assert summary.total_verify_ms == pytest.approx(9.0)
        # Round 1: tokens_generated = num_accepted(3) + bonus(1) = 4
        # Round 2: tokens_generated = num_accepted(1) + bonus(0) = 1
        assert summary.total_tokens == 5

    def test_summary_average_acceptance_rate(self):
        summary = GenerationSummary()
        summary.record(
            SpeculationRoundStats(
                num_proposed=4, num_accepted=4,
                draft_time_ms=1.0, verify_time_ms=1.0, bonus_token=True,
            )
        )
        summary.record(
            SpeculationRoundStats(
                num_proposed=4, num_accepted=0,
                draft_time_ms=1.0, verify_time_ms=1.0, bonus_token=False,
            )
        )
        # (1.0 + 0.0) / 2 = 0.5
        assert summary.average_acceptance_rate == pytest.approx(0.5)

    def test_summary_average_tokens_per_round(self):
        summary = GenerationSummary()
        # Round 1: 2 accepted + 1 bonus = tokens_generated=3
        summary.record(
            SpeculationRoundStats(
                num_proposed=2, num_accepted=2,
                draft_time_ms=1.0, verify_time_ms=1.0, bonus_token=True,
            )
        )
        # Round 2: 2 accepted + 0 bonus = tokens_generated=2
        summary.record(
            SpeculationRoundStats(
                num_proposed=2, num_accepted=2,
                draft_time_ms=1.0, verify_time_ms=1.0, bonus_token=False,
            )
        )
        # (3 + 2) / 2 = 2.5
        assert summary.average_tokens_per_round == pytest.approx(2.5)

    def test_summary_bonus_token_rate(self):
        summary = GenerationSummary()
        for bonus in [True, True, False, True]:
            summary.record(
                SpeculationRoundStats(
                    num_proposed=3, num_accepted=3 if bonus else 2,
                    draft_time_ms=1.0, verify_time_ms=1.0, bonus_token=bonus,
                )
            )
        # 3 out of 4 rounds had bonus
        assert summary.bonus_token_rate == pytest.approx(0.75)

    def test_summary_empty_returns_zero_rates(self):
        summary = GenerationSummary()
        assert summary.average_acceptance_rate == pytest.approx(0.0)
        assert summary.average_tokens_per_round == pytest.approx(0.0)
        assert summary.bonus_token_rate == pytest.approx(0.0)
        assert summary.tokens_per_second == pytest.approx(0.0)

    def test_summary_tokens_per_second(self):
        summary = GenerationSummary(
            total_tokens=100,
            wall_time_s=2.0,
        )
        assert summary.tokens_per_second == pytest.approx(50.0)

    def test_scheduler_stats_match_acceptance(self):
        """End-to-end: stats emitted by scheduler match observed token counts."""
        k = 4
        draft_tok = 3

        draft = MockDraftEngine(
            token_sequence=[draft_tok] * 50,
            distributions=[_one_hot(draft_tok)] * 50,
        )
        bonus_tok = 9
        round_dists = [_one_hot(draft_tok)] * k + [_one_hot(bonus_tok)]
        target = MockTargetEngine(
            results=[
                VerificationResult(distributions=round_dists, verify_time_ms=1.0)
            ] * 10
        )
        sched = _make_scheduler(draft=draft, target=target, k=k)

        all_tokens, stats_list = _run_all(sched, max_tokens=k + 1, seed=0)

        s = stats_list[0]
        assert s.num_proposed == k
        assert s.num_accepted == k
        assert s.bonus_token is True
        assert s.tokens_generated == k + 1
        # Total collected tokens should equal tokens_generated from the round(s)
        assert len(all_tokens) == sum(s.tokens_generated for s in stats_list)
