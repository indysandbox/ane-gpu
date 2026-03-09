"""Speculative decoding main loop — orchestrates draft, verify, and sampling.

This module implements the outer loop that ties the draft engine (ANE/MLX),
the target engine (GPU/MLX), and rejection sampling together into a working
speculative decoding pipeline.

Algorithm reference: Leviathan et al., 2023
"Fast Inference from Transformers via Speculative Decoding"
(https://arxiv.org/abs/2211.17192)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Generator

import numpy as np

from src.orchestrator.sampling import rejection_sample, sample_from_distribution

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class SpeculationRoundStats:
    """Statistics for a single speculation round."""

    num_proposed: int        # K tokens the draft model proposed
    num_accepted: int        # How many draft tokens were accepted
    draft_time_ms: float     # Time spent in draft model
    verify_time_ms: float    # Time spent in target model verification
    bonus_token: bool        # Whether a bonus token was generated (all K accepted)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed tokens that were accepted (ignores bonus)."""
        if self.num_proposed == 0:
            return 0.0
        return self.num_accepted / self.num_proposed

    @property
    def tokens_generated(self) -> int:
        """Total tokens produced in this round (accepted + optional bonus)."""
        return self.num_accepted + (1 if self.bonus_token else 0)


@dataclass
class GenerationSummary:
    """Aggregate statistics across all speculation rounds."""

    total_tokens: int = 0
    total_rounds: int = 0
    total_draft_ms: float = 0.0
    total_verify_ms: float = 0.0
    wall_time_s: float = 0.0
    rounds: list[SpeculationRoundStats] = field(default_factory=list)

    def record(self, stats: SpeculationRoundStats) -> None:
        self.rounds.append(stats)
        self.total_rounds += 1
        self.total_tokens += stats.tokens_generated
        self.total_draft_ms += stats.draft_time_ms
        self.total_verify_ms += stats.verify_time_ms

    @property
    def tokens_per_second(self) -> float:
        if self.wall_time_s <= 0:
            return 0.0
        return self.total_tokens / self.wall_time_s

    @property
    def average_acceptance_rate(self) -> float:
        if not self.rounds:
            return 0.0
        return sum(r.acceptance_rate for r in self.rounds) / len(self.rounds)

    @property
    def average_tokens_per_round(self) -> float:
        if not self.rounds:
            return 0.0
        return sum(r.tokens_generated for r in self.rounds) / len(self.rounds)

    @property
    def bonus_token_rate(self) -> float:
        if not self.rounds:
            return 0.0
        return sum(1 for r in self.rounds if r.bonus_token) / len(self.rounds)


# ---------------------------------------------------------------------------
# EOS sentinel
# ---------------------------------------------------------------------------

_UNSET = object()


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class SpeculativeScheduler:
    """Orchestrates speculative decoding across draft and target engines.

    The scheduler is engine-agnostic: it only requires that the engines
    expose the documented interface (propose / prefill / verify / truncate_cache).
    This means it works with both the ANE (Core ML) draft engine and the
    fallback MLX draft engine.

    Usage::

        scheduler = SpeculativeScheduler(draft_engine, target_engine, k=5)
        for new_tokens, stats in scheduler.generate(prompt_tokens, max_tokens=200):
            print(tokenizer.decode(new_tokens), end="", flush=True)
    """

    def __init__(
        self,
        draft_engine,
        target_engine,
        k: int = 5,
        eos_token_id: int | None = None,
    ) -> None:
        """Initialise the scheduler.

        Args:
            draft_engine:
                Any engine with a ``propose(context_tokens, k)`` method that
                returns ``(draft_tokens: list[int], draft_distributions:
                list[np.ndarray])``.
            target_engine:
                A :class:`src.target.engine.TargetEngine` (or compatible
                object) with ``prefill``, ``verify``, and ``truncate_cache``
                methods.
            k:
                Number of draft tokens to propose per round (speculation
                window).  Typical values: 3–8.
            eos_token_id:
                Vocabulary index of the EOS token.  When generated or
                encountered in draft tokens, the loop stops.  If ``None``,
                EOS detection is disabled (generation stops only at
                ``max_tokens``).
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.draft_engine = draft_engine
        self.target_engine = target_engine
        self.k = k
        self.eos_token_id = eos_token_id

        logger.info(
            "SpeculativeScheduler initialised: k=%d, eos_token_id=%s",
            k,
            eos_token_id,
        )

    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> Generator[tuple[list[int], SpeculationRoundStats], None, None]:
        """Main speculative decoding loop.

        Yields ``(new_tokens, stats)`` at each speculation round.  Callers
        can stream output by decoding and printing *new_tokens* as they arrive.

        Algorithm
        ---------
        1. Draft model proposes K tokens with probability distributions q(x).
        2. Target model verifies all K tokens in one batched forward pass,
           producing p(x) for each position plus one extra (the bonus position).
        3. For each draft token (left to right):
           - Accept with probability min(1, p(x) / q(x)).
           - On rejection: sample a correction token from the residual
             distribution norm(max(0, p(x) - q(x))), then stop.
        4. If all K accepted: sample one bonus token from target's distribution
           at position K (comes free from the verification pass).
        5. Roll back the target's KV cache to the last accepted position.
        6. Reset the draft engine's state to reflect the accepted context.
        7. Yield accepted tokens + stats.

        Args:
            prompt_tokens:
                Token IDs for the prompt.  Must be non-empty.
            max_tokens:
                Maximum number of new tokens to generate.
            temperature:
                Sampling temperature (1.0 = no scaling, <1 = sharper,
                >1 = flatter).
            top_p:
                Nucleus sampling threshold.  1.0 disables nucleus filtering.
            seed:
                Optional integer random seed for NumPy RNG (reproducibility).

        Yields:
            A ``(new_tokens, stats)`` tuple for each speculation round where
            at least one token was produced.  *new_tokens* is a list of
            integer token IDs; *stats* is a :class:`SpeculationRoundStats`.
        """
        if not prompt_tokens:
            raise ValueError("prompt_tokens must be non-empty")

        rng = np.random.default_rng(seed)

        # ----------------------------------------------------------------
        # Prefill target with the prompt
        # ----------------------------------------------------------------
        logger.info("Prefilling target engine with %d prompt tokens", len(prompt_tokens))
        cache = self.target_engine.prefill(prompt_tokens)

        # context_tokens tracks the full sequence (prompt + generated so far).
        # We keep it in sync with the target KV cache at all times.
        context_tokens: list[int] = list(prompt_tokens)
        generated: int = 0

        logger.info("Starting speculative decoding loop (max_tokens=%d, K=%d)", max_tokens, self.k)

        # ----------------------------------------------------------------
        # Main loop
        # ----------------------------------------------------------------
        while generated < max_tokens:
            # Clip K so we never overshoot max_tokens
            k = min(self.k, max_tokens - generated)

            # ---- 1. Draft phase -----------------------------------------
            t_draft_start = time.perf_counter()
            draft_tokens, draft_distributions = self.draft_engine.propose(
                context_tokens, k
            )
            draft_time_ms = (time.perf_counter() - t_draft_start) * 1000

            # Defensive: ensure lists
            draft_tokens = list(draft_tokens)
            draft_distributions = list(draft_distributions)
            actual_k = len(draft_tokens)

            if actual_k == 0:
                logger.warning("Draft engine returned no tokens — stopping")
                break

            logger.debug(
                "Draft proposed %d tokens in %.1f ms: %s",
                actual_k,
                draft_time_ms,
                draft_tokens,
            )

            # ---- 2. Verify phase -----------------------------------------
            # target_engine.verify appends the K draft token positions to the
            # cache and returns K+1 distributions (the +1 is the bonus position).
            t_verify_start = time.perf_counter()
            verification = self.target_engine.verify(draft_tokens, cache)
            verify_time_ms = (time.perf_counter() - t_verify_start) * 1000

            target_distributions: list[np.ndarray] = verification.distributions

            # ---- 3. Rejection sampling -----------------------------------
            accepted_tokens: list[int] = []
            rejected = False

            for i in range(actual_k):
                target_dist = target_distributions[i]
                draft_dist = draft_distributions[i]
                draft_tok = draft_tokens[i]

                accepted_flag, token = rejection_sample(
                    target_prob=target_dist,
                    draft_prob=draft_dist,
                    draft_token=draft_tok,
                    temperature=temperature,
                    rng=rng,
                )

                accepted_tokens.append(token)

                if not accepted_flag:
                    # Correction token already sampled by rejection_sample.
                    # Stop — subsequent draft tokens are invalid.
                    rejected = True
                    logger.debug(
                        "Rejection at position %d (draft=%d → correction=%d)",
                        i,
                        draft_tok,
                        token,
                    )
                    break

                # EOS check on each accepted token
                if self.eos_token_id is not None and token == self.eos_token_id:
                    logger.info("EOS token produced at draft position %d — stopping", i)
                    # Truncate cache to context + accepted so far
                    new_ctx_len = len(context_tokens) + len(accepted_tokens)
                    cache = self.target_engine.truncate_cache(cache, new_ctx_len)
                    context_tokens.extend(accepted_tokens)
                    generated += len(accepted_tokens)
                    stats = SpeculationRoundStats(
                        num_proposed=actual_k,
                        num_accepted=len(accepted_tokens) - 1,  # EOS not a "useful" token
                        draft_time_ms=draft_time_ms,
                        verify_time_ms=verify_time_ms,
                        bonus_token=False,
                    )
                    logger.debug("Round stats: %s", _fmt_stats(stats))
                    yield accepted_tokens, stats
                    return

            # ---- 4. Bonus token ------------------------------------------
            bonus_token_generated = False
            if not rejected and len(target_distributions) > actual_k:
                # All K drafts accepted — the K-th target distribution (0-indexed)
                # is the distribution for the position right after the last draft
                # token, which we get for free from the verification pass.
                bonus_dist = target_distributions[actual_k]
                bonus_tok = sample_from_distribution(
                    bonus_dist,
                    temperature=temperature,
                    top_p=top_p,
                    rng=rng,
                )
                accepted_tokens.append(bonus_tok)
                bonus_token_generated = True
                logger.debug("Bonus token: %d", bonus_tok)

                if self.eos_token_id is not None and bonus_tok == self.eos_token_id:
                    logger.info("EOS token produced as bonus token — stopping")
                    # Truncate cache, update context, yield, return
                    new_ctx_len = len(context_tokens) + len(accepted_tokens)
                    cache = self.target_engine.truncate_cache(cache, new_ctx_len)
                    context_tokens.extend(accepted_tokens)
                    generated += len(accepted_tokens)
                    num_accepted = actual_k  # All K were accepted before bonus
                    stats = SpeculationRoundStats(
                        num_proposed=actual_k,
                        num_accepted=num_accepted,
                        draft_time_ms=draft_time_ms,
                        verify_time_ms=verify_time_ms,
                        bonus_token=True,
                    )
                    logger.debug("Round stats: %s", _fmt_stats(stats))
                    yield accepted_tokens, stats
                    return

            # ---- 5. Cache and context management -------------------------
            # The target KV cache currently extends to
            #   len(context_tokens) + actual_k
            # (because verify() consumed all K draft tokens).
            # We need to truncate it back to the last accepted position.

            new_ctx_len = len(context_tokens) + len(accepted_tokens)
            cache = self.target_engine.truncate_cache(cache, new_ctx_len)

            # Update the running context
            context_tokens.extend(accepted_tokens)
            generated += len(accepted_tokens)

            # Reset draft engine to the new accepted context.
            # The interface only guarantees propose(); if a sync/reset method
            # exists we call it so cached state stays consistent.
            if hasattr(self.draft_engine, "reset"):
                self.draft_engine.reset()

            # ---- 6. Yield stats ------------------------------------------
            num_draft_accepted = (
                actual_k if not rejected else len(accepted_tokens) - 1
            )
            stats = SpeculationRoundStats(
                num_proposed=actual_k,
                num_accepted=num_draft_accepted,
                draft_time_ms=draft_time_ms,
                verify_time_ms=verify_time_ms,
                bonus_token=bonus_token_generated,
            )

            logger.debug("Round stats: %s", _fmt_stats(stats))

            yield accepted_tokens, stats

        logger.info(
            "Speculative decoding complete: %d tokens generated",
            generated,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_stats(stats: SpeculationRoundStats) -> str:
    return (
        f"proposed={stats.num_proposed} accepted={stats.num_accepted} "
        f"bonus={stats.bonus_token} "
        f"draft={stats.draft_time_ms:.1f}ms verify={stats.verify_time_ms:.1f}ms"
    )
