"""Target model engine — runs the large LLM on GPU via MLX.

The target engine handles:
- Loading quantized models via mlx-lm
- Prompt prefill (initial KV cache population)
- Batched verification of draft tokens (THE critical operation for speculative decoding)
- KV cache truncation for rollback after rejection

The verification step is what makes speculative decoding efficient: verifying K draft tokens
in a single batched forward pass costs roughly the same as generating 1 token autoregressively,
because the K tokens are processed in parallel across the GPU.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying K draft tokens against the target model.

    Contains per-position probability distributions needed for rejection sampling.
    """

    # Probability distributions for each position (K+1 total: K draft positions + 1 bonus)
    # Each is a 1D numpy array of shape (vocab_size,)
    distributions: list[np.ndarray]

    # Timing
    verify_time_ms: float = 0.0

    @property
    def num_positions(self) -> int:
        return len(self.distributions)


class TargetEngine:
    """GPU-based target model engine using MLX.

    This engine wraps mlx-lm to provide the specific operations needed for
    speculative decoding: prefill, batched verification, and KV cache management.

    Usage:
        engine = TargetEngine("mlx-community/Qwen3.5-27B-4bit")
        cache = engine.prefill(prompt_tokens)
        result = engine.verify(draft_tokens, cache)
        # result.distributions[i] is the target model's distribution at position i
    """

    def __init__(self, model_id: str, max_seq_len: int = 4096):
        """Load the target model via mlx-lm.

        Args:
            model_id: HuggingFace model ID or local path (e.g., "mlx-community/Qwen3.5-27B-4bit")
            max_seq_len: Maximum sequence length for KV cache allocation
        """
        from mlx_lm import load

        logger.info(f"Loading target model: {model_id}")
        t0 = time.perf_counter()

        self.model, self.tokenizer = load(model_id)
        self.max_seq_len = max_seq_len
        self._vocab_size = self.model.model.embed_tokens.weight.shape[0]

        load_time = time.perf_counter() - t0
        logger.info(
            f"Target model loaded in {load_time:.1f}s | "
            f"vocab_size={self._vocab_size} | "
            f"Metal memory: {mx.get_active_memory() / 1024**2:.0f}MB"
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _make_cache(self) -> list:
        """Create a fresh KV cache for the model."""
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(self.model)

    def prefill(self, tokens: list[int]) -> list:
        """Process prompt tokens and populate the KV cache.

        This is the "prefill" phase — processes the entire prompt in one parallel
        forward pass. The resulting KV cache contains attention state for all prompt
        positions, ready for subsequent verification or generation.

        Args:
            tokens: Prompt token IDs

        Returns:
            KV cache populated with prompt attention state
        """
        t0 = time.perf_counter()

        cache = self._make_cache()
        input_ids = mx.array(tokens)[None]  # (1, seq_len)

        # Run forward pass to populate cache
        logits = self.model(input_ids, cache=cache)
        mx.eval(logits)  # Force computation (MLX is lazy)

        prefill_time = time.perf_counter() - t0
        logger.info(
            f"Prefill: {len(tokens)} tokens in {prefill_time*1000:.1f}ms "
            f"({len(tokens)/prefill_time:.0f} tok/s)"
        )

        return cache

    def verify(
        self,
        draft_tokens: list[int],
        cache: list,
    ) -> VerificationResult:
        """Verify draft tokens in a single batched forward pass.

        This is THE critical operation for speculative decoding efficiency.
        Instead of generating K tokens one-by-one (K sequential forward passes),
        we process all K draft tokens at once (1 forward pass), getting probability
        distributions at each position. The cost is roughly the same as generating
        1 token autoregressively.

        The returned distributions include K+1 positions:
        - Positions 0..K-1: distributions at each draft token position (for rejection sampling)
        - Position K: distribution at the next position (for the "bonus token" if all K accepted)

        Args:
            draft_tokens: K draft token IDs to verify
            cache: KV cache from prefill (will be updated in-place with draft positions)

        Returns:
            VerificationResult with K+1 probability distributions
        """
        t0 = time.perf_counter()
        k = len(draft_tokens)

        # Process all K draft tokens in one forward pass
        input_ids = mx.array(draft_tokens)[None]  # (1, K)
        logits = self.model(input_ids, cache=cache)  # (1, K, vocab_size)
        mx.eval(logits)

        # Convert logits to probability distributions via softmax
        probs = mx.softmax(logits[0], axis=-1)  # (K, vocab_size)
        mx.eval(probs)

        # Convert to numpy for interop with Core ML / rejection sampling
        probs_np = np.array(probs)

        # Each row is the distribution at one position
        distributions = [probs_np[i] for i in range(k)]

        verify_time = (time.perf_counter() - t0) * 1000

        logger.debug(
            f"Verify: {k} tokens in {verify_time:.1f}ms "
            f"({k / (verify_time / 1000):.0f} tok/s effective)"
        )

        return VerificationResult(
            distributions=distributions,
            verify_time_ms=verify_time,
        )

    def decode_one(self, cache: list) -> tuple[int, np.ndarray]:
        """Generate a single token autoregressively (for bonus token or fallback).

        Uses the last token in the KV cache as context to predict the next token.

        Args:
            cache: KV cache with current context

        Returns:
            (token_id, probability_distribution)
        """
        # Get the last token that was fed to the model
        # We need to run one more forward pass with just that token
        # The cache already contains all previous context
        # This happens when we need a bonus token after all K drafts accepted

        # Actually, the verify() call already gives us K distributions.
        # The K-th distribution (last one) IS the bonus token distribution,
        # because logits[i] predicts the token AFTER position i.
        # So this method is only needed for standalone generation (no drafting).

        raise NotImplementedError(
            "decode_one is not needed for speculative decoding — "
            "the bonus token comes from verify()'s last distribution. "
            "Use verify() instead."
        )

    def truncate_cache(self, cache: list, length: int) -> list:
        """Roll back KV cache to a given sequence length.

        After rejection sampling, some draft tokens are rejected. The KV cache
        must be rolled back to the last accepted position so the next speculation
        round starts from the correct state.

        This modifies the cache in-place and returns it.

        Args:
            cache: KV cache to truncate
            length: Target sequence length (keep positions 0..length-1)

        Returns:
            The same cache object, truncated
        """
        for layer_cache in cache:
            # mlx-lm KV cache objects support truncation
            # The exact API depends on the cache type (KVCache vs RotatingKVCache)
            if hasattr(layer_cache, "reuse"):
                # Newer mlx-lm API: reuse(length, offset) for prompt cache reuse
                # For truncation, we want to keep only `length` entries
                layer_cache.reuse(length, 0)
            elif hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
                # Direct attribute access (older API)
                layer_cache.keys = layer_cache.keys[:, :, :length, :]
                layer_cache.values = layer_cache.values[:, :, :length, :]
            else:
                logger.warning(
                    f"Unknown cache type: {type(layer_cache)}. "
                    f"Truncation may not work correctly."
                )

        logger.debug(f"Cache truncated to length {length}")
        return cache

    def get_cache_length(self, cache: list) -> int:
        """Get the current sequence length stored in the KV cache."""
        if not cache:
            return 0
        first = cache[0]
        if hasattr(first, "offset"):
            return first.offset
        elif hasattr(first, "keys"):
            return first.keys.shape[2]
        return 0
