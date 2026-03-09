"""Tests for the target (GPU/MLX) engine.

Tests marked @pytest.mark.slow require loading the actual 27B model (~17GB).
Run with: pytest tests/test_target_engine.py -m "not slow" for quick tests.
Run with: pytest tests/test_target_engine.py for full tests (needs ~17GB GPU memory).
"""

import numpy as np
import pytest


class TestTargetEngineUnit:
    """Unit tests that don't require model loading."""

    def test_verification_result_properties(self):
        """VerificationResult should report correct number of positions."""
        from src.target.engine import VerificationResult

        dists = [np.ones(100) / 100 for _ in range(5)]
        result = VerificationResult(distributions=dists, verify_time_ms=10.0)
        assert result.num_positions == 5
        assert result.verify_time_ms == 10.0


@pytest.mark.slow
class TestTargetEngineIntegration:
    """Integration tests that load the actual model.

    These tests are slow (~30s+ to load model) and memory-intensive (~17GB).
    """

    @pytest.fixture(scope="class")
    def engine(self):
        """Load the target engine once for all tests in this class."""
        from src.target.engine import TargetEngine

        return TargetEngine("mlx-community/Qwen3.5-27B-4bit")

    def test_model_loads(self, engine):
        """Model should load successfully."""
        assert engine.vocab_size > 0
        assert engine.vocab_size == 248077  # Qwen3.5 vocab size

    def test_prefill(self, engine):
        """Prefill should produce a valid KV cache."""
        # Simple prompt
        tokens = engine.tokenizer.encode("Hello, world!")
        cache = engine.prefill(tokens)

        assert cache is not None
        assert len(cache) > 0  # Should have cache entries for each layer
        assert engine.get_cache_length(cache) == len(tokens)

    def test_verify_returns_distributions(self, engine):
        """Verify should return probability distributions for each draft position."""
        prompt = engine.tokenizer.encode("The capital of France is")
        cache = engine.prefill(prompt)

        # Fake draft tokens (just some valid token IDs)
        draft_tokens = [100, 200, 300, 400, 500]
        result = engine.verify(draft_tokens, cache)

        # Should get 5 distributions (one per draft token)
        assert result.num_positions == 5

        # Each distribution should be a valid probability distribution
        for dist in result.distributions:
            assert dist.shape == (engine.vocab_size,)
            assert np.all(dist >= 0), "Probabilities must be non-negative"
            assert abs(dist.sum() - 1.0) < 1e-4, f"Probabilities must sum to 1, got {dist.sum()}"

        assert result.verify_time_ms > 0

    def test_cache_truncation(self, engine):
        """Cache truncation should allow re-verification from an earlier position."""
        prompt = engine.tokenizer.encode("Once upon a time")
        cache = engine.prefill(prompt)
        original_length = engine.get_cache_length(cache)

        # Verify some draft tokens
        draft_tokens = [100, 200, 300]
        engine.verify(draft_tokens, cache)

        # Cache should now be longer
        assert engine.get_cache_length(cache) == original_length + len(draft_tokens)

        # Truncate back to original length
        engine.truncate_cache(cache, original_length)
        assert engine.get_cache_length(cache) == original_length
