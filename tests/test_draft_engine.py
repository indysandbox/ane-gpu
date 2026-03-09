"""Tests for the draft model engine.

Unit tests use mock models and run in milliseconds.
Integration tests (marked @pytest.mark.slow) load the real Qwen3.5-0.8B model
via MLX and require ~2GB of free memory plus a network connection on first run.

Quick run (no model download):
    pytest tests/test_draft_engine.py -m "not slow" -v

Full run (loads real model, ~30s):
    pytest tests/test_draft_engine.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


FAKE_VOCAB_SIZE = 1000
FAKE_SEQ_LEN = 8


def _uniform_probs(vocab_size: int = FAKE_VOCAB_SIZE) -> np.ndarray:
    """Return a valid uniform probability distribution as float32."""
    return np.full(vocab_size, 1.0 / vocab_size, dtype=np.float32)


def _make_fake_logits(vocab_size: int, seq_len: int, batch: int = 1) -> mx.array:
    """Return a deterministic MLX logits tensor shaped (batch, seq_len, vocab_size).

    All logits are 0 so softmax gives uniform probabilities — easy to assert on.
    """
    return mx.zeros((batch, seq_len, vocab_size))


def _build_mock_mlx_model(vocab_size: int = FAKE_VOCAB_SIZE) -> MagicMock:
    """Return a minimal mock that looks like an mlx_lm model.

    The mock's __call__ always returns zero logits of shape
    (1, variable_seq_len, vocab_size).
    """
    model = MagicMock(name="mlx_model")

    # embed_tokens.weight.shape[0] is how DraftEngine reads vocab size
    embed_tokens = MagicMock()
    embed_tokens.weight = MagicMock()
    embed_tokens.weight.shape = (vocab_size, 64)
    model.model.embed_tokens = embed_tokens

    def _forward(input_ids: mx.array, **kwargs) -> mx.array:
        seq_len = input_ids.shape[1]
        return _make_fake_logits(vocab_size, seq_len)

    model.side_effect = _forward
    return model


def _build_mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock(name="tokenizer")
    tokenizer.encode.return_value = [1, 2, 3]
    return tokenizer


# ---------------------------------------------------------------------------
# Unit tests — no real model loaded
# ---------------------------------------------------------------------------


class TestDraftEngineProperties:
    """Test basic DraftEngine properties on a mock-backed instance."""

    def _make_engine(self, vocab_size: int = FAKE_VOCAB_SIZE) -> "DraftEngine":
        """Construct a DraftEngine with a mocked MLX model."""
        from src.draft.engine import DraftEngine, _MLXState

        state = _MLXState(
            model=_build_mock_mlx_model(vocab_size),
            tokenizer=_build_mock_tokenizer(),
            vocab_size=vocab_size,
        )
        return DraftEngine(backend="mlx", state=state)

    def test_backend_property(self):
        engine = self._make_engine()
        assert engine.backend == "mlx"

    def test_vocab_size_property(self):
        engine = self._make_engine(vocab_size=32000)
        assert engine.vocab_size == 32000

    def test_invalid_backend_raises(self):
        from src.draft.engine import DraftEngine, _MLXState

        state = _MLXState(
            model=_build_mock_mlx_model(),
            tokenizer=_build_mock_tokenizer(),
            vocab_size=FAKE_VOCAB_SIZE,
        )
        with pytest.raises(ValueError, match="backend must be"):
            DraftEngine(backend="cpu", state=state)

    def test_reset_clears_cache(self):
        from src.draft.engine import _MLXState

        engine = self._make_engine()
        state: _MLXState = engine._state  # type: ignore[assignment]
        state.cache = ["some_cache_object"]
        engine.reset()
        assert state.cache is None


class TestCoreMLBackendStub:
    """from_coreml should raise NotImplementedError with a helpful message."""

    def test_from_coreml_raises(self):
        from src.draft.engine import DraftEngine

        with pytest.raises(NotImplementedError, match="Core ML conversion not yet available"):
            DraftEngine.from_coreml("models/draft.mlpackage", seq_len=128)


class TestFromMlxConstructor:
    """Test DraftEngine.from_mlx() without actually loading weights."""

    def test_from_mlx_loads_model(self):
        """from_mlx should invoke mlx_lm.load and wire up state correctly.

        We intercept the deferred ``from mlx_lm import load`` inside from_mlx
        by patching the fully-qualified name that Python resolves at call time.
        """
        from src.draft.engine import DraftEngine

        mock_model = _build_mock_mlx_model(vocab_size=FAKE_VOCAB_SIZE)
        mock_tokenizer = _build_mock_tokenizer()

        # Patch the deferred import target: mlx_lm.load is imported inside from_mlx
        # as `from mlx_lm import load`, so patching `mlx_lm.load` is sufficient.
        with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)):
            engine = DraftEngine.from_mlx("fake/model-id")

        assert engine.backend == "mlx"
        assert engine.vocab_size == FAKE_VOCAB_SIZE


class TestPredictNext:
    """Tests for DraftEngine.predict_next()."""

    @pytest.fixture
    def engine(self) -> "DraftEngine":
        from src.draft.engine import DraftEngine, _MLXState

        state = _MLXState(
            model=_build_mock_mlx_model(FAKE_VOCAB_SIZE),
            tokenizer=_build_mock_tokenizer(),
            vocab_size=FAKE_VOCAB_SIZE,
        )
        return DraftEngine(backend="mlx", state=state)

    def test_returns_tuple(self, engine):
        result = engine.predict_next([1, 2, 3])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_token_is_int(self, engine):
        token, _ = engine.predict_next([1, 2, 3])
        assert isinstance(token, int)

    def test_token_in_vocab_range(self, engine):
        token, _ = engine.predict_next([1, 2, 3])
        assert 0 <= token < FAKE_VOCAB_SIZE

    def test_distribution_shape(self, engine):
        _, probs = engine.predict_next([1, 2, 3])
        assert probs.shape == (FAKE_VOCAB_SIZE,)

    def test_distribution_non_negative(self, engine):
        _, probs = engine.predict_next([1, 2, 3])
        assert np.all(probs >= 0), "All probabilities must be non-negative"

    def test_distribution_sums_to_one(self, engine):
        _, probs = engine.predict_next([1, 2, 3])
        assert abs(float(probs.sum()) - 1.0) < 1e-4, (
            f"Probabilities must sum to ~1.0, got {probs.sum()}"
        )

    def test_distribution_is_numpy(self, engine):
        _, probs = engine.predict_next([1, 2, 3])
        assert isinstance(probs, np.ndarray), "Distribution must be a numpy array"

    def test_uniform_logits_give_uniform_probs(self, engine):
        """Zero logits -> softmax -> uniform distribution."""
        _, probs = engine.predict_next([1, 2, 3])
        expected = 1.0 / FAKE_VOCAB_SIZE
        # Each probability should be very close to the uniform value
        assert np.allclose(probs, expected, atol=1e-5), (
            f"Expected uniform probs ({expected:.6f}), got max={probs.max():.6f} min={probs.min():.6f}"
        )


class TestPropose:
    """Tests for DraftEngine.propose()."""

    @pytest.fixture
    def engine(self) -> "DraftEngine":
        from src.draft.engine import DraftEngine, _MLXState

        state = _MLXState(
            model=_build_mock_mlx_model(FAKE_VOCAB_SIZE),
            tokenizer=_build_mock_tokenizer(),
            vocab_size=FAKE_VOCAB_SIZE,
        )
        return DraftEngine(backend="mlx", state=state)

    def test_returns_correct_count_k1(self, engine):
        tokens, dists = engine.propose([1, 2, 3], k=1)
        assert len(tokens) == 1
        assert len(dists) == 1

    def test_returns_correct_count_k5(self, engine):
        tokens, dists = engine.propose([1, 2, 3], k=5)
        assert len(tokens) == 5
        assert len(dists) == 5

    def test_returns_correct_count_k10(self, engine):
        tokens, dists = engine.propose([1, 2, 3], k=10)
        assert len(tokens) == 10
        assert len(dists) == 10

    def test_tokens_and_dists_lengths_match(self, engine):
        for k in (1, 3, 7):
            tokens, dists = engine.propose([10, 20, 30], k=k)
            assert len(tokens) == len(dists) == k, (
                f"k={k}: len(tokens)={len(tokens)}, len(dists)={len(dists)}"
            )

    def test_each_dist_has_correct_shape(self, engine):
        _, dists = engine.propose([1, 2, 3], k=5)
        for i, d in enumerate(dists):
            assert d.shape == (FAKE_VOCAB_SIZE,), (
                f"Distribution {i} has wrong shape: {d.shape}"
            )

    def test_each_dist_non_negative(self, engine):
        _, dists = engine.propose([1, 2, 3], k=5)
        for i, d in enumerate(dists):
            assert np.all(d >= 0), f"Distribution {i} contains negative values"

    def test_each_dist_sums_to_one(self, engine):
        _, dists = engine.propose([1, 2, 3], k=5)
        for i, d in enumerate(dists):
            total = float(d.sum())
            assert abs(total - 1.0) < 1e-4, (
                f"Distribution {i} sums to {total:.6f}, expected ~1.0"
            )

    def test_each_dist_is_numpy(self, engine):
        _, dists = engine.propose([1, 2, 3], k=5)
        for i, d in enumerate(dists):
            assert isinstance(d, np.ndarray), (
                f"Distribution {i} is {type(d)}, expected np.ndarray"
            )

    def test_tokens_are_ints_in_vocab_range(self, engine):
        tokens, _ = engine.propose([1, 2, 3], k=5)
        for i, t in enumerate(tokens):
            assert isinstance(t, int), f"Token {i} is {type(t)}, expected int"
            assert 0 <= t < FAKE_VOCAB_SIZE, f"Token {i}={t} out of range [0, {FAKE_VOCAB_SIZE})"

    def test_invalid_k_raises(self, engine):
        with pytest.raises(ValueError, match="k must be >= 1"):
            engine.propose([1, 2, 3], k=0)

    def test_empty_context_raises(self, engine):
        with pytest.raises(ValueError, match="context must be non-empty"):
            engine.propose([], k=3)

    def test_model_called_k_times(self, engine):
        """The mock model should be called once per draft step."""
        from src.draft.engine import _MLXState

        state: _MLXState = engine._state  # type: ignore[assignment]
        call_count_before = state.model.call_count

        engine.propose([1, 2, 3], k=4)

        calls_made = state.model.call_count - call_count_before
        assert calls_made == 4, f"Expected 4 forward passes for k=4, got {calls_made}"

    def test_context_grows_across_steps(self, engine):
        """Each step should use a context one token longer than the previous."""
        from src.draft.engine import _MLXState

        state: _MLXState = engine._state  # type: ignore[assignment]
        call_args_list: list = []

        original_side_effect = state.model.side_effect

        def recording_side_effect(input_ids: mx.array, **kwargs) -> mx.array:
            call_args_list.append(input_ids.shape[1])
            return original_side_effect(input_ids, **kwargs)

        state.model.side_effect = recording_side_effect

        context = [10, 20, 30]
        engine.propose(context, k=3)

        # Lengths should be: len(context), len(context)+1, len(context)+2
        assert call_args_list == [3, 4, 5], (
            f"Expected context lengths [3, 4, 5], got {call_args_list}"
        )


class TestReset:
    """Tests for DraftEngine.reset()."""

    def test_reset_idempotent(self):
        from src.draft.engine import DraftEngine, _MLXState

        state = _MLXState(
            model=_build_mock_mlx_model(),
            tokenizer=_build_mock_tokenizer(),
            vocab_size=FAKE_VOCAB_SIZE,
        )
        engine = DraftEngine(backend="mlx", state=state)
        # Calling reset multiple times should not raise
        engine.reset()
        engine.reset()

    def test_reset_after_propose(self):
        """After propose + reset, internal cache should be cleared."""
        from src.draft.engine import DraftEngine, _MLXState

        state = _MLXState(
            model=_build_mock_mlx_model(),
            tokenizer=_build_mock_tokenizer(),
            vocab_size=FAKE_VOCAB_SIZE,
        )
        engine = DraftEngine(backend="mlx", state=state)
        engine.propose([1, 2, 3], k=2)
        engine.reset()
        assert state.cache is None


# ---------------------------------------------------------------------------
# Integration test — loads actual Qwen3.5-0.8B model
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDraftEngineIntegration:
    """Integration tests that load the real Qwen3.5-0.8B model via MLX.

    Requires network access on first run (~1.5GB download).
    Runtime: ~20-60s depending on hardware.
    """

    DRAFT_MODEL_ID = "mlx-community/Qwen3.5-0.8B-4bit"
    CONTEXT = "The speed of light in a vacuum is approximately"

    @pytest.fixture(scope="class")
    def engine(self):
        """Load the draft engine once for all integration tests."""
        from src.draft.engine import DraftEngine

        return DraftEngine.from_mlx(self.DRAFT_MODEL_ID)

    @pytest.fixture(scope="class")
    def context_ids(self, engine):
        """Tokenize the test context."""
        return engine._state.tokenizer.encode(self.CONTEXT)

    def test_model_loads(self, engine):
        assert engine.backend == "mlx"
        assert engine.vocab_size > 0

    def test_vocab_size_matches_qwen35(self, engine):
        # Qwen3.5 vocabulary is 151,936 tokens (BPE + specials)
        assert engine.vocab_size > 100_000, (
            f"Vocab size {engine.vocab_size} seems too small for Qwen3.5"
        )

    def test_predict_next_returns_valid_distribution(self, engine, context_ids):
        token, probs = engine.predict_next(context_ids)

        assert isinstance(token, int)
        assert 0 <= token < engine.vocab_size

        assert isinstance(probs, np.ndarray)
        assert probs.shape == (engine.vocab_size,)
        assert np.all(probs >= 0), "Probabilities must be non-negative"
        assert abs(float(probs.sum()) - 1.0) < 1e-3, (
            f"Probabilities must sum to ~1.0, got {probs.sum()}"
        )

    def test_propose_returns_five_tokens(self, engine, context_ids):
        tokens, dists = engine.propose(context_ids, k=5)
        assert len(tokens) == 5
        assert len(dists) == 5

    def test_propose_distributions_are_valid(self, engine, context_ids):
        """Each draft distribution must be a valid probability distribution."""
        _, dists = engine.propose(context_ids, k=5)

        for i, d in enumerate(dists):
            assert isinstance(d, np.ndarray), f"dist[{i}] is not a numpy array"
            assert d.shape == (engine.vocab_size,), (
                f"dist[{i}] shape {d.shape} != ({engine.vocab_size},)"
            )
            assert np.all(d >= 0), f"dist[{i}] contains negative probabilities"
            total = float(d.sum())
            assert abs(total - 1.0) < 1e-3, (
                f"dist[{i}] sums to {total:.6f}, expected ~1.0"
            )

    def test_propose_tokens_in_vocab_range(self, engine, context_ids):
        tokens, _ = engine.propose(context_ids, k=5)
        for i, t in enumerate(tokens):
            assert isinstance(t, int), f"token[{i}] is {type(t)}, expected int"
            assert 0 <= t < engine.vocab_size, (
                f"token[{i}]={t} out of range [0, {engine.vocab_size})"
            )

    def test_reset_allows_reuse(self, engine, context_ids):
        """reset() should allow propose() to be called again without errors."""
        engine.propose(context_ids, k=3)
        engine.reset()
        tokens, dists = engine.propose(context_ids, k=3)
        assert len(tokens) == 3
        assert len(dists) == 3

    def test_propose_consistency(self, engine, context_ids):
        """Two independent propose() calls should return plausible tokens
        (this is stochastic, so we just check structural validity)."""
        tokens_a, dists_a = engine.propose(context_ids, k=4)
        engine.reset()
        tokens_b, dists_b = engine.propose(context_ids, k=4)

        # Both calls should produce 4 structurally valid results
        for tokens, dists in [(tokens_a, dists_a), (tokens_b, dists_b)]:
            assert len(tokens) == 4
            for d in dists:
                assert d.shape == (engine.vocab_size,)
                assert abs(float(d.sum()) - 1.0) < 1e-3
