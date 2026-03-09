"""Draft model engine for speculative decoding.

Supports two backends:
- Core ML (.mlpackage) for ANE execution — stub, pending architecture support
- MLX (GPU) for execution as a working fallback when ANE conversion isn't possible

The draft engine's job in speculative decoding is to cheaply propose K candidate
tokens that the target model will then verify in a single batched pass.  To enable
rejection sampling, every proposed token comes with the full probability distribution
the draft model assigned at that position (not just argmax).

Usage (MLX backend):
    engine = DraftEngine.from_mlx("mlx-community/Qwen3.5-0.8B-4bit")
    tokens, dists = engine.propose(context_ids, k=5)
    # dists[i] is np.ndarray of shape (vocab_size,) — needed for rejection sampling
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal state containers
# ---------------------------------------------------------------------------


@dataclass
class _MLXState:
    """Holds the loaded MLX model and tokenizer plus any mutable inference state."""

    model: object  # mlx_lm model (untyped to avoid hard import at module level)
    tokenizer: object  # tokenizer (untyped for the same reason)
    vocab_size: int
    # KV cache for the current context — None means no cache has been built yet
    cache: Optional[list] = field(default=None, repr=False)


@dataclass
class _CoreMLState:
    """Placeholder for future Core ML state.  Currently unused."""

    model_path: str
    seq_len: int
    vocab_size: int


# ---------------------------------------------------------------------------
# DraftEngine
# ---------------------------------------------------------------------------


class DraftEngine:
    """Draft model engine for speculative decoding.

    Supports two backends:
    - Core ML (.mlpackage) for ANE execution
    - MLX for GPU execution (fallback when ANE conversion isn't possible)

    Use the class-method constructors rather than calling __init__ directly:
        engine = DraftEngine.from_mlx("mlx-community/Qwen3.5-0.8B-4bit")
        engine = DraftEngine.from_coreml("models/draft.mlpackage", seq_len=128)
    """

    # ------------------------------------------------------------------
    # Private constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        backend: str,
        state: _MLXState | _CoreMLState,
    ) -> None:
        if backend not in ("mlx", "coreml"):
            raise ValueError(f"backend must be 'mlx' or 'coreml', got {backend!r}")
        self._backend = backend
        self._state = state

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_mlx(cls, model_id: str, max_seq_len: int = 2048) -> "DraftEngine":
        """Load draft model via MLX for GPU execution.

        Args:
            model_id: HuggingFace model ID or local MLX model path.
                      Examples: "mlx-community/Qwen3.5-0.8B-4bit",
                                "Qwen/Qwen3.5-0.8B"
            max_seq_len: Maximum context length.  Informational only for the
                         MLX backend (which doesn't pre-allocate KV caches).

        Returns:
            DraftEngine configured to use the MLX backend.
        """
        from mlx_lm import load  # deferred — heavy import

        logger.info("Loading draft model via MLX: %s", model_id)
        t0 = time.perf_counter()

        model, tokenizer = load(model_id)

        # Determine vocab size from the embedding table
        vocab_size = int(model.model.embed_tokens.weight.shape[0])

        elapsed = time.perf_counter() - t0
        logger.info(
            "Draft model loaded in %.1fs | vocab_size=%d | backend=mlx",
            elapsed,
            vocab_size,
        )

        state = _MLXState(model=model, tokenizer=tokenizer, vocab_size=vocab_size)
        return cls(backend="mlx", state=state)

    @classmethod
    def from_coreml(cls, model_path: str, seq_len: int = 128) -> "DraftEngine":
        """Load draft model via Core ML for ANE execution.

        NOTE: Core ML conversion is not yet available for architectures with
        hybrid attention (Gated Delta Networks / linear_attention).  This
        constructor currently raises NotImplementedError.  The interface is
        defined here so the orchestrator can reference it once the conversion
        pipeline is unblocked.

        Args:
            model_path: Path to the .mlpackage directory.
            seq_len: Fixed sequence length the model was compiled with.

        Returns:
            DraftEngine configured to use the Core ML backend.

        Raises:
            NotImplementedError: Until ANE conversion is resolved.
        """
        raise NotImplementedError(
            "Core ML conversion not yet available for this architecture. "
            "Qwen3.5's hybrid Gated Delta Network layers use ops that "
            "coremltools cannot convert to MIL. "
            "Use DraftEngine.from_mlx() as a GPU-based fallback."
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Vocabulary size reported by the loaded model."""
        return self._state.vocab_size

    @property
    def backend(self) -> str:
        """Active backend: 'mlx' or 'coreml'."""
        return self._backend

    # ------------------------------------------------------------------
    # Core inference API
    # ------------------------------------------------------------------

    def predict_next(self, input_ids: list[int]) -> tuple[int, np.ndarray]:
        """Given a token sequence, return the next token and full probability distribution.

        Runs a single forward pass through the draft model, applies softmax to
        the final-position logits, and samples the next token.

        Args:
            input_ids: Current context as a list of token IDs.

        Returns:
            (next_token, probs) where:
            - next_token: sampled token ID (int)
            - probs: full probability distribution, shape (vocab_size,) as float32 numpy array
        """
        if self._backend == "mlx":
            return self._predict_next_mlx(input_ids)
        else:
            raise NotImplementedError(f"predict_next not implemented for backend {self._backend!r}")

    def propose(
        self,
        context: list[int],
        k: int,
    ) -> tuple[list[int], list[np.ndarray]]:
        """Autoregressively generate k draft tokens with their probability distributions.

        Each step:
        1. Run a forward pass on the current context.
        2. Softmax the last-position logits to get a probability distribution.
        3. Sample the next token from that distribution.
        4. Append the sampled token to the context for the next step.

        The returned probability distributions are required by the orchestrator
        for rejection sampling (Leviathan et al., 2023).

        Args:
            context: Current token context (prompt + any tokens generated so far).
            k: Number of draft tokens to propose.

        Returns:
            (draft_tokens, draft_distributions) where:
            - draft_tokens: list of k token IDs
            - draft_distributions: list of k probability arrays, each shape (vocab_size,)
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not context:
            raise ValueError("context must be non-empty")

        if self._backend == "mlx":
            return self._propose_mlx(context, k)
        else:
            raise NotImplementedError(f"propose not implemented for backend {self._backend!r}")

    def reset(self) -> None:
        """Reset any internal state (KV cache, generation counters, etc.).

        Call this between independent generation requests to ensure the draft
        engine starts from a clean state.
        """
        if self._backend == "mlx":
            state: _MLXState = self._state  # type: ignore[assignment]
            state.cache = None
            logger.debug("MLX draft engine state reset")
        # Core ML backend has no persistent state to clear (stateless .mlpackage)

    # ------------------------------------------------------------------
    # MLX backend internals
    # ------------------------------------------------------------------

    def _logits_to_probs(self, logits_last: mx.array) -> np.ndarray:
        """Convert a single-position logit vector to a numpy probability array.

        Args:
            logits_last: 1-D MLX array of shape (vocab_size,).

        Returns:
            1-D float32 numpy array of shape (vocab_size,), summing to ~1.
        """
        probs_mx = mx.softmax(logits_last.astype(mx.float32), axis=-1)
        mx.eval(probs_mx)
        return np.array(probs_mx, dtype=np.float32)

    def _sample_token(self, probs_np: np.ndarray) -> int:
        """Multinomial sample from a probability array.

        Args:
            probs_np: 1-D float numpy array (will be cast to float64 and re-normalised).

        Returns:
            Sampled token index.
        """
        p = probs_np.astype(np.float64)
        total = p.sum()
        if total <= 0.0 or not np.isfinite(total):
            logger.warning("Degenerate distribution in _sample_token; using argmax fallback.")
            return int(np.argmax(probs_np))
        p /= total
        return int(np.random.choice(len(p), p=p))

    def _predict_next_mlx(self, input_ids: list[int]) -> tuple[int, np.ndarray]:
        """MLX implementation of predict_next.

        Runs a stateless forward pass (no KV cache) over the full input sequence.
        This is correct but not maximally efficient — propose() calls this
        repeatedly, each time extending the context by one token.  A KV-cache
        optimisation is a Phase 3 concern.
        """
        state: _MLXState = self._state  # type: ignore[assignment]

        ids_mx = mx.array(input_ids)[None]  # (1, seq_len)

        # Forward pass — stateless (no cache)
        logits = state.model(ids_mx)  # (1, seq_len, vocab_size)
        mx.eval(logits)

        # Take the logits at the last position
        logits_last = logits[0, -1, :]  # (vocab_size,)

        probs_np = self._logits_to_probs(logits_last)
        next_token = self._sample_token(probs_np)

        return next_token, probs_np

    def _propose_mlx(
        self,
        context: list[int],
        k: int,
    ) -> tuple[list[int], list[np.ndarray]]:
        """MLX implementation of propose.

        Autoregressively generates k tokens, each time appending the sampled
        token to the running context before the next forward pass.
        """
        draft_tokens: list[int] = []
        draft_distributions: list[np.ndarray] = []

        current_context = list(context)

        t0 = time.perf_counter()

        for step in range(k):
            token, probs = self._predict_next_mlx(current_context)
            draft_tokens.append(token)
            draft_distributions.append(probs)
            current_context.append(token)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Draft proposed %d tokens in %.1fms (%.1f ms/token)",
            k,
            elapsed_ms,
            elapsed_ms / k,
        )

        return draft_tokens, draft_distributions
