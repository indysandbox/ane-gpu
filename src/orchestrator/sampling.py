"""Rejection sampling for speculative decoding.

Implements the algorithm from Leviathan et al., 2023 —
"Fast Inference from Transformers via Speculative Decoding"
(https://arxiv.org/abs/2211.17192).

This module is pure math: no model, no device, no framework imports beyond
NumPy.  All probability distributions are 1-D float64 arrays over the
vocabulary.  The caller is responsible for converting logits from whatever
framework (MLX / Core ML) into NumPy before calling these functions.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Numerical floor for probabilities to avoid log(0) and divide-by-zero.
_EPS: float = 1e-10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rejection_sample(
    target_prob: np.ndarray,
    draft_prob: np.ndarray,
    draft_token: int,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[bool, int]:
    """Speculative decoding rejection sampling (Leviathan et al., 2023).

    Given that the draft model proposed *draft_token* with probability
    q(draft_token), accept it with probability min(1, p(x)/q(x)) where p is
    the target distribution.  On rejection, sample a correction token from the
    residual distribution norm(max(0, p(x) - q(x))).

    Parameters
    ----------
    target_prob:
        Probability distribution from the target model, shape ``(vocab_size,)``.
        Must be non-negative and sum to approximately 1.
    draft_prob:
        Probability distribution from the draft model, shape ``(vocab_size,)``.
        Must be non-negative and sum to approximately 1.
    draft_token:
        Index of the token the draft model chose.
    temperature:
        Passed to :func:`sample_from_distribution` when drawing a correction
        token from the residual.  At temperature=1.0 (default) the residual is
        sampled as-is.
    rng:
        Optional ``np.random.Generator`` for reproducibility.  If ``None`` a
        fresh default generator is created (not reproducible across calls).

    Returns
    -------
    accepted : bool
        ``True`` if the draft token is accepted.
    token : int
        The accepted draft token, or a correction token drawn from the residual
        distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    target_prob = _validate_distribution(target_prob, "target_prob")
    draft_prob = _validate_distribution(draft_prob, "draft_prob")

    p_x = float(target_prob[draft_token])
    q_x = float(draft_prob[draft_token])

    # Acceptance ratio: min(1, p(x) / q(x)).
    # If the draft placed zero probability on this token the ratio is infinite
    # → accept with probability 1 (target assigned positive probability means
    #   the draft was too conservative).
    if q_x <= _EPS:
        acceptance_prob = 1.0
    else:
        acceptance_prob = min(1.0, p_x / q_x)

    u = rng.uniform(0.0, 1.0)
    if u <= acceptance_prob:
        return True, draft_token

    # Rejected: sample correction from residual distribution.
    residual = compute_residual_distribution(target_prob, draft_prob)
    correction_token = sample_from_distribution(residual, temperature=temperature, rng=rng)
    return False, correction_token


def compute_residual_distribution(
    target_dist: np.ndarray,
    draft_dist: np.ndarray,
) -> np.ndarray:
    """Compute norm(max(0, p(x) - q(x))) for correction sampling.

    The residual distribution captures the probability mass that the draft
    model "missed" relative to the target.  After normalisation it is a valid
    probability distribution over the vocabulary.

    Parameters
    ----------
    target_dist:
        Target model probabilities, shape ``(vocab_size,)``.
    draft_dist:
        Draft model probabilities, same shape.

    Returns
    -------
    np.ndarray
        Normalised residual distribution, shape ``(vocab_size,)``.  Guaranteed
        to be non-negative and sum to 1.  If the residual is all zeros (e.g.
        draft == target), falls back to the target distribution.
    """
    target_dist = _validate_distribution(target_dist, "target_dist")
    draft_dist = _validate_distribution(draft_dist, "draft_dist")

    residual = np.maximum(0.0, target_dist - draft_dist)
    total = float(residual.sum())

    if total <= _EPS:
        # Draft matches target exactly — fall back to target for correctness.
        logger.debug(
            "Residual distribution is zero-sum (draft ≈ target); "
            "falling back to target distribution."
        )
        return target_dist.copy()

    return residual / total


def sample_from_distribution(
    probs: np.ndarray,
    temperature: float = 1.0,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample a token from a probability distribution with temperature/top-p.

    Parameters
    ----------
    probs:
        Probability distribution, shape ``(vocab_size,)``.  Non-negative; need
        not sum exactly to 1 (will be re-normalised after filtering).
    temperature:
        Scaling factor applied in log-space.  ``temperature=0`` gives greedy
        (argmax); ``temperature=1`` samples as-is; values >1 flatten the
        distribution.
    top_p:
        Nucleus sampling threshold in (0, 1].  ``top_p=1.0`` disables nucleus
        filtering.
    rng:
        Optional ``np.random.Generator`` for reproducibility.

    Returns
    -------
    int
        Sampled vocabulary index.
    """
    if rng is None:
        rng = np.random.default_rng()

    probs = _validate_distribution(probs, "probs", allow_unnormalised=True)

    # --- Temperature ---
    if temperature <= 0.0:
        # Greedy decoding.
        return int(np.argmax(probs))

    if temperature != 1.0:
        # Convert to logits, scale, convert back.
        logits = np.log(np.clip(probs, _EPS, None))
        logits = logits / temperature
        # Numerical stability: subtract max before softmax.
        logits = logits - logits.max()
        probs = np.exp(logits)

    # Re-normalise after temperature (or if input was unnormalised).
    probs = _safe_normalise(probs)

    # --- Top-p (nucleus) filtering ---
    if top_p < 1.0:
        probs = _apply_top_p(probs, top_p)

    return int(rng.choice(len(probs), p=probs))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits and return a probability distribution.

    Parameters
    ----------
    logits:
        Raw (pre-softmax) logits, shape ``(vocab_size,)``.  May be any real
        numbers; need not be normalised.
    temperature:
        Scaling divisor.  ``temperature=0`` returns a one-hot at argmax;
        ``temperature=1`` is standard softmax.

    Returns
    -------
    np.ndarray
        Probability distribution, shape ``(vocab_size,)``.  Non-negative, sums
        to 1.
    """
    logits = np.asarray(logits, dtype=np.float64)

    # Replace NaN/Inf before any arithmetic to prevent NaN propagation.
    if np.any(~np.isfinite(logits)):
        logger.warning("apply_temperature: logits contain NaN or Inf; replacing with zeros.")
        logits = np.where(np.isfinite(logits), logits, 0.0)

    if temperature <= 0.0:
        # One-hot at argmax.
        result = np.zeros_like(logits)
        result[int(np.argmax(logits))] = 1.0
        return result

    scaled = logits / temperature
    # Subtract max for numerical stability before exp.
    scaled = scaled - scaled.max()
    probs = np.exp(scaled)
    return _safe_normalise(probs)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_distribution(
    arr: np.ndarray,
    name: str,
    *,
    allow_unnormalised: bool = False,
) -> np.ndarray:
    """Return a float64 copy of *arr*, clamping negatives and warning on issues.

    Parameters
    ----------
    arr:
        Input array to validate.
    name:
        Human-readable name used in log messages.
    allow_unnormalised:
        If ``False`` (default), log a warning when the sum deviates from 1 by
        more than 1e-4.

    Returns
    -------
    np.ndarray
        Validated, non-negative float64 array.
    """
    arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if len(arr) == 0:
        raise ValueError(f"{name} must be non-empty")

    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        logger.warning("%s contains NaN or Inf; replacing with zeros.", name)
        arr = np.where(np.isfinite(arr), arr, 0.0)

    if np.any(arr < 0.0):
        logger.warning("%s contains negative values; clamping to zero.", name)
        arr = np.maximum(0.0, arr)

    if not allow_unnormalised:
        total = float(arr.sum())
        if abs(total - 1.0) > 1e-4:
            logger.warning(
                "%s sums to %.6f instead of 1.0; normalising.", name, total
            )
            arr = _safe_normalise(arr)

    return arr


def _safe_normalise(probs: np.ndarray) -> np.ndarray:
    """Normalise *probs* to sum to 1; fall back to uniform on zero-sum input."""
    total = float(probs.sum())
    if total <= _EPS:
        logger.warning(
            "Probability array sums to zero; falling back to uniform distribution."
        )
        return np.full(len(probs), 1.0 / len(probs), dtype=np.float64)
    return probs / total


def _apply_top_p(probs: np.ndarray, top_p: float) -> np.ndarray:
    """Nucleus (top-p) filtering: zero out tokens outside the top-p mass.

    Parameters
    ----------
    probs:
        Normalised probability distribution.
    top_p:
        Cumulative probability threshold in (0, 1].

    Returns
    -------
    np.ndarray
        Filtered and re-normalised probability distribution.
    """
    sorted_indices = np.argsort(probs)[::-1]  # descending
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)

    # Keep tokens whose cumulative mass ≤ top_p; always keep at least one.
    # The cutoff is the first index where cumulative mass exceeds top_p.
    cutoff_mask = np.zeros(len(probs), dtype=bool)
    cutoff_mask[sorted_indices] = (cumulative - sorted_probs) < top_p

    filtered = np.where(cutoff_mask, probs, 0.0)
    return _safe_normalise(filtered)
