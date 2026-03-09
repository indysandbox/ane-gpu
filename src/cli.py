"""Command-line entry point for ANE-GPU speculative decoding.

Usage::

    # Basic prompt
    python -m src.cli --prompt "Explain quantum entanglement in simple terms."

    # Full options
    python -m src.cli \\
        --prompt "Write a haiku about silicon." \\
        --draft-model Qwen/Qwen3.5-0.8B \\
        --target-model mlx-community/Qwen3.5-27B-4bit \\
        --K 5 \\
        --max-tokens 300 \\
        --temperature 0.8 \\
        --seed 42 \\
        --draft-backend mlx
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

# ---------------------------------------------------------------------------
# Logging — configure before any other imports so handlers propagate
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional heavy imports — handled gracefully
# ---------------------------------------------------------------------------

try:
    from src.orchestrator.scheduler import GenerationSummary, SpeculativeScheduler
    from src.utils.memory import get_memory_snapshot
except ImportError as _e:
    logger.error("Failed to import core modules: %s", _e)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Engine loaders
# ---------------------------------------------------------------------------


def _load_draft_engine(model_id: str, backend: str):
    """Load the draft engine for the chosen backend.

    Args:
        model_id: HuggingFace model ID or local path.
        backend:  "mlx" or "coreml".

    Returns:
        A draft engine object with a ``propose(context_tokens, k)`` method.

    Raises:
        SystemExit: If the requested backend cannot be imported.
    """
    if backend == "coreml":
        try:
            from src.draft.engine import DraftEngine

            logger.info("Loading Core ML draft engine: %s", model_id)
            return DraftEngine.from_coreml(model_id)
        except (ImportError, NotImplementedError) as exc:
            logger.warning(
                "Core ML draft engine not available (%s). "
                "Falling back to MLX draft engine.",
                exc,
            )
            backend = "mlx"

    # MLX backend (default / fallback)
    try:
        from src.draft.engine import DraftEngine

        logger.info("Loading MLX draft engine: %s", model_id)
        return DraftEngine.from_mlx(model_id)
    except ImportError:
        logger.warning(
            "MLX draft engine (src.draft.engine) not available. "
            "Using stub draft engine — output will be random."
        )
        return _StubDraftEngine(model_id)


def _load_target_engine(model_id: str):
    """Load the target engine.

    Args:
        model_id: HuggingFace / MLX community model ID.

    Returns:
        A :class:`src.target.engine.TargetEngine` instance.

    Raises:
        SystemExit: If MLX or the target engine module is unavailable.
    """
    try:
        from src.target.engine import TargetEngine

        logger.info("Loading target engine: %s", model_id)
        return TargetEngine(model_id)
    except ImportError as exc:
        logger.error(
            "Cannot import TargetEngine (is mlx-lm installed?): %s", exc
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Stub draft engine — used when real engines are not yet available
# ---------------------------------------------------------------------------


class _StubDraftEngine:
    """Minimal stub that proposes random tokens.

    This keeps the CLI runnable end-to-end even before the real draft engine
    is implemented.  Acceptance rate will be very low, but the logic can still
    be exercised.
    """

    def __init__(self, model_id: str) -> None:
        import numpy as np

        self._rng = np.random.default_rng()
        self._vocab_size = 151936  # Qwen3.5 tokenizer size
        logger.warning(
            "Using StubDraftEngine — '%s' was not loaded. "
            "Tokens will be random; acceptance rate will be near zero.",
            model_id,
        )

    def propose(self, context_tokens: list[int], k: int):  # noqa: ARG002
        import numpy as np

        draft_tokens = self._rng.integers(0, self._vocab_size, size=k).tolist()
        # Uniform distributions — worst-case for acceptance rate
        distributions = [
            np.full(self._vocab_size, 1.0 / self._vocab_size)
            for _ in range(k)
        ]
        return draft_tokens, distributions

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------


def _get_tokenizer(draft_model_id: str):
    """Return a tokenizer compatible with both draft and target models.

    We load from the draft model ID since, for Qwen3.5, draft and target
    share the same 262K tokenizer.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore[import]

        logger.info("Loading tokenizer from: %s", draft_model_id)
        return AutoTokenizer.from_pretrained(draft_model_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load tokenizer: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="ANE-GPU speculative decoding — run a large target model "
                    "accelerated by a small draft model on Apple Silicon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    parser.add_argument(
        "--draft-model",
        default="Qwen/Qwen3.5-0.8B",
        metavar="MODEL_ID",
        help="HuggingFace model ID (or local path) for the draft model.",
    )
    parser.add_argument(
        "--target-model",
        default="mlx-community/Qwen3.5-27B-4bit",
        metavar="MODEL_ID",
        help="HuggingFace / MLX community model ID for the target model.",
    )
    parser.add_argument(
        "--draft-backend",
        choices=("mlx", "coreml"),
        default="mlx",
        help="Which backend to use for the draft model.",
    )

    # Generation parameters
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt to complete.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5,
        dest="k",
        metavar="K",
        help="Speculation window: number of draft tokens per round.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (1.0 = no scaling).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold (1.0 disables it).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Integer random seed for reproducibility.",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Apply requested log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Validate K
    if args.k < 1:
        parser.error(f"--K must be >= 1, got {args.k}")
    if args.temperature < 0.0:
        parser.error(f"--temperature must be >= 0, got {args.temperature}")
    if not (0.0 < args.top_p <= 1.0):
        parser.error(f"--top-p must be in (0, 1], got {args.top_p}")

    # ------------------------------------------------------------------
    # Memory snapshot before loading
    # ------------------------------------------------------------------
    mem_before = get_memory_snapshot()
    logger.info("Memory before model loading: %s", mem_before)

    # ------------------------------------------------------------------
    # Load tokenizer and engines
    # ------------------------------------------------------------------
    tokenizer = _get_tokenizer(args.draft_model)

    logger.info("Loading draft engine (%s backend)...", args.draft_backend)
    draft_engine = _load_draft_engine(args.draft_model, args.draft_backend)

    logger.info("Loading target engine...")
    target_engine = _load_target_engine(args.target_model)

    mem_after_load = get_memory_snapshot()
    logger.info("Memory after model loading: %s", mem_after_load)

    # ------------------------------------------------------------------
    # Encode prompt
    # ------------------------------------------------------------------
    prompt_text: str = args.prompt
    logger.info("Encoding prompt (%d characters)...", len(prompt_text))

    # transformers tokenizers return a dict; extract input_ids
    encoded = tokenizer(prompt_text, return_tensors=None)
    if isinstance(encoded, dict):
        prompt_tokens: list[int] = encoded["input_ids"]
    else:
        prompt_tokens = list(encoded)

    logger.info("Prompt encoded to %d tokens.", len(prompt_tokens))

    # EOS token ID — used to detect when generation should stop
    eos_token_id: int | None = getattr(tokenizer, "eos_token_id", None)
    logger.info("EOS token ID: %s", eos_token_id)

    # ------------------------------------------------------------------
    # Build scheduler
    # ------------------------------------------------------------------
    scheduler = SpeculativeScheduler(
        draft_engine=draft_engine,
        target_engine=target_engine,
        k=args.k,
        eos_token_id=eos_token_id,
    )

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------
    print()  # Blank line before output
    print(prompt_text, end="", flush=True)

    summary = GenerationSummary()
    t_gen_start = time.perf_counter()

    for new_tokens, stats in scheduler.generate(
        prompt_tokens=prompt_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    ):
        # Decode and stream each batch of accepted tokens
        text_chunk = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(text_chunk, end="", flush=True)
        summary.record(stats)

    summary.wall_time_s = time.perf_counter() - t_gen_start
    print()  # Newline after generated text
    print()  # Blank line before summary

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    mem_final = get_memory_snapshot()

    print("=" * 60)
    print("Generation complete")
    print("=" * 60)
    print(f"  Total tokens generated : {summary.total_tokens}")
    print(f"  Wall-clock time        : {summary.wall_time_s:.2f}s")
    print(f"  Effective throughput   : {summary.tokens_per_second:.1f} tok/s")
    print(f"  Speculation rounds     : {summary.total_rounds}")
    print(f"  Avg acceptance rate    : {summary.average_acceptance_rate:.1%}")
    print(f"  Avg tokens / round     : {summary.average_tokens_per_round:.2f}")
    print(f"  Bonus token rate       : {summary.bonus_token_rate:.1%}")
    print(f"  Draft time (total)     : {summary.total_draft_ms:.0f}ms")
    print(f"  Verify time (total)    : {summary.total_verify_ms:.0f}ms")
    print(f"  Peak Metal memory      : {mem_final.metal_peak_mb:.0f}MB")
    print(f"  Process RSS            : {mem_final.process_rss_mb:.0f}MB")
    if mem_final.system_pressure:
        print(f"  System pressure        : {mem_final.system_pressure}")
    print("=" * 60)


if __name__ == "__main__":
    main()
