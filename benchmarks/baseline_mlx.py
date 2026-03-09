"""Baseline MLX generation benchmark.

Measures target model throughput (tokens/s), time-to-first-token, and memory
usage with no speculative decoding. This is the performance bar we need to beat.

Usage:
    python benchmarks/baseline_mlx.py
    python benchmarks/baseline_mlx.py --model mlx-community/Qwen3.5-14B-4bit
    python benchmarks/baseline_mlx.py --max-tokens 100 --prompt "Hello, world"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Ensure project root is on the path so src.utils resolves correctly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.memory import MemorySnapshot, get_memory_snapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default prompts — chosen to be diverse in domain and style so they stress
# different parts of the model's vocabulary distribution.
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS: list[str] = [
    (
        "Explain the difference between supervised and unsupervised learning "
        "in machine learning, and give two concrete examples of each."
    ),
    (
        "Write a Python function that implements binary search on a sorted list "
        "and returns the index of the target value, or -1 if not found."
    ),
    (
        "Describe the causes and consequences of the French Revolution, "
        "focusing on its long-term impact on European political thought."
    ),
]

DEFAULT_MODEL = "mlx-community/Qwen3.5-27B-4bit"
DEFAULT_MAX_TOKENS = 200


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PromptResult:
    """Timing and memory results for a single prompt."""

    prompt: str
    prompt_tokens: int = 0
    generated_tokens: int = 0

    # Timing (seconds)
    prefill_time_s: float = 0.0   # time-to-first-token
    generation_time_s: float = 0.0  # time for tokens 2..N (excludes first token)

    # Memory snapshots
    mem_after_prefill: Optional[MemorySnapshot] = None
    mem_after_generation: Optional[MemorySnapshot] = None

    @property
    def tokens_per_second(self) -> float:
        """Throughput of the *generation* phase only (excludes prefill)."""
        # We generated (generated_tokens - 1) tokens after the first; if only
        # one token was produced the generation phase had zero duration.
        decode_tokens = max(self.generated_tokens - 1, 0)
        if self.generation_time_s <= 0 or decode_tokens == 0:
            return 0.0
        return decode_tokens / self.generation_time_s

    @property
    def time_to_first_token_ms(self) -> float:
        return self.prefill_time_s * 1000.0

    @property
    def peak_metal_mb(self) -> float:
        snap = self.mem_after_generation or self.mem_after_prefill
        return snap.metal_peak_mb if snap else 0.0

    @property
    def active_metal_mb(self) -> float:
        snap = self.mem_after_generation or self.mem_after_prefill
        return snap.metal_active_mb if snap else 0.0


@dataclass
class BenchmarkResults:
    """Aggregate results across all prompts."""

    model_name: str
    max_tokens: int
    prompt_results: list[PromptResult] = field(default_factory=list)

    @property
    def mean_tokens_per_second(self) -> float:
        vals = [r.tokens_per_second for r in self.prompt_results if r.tokens_per_second > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_ttft_ms(self) -> float:
        vals = [r.time_to_first_token_ms for r in self.prompt_results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def peak_metal_mb(self) -> float:
        vals = [r.peak_metal_mb for r in self.prompt_results]
        return max(vals) if vals else 0.0

    @property
    def active_metal_mb(self) -> float:
        """Active memory at end of last prompt (most representative figure)."""
        if not self.prompt_results:
            return 0.0
        return self.prompt_results[-1].active_metal_mb


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------


def _encode_prompt(tokenizer, prompt: str) -> list[int]:
    """Tokenize a prompt and return token IDs as a plain list."""
    encoded = tokenizer.encode(prompt)
    # mlx_lm tokenizers may return a list or a dict; normalise to list[int].
    if isinstance(encoded, dict):
        return list(encoded["input_ids"])
    return list(encoded)


def run_single_prompt(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
) -> PromptResult:
    """Run generation for one prompt and return detailed timing results.

    Timing breakdown:
    - prefill_time_s  : from just before model forward pass with the full
                        prompt until the *first* output token is evaluated.
    - generation_time_s : time to produce tokens 2 through N (decode phase).

    We use ``mlx_lm.utils.generate_step`` which yields (token, logprobs) pairs
    one at a time, giving us per-step control over timing.  Each yielded token
    is *lazy* until we call ``mx.eval()``.
    """
    import mlx.core as mx
    from mlx_lm.utils import generate_step

    result = PromptResult(prompt=prompt)

    # Tokenise
    prompt_tokens = _encode_prompt(tokenizer, prompt)
    result.prompt_tokens = len(prompt_tokens)
    logger.debug("Prompt has %d tokens.", result.prompt_tokens)

    # Convert to MLX array and force evaluation so that the prompt encoding
    # cost is not included in the prefill timer.
    prompt_array = mx.array(prompt_tokens)
    mx.eval(prompt_array)

    # Reset peak memory counter so we get a clean reading for this prompt.
    mx.reset_peak_memory()

    # -----------------------------------------------------------------------
    # Prefill: generate the first token and measure time-to-first-token.
    # -----------------------------------------------------------------------
    token_iterator = generate_step(
        prompt=prompt_array,
        model=model,
        temp=0.0,     # greedy — deterministic, easy to compare runs
        top_p=1.0,
        repetition_penalty=1.0,
        repetition_context_size=20,
    )

    t_prefill_start = time.perf_counter()
    first_token, _ = next(token_iterator)
    mx.eval(first_token)  # force evaluation before stopping the clock
    t_prefill_end = time.perf_counter()

    result.prefill_time_s = t_prefill_end - t_prefill_start
    result.mem_after_prefill = get_memory_snapshot()
    logger.debug(
        "Prefill complete: %.1f ms | %s",
        result.prefill_time_s * 1000,
        result.mem_after_prefill,
    )

    generated: list[int] = [int(first_token.item())]

    # -----------------------------------------------------------------------
    # Decode: generate tokens 2..max_tokens and measure throughput.
    # -----------------------------------------------------------------------
    eos_token_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)

    t_decode_start = time.perf_counter()

    for _step in range(max_tokens - 1):  # -1 because we already have token 1
        try:
            token, _ = next(token_iterator)
        except StopIteration:
            logger.debug("Generator exhausted after %d tokens.", len(generated))
            break

        mx.eval(token)  # materialise before appending / checking EOS
        token_id = int(token.item())
        generated.append(token_id)

        if eos_token_id is not None and token_id == eos_token_id:
            logger.debug("EOS token encountered at step %d.", len(generated))
            break

    t_decode_end = time.perf_counter()

    result.generation_time_s = t_decode_end - t_decode_start
    result.generated_tokens = len(generated)
    result.mem_after_generation = get_memory_snapshot()

    logger.info(
        "Prompt %d tokens → %d generated | TTFT %.1f ms | %.1f tok/s | %s",
        result.prompt_tokens,
        result.generated_tokens,
        result.time_to_first_token_ms,
        result.tokens_per_second,
        result.mem_after_generation,
    )

    return result


def run_benchmark(
    model_name: str,
    prompts: list[str],
    max_tokens: int,
) -> BenchmarkResults:
    """Load the model and run the full benchmark suite."""
    import mlx.core as mx
    from mlx_lm import load

    results = BenchmarkResults(model_name=model_name, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Load model — this is intentionally outside the timed region.
    # ------------------------------------------------------------------
    print(f"Loading model: {model_name}")
    logger.info("Loading model: %s", model_name)

    t_load_start = time.perf_counter()
    model, tokenizer = load(model_name)
    # Force all model weights into memory before timing starts.
    mx.eval(model.parameters())
    t_load_end = time.perf_counter()

    mem_after_load = get_memory_snapshot()
    print(
        f"Model loaded in {t_load_end - t_load_start:.1f}s | "
        f"Metal active: {mem_after_load.metal_active_mb:.0f} MB | "
        f"Peak: {mem_after_load.metal_peak_mb:.0f} MB"
    )
    logger.info("Model load complete: %s", mem_after_load)

    # ------------------------------------------------------------------
    # Run each prompt
    # ------------------------------------------------------------------
    for idx, prompt in enumerate(prompts, start=1):
        print(f"\n--- Prompt {idx}/{len(prompts)} ---")
        short = prompt[:80].replace("\n", " ")
        print(f"  {short}{'...' if len(prompt) > 80 else ''}")

        prompt_result = run_single_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        results.prompt_results.append(prompt_result)

        print(
            f"  TTFT:       {prompt_result.time_to_first_token_ms:>8.1f} ms"
        )
        print(
            f"  Throughput: {prompt_result.tokens_per_second:>8.1f} tok/s  "
            f"({prompt_result.generated_tokens} tokens in "
            f"{prompt_result.generation_time_s:.2f}s)"
        )
        print(
            f"  Metal mem:  {prompt_result.active_metal_mb:>8.0f} MB active  "
            f"/ {prompt_result.peak_metal_mb:.0f} MB peak"
        )

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(results: BenchmarkResults) -> None:
    """Print a formatted summary table of benchmark results."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  BASELINE MLX BENCHMARK — SUMMARY")
    print(sep)
    print(f"  Model      : {results.model_name}")
    print(f"  Max tokens : {results.max_tokens}")
    print(f"  Prompts    : {len(results.prompt_results)}")
    print(sep)

    # Per-prompt rows
    header = f"  {'#':>2}  {'TTFT (ms)':>10}  {'Tok/s':>8}  {'Gen tok':>7}  {'Peak MB':>8}"
    print(header)
    print(f"  {'-'*2}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*8}")

    for idx, r in enumerate(results.prompt_results, start=1):
        print(
            f"  {idx:>2}  "
            f"{r.time_to_first_token_ms:>10.1f}  "
            f"{r.tokens_per_second:>8.1f}  "
            f"{r.generated_tokens:>7d}  "
            f"{r.peak_metal_mb:>8.0f}"
        )

    print(sep)
    print(
        f"  {'AVG':>2}  "
        f"{results.mean_ttft_ms:>10.1f}  "
        f"{results.mean_tokens_per_second:>8.1f}  "
        f"{'':>7}  "
        f"{results.peak_metal_mb:>8.0f}"
    )
    print(sep)
    print(
        f"\n  Active Metal memory (end of run): "
        f"{results.active_metal_mb:.0f} MB "
        f"({results.active_metal_mb / 1024:.2f} GB)"
    )
    print(
        f"  Peak Metal memory (run-wide):     "
        f"{results.peak_metal_mb:.0f} MB "
        f"({results.peak_metal_mb / 1024:.2f} GB)"
    )
    print(
        f"\n  *** Baseline to beat: {results.mean_tokens_per_second:.1f} tok/s ***\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure baseline MLX generation throughput (no speculative decoding). "
            "Results establish the performance target for ANE-GPU speculative decoding."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model ID or local path to load via mlx_lm.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        dest="max_tokens",
        help="Maximum number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        metavar="PROMPT",
        help=(
            "Prompt to benchmark (may be specified multiple times). "
            "Overrides the three built-in default prompts."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="Python logging level (debug info goes to stderr; progress goes to stdout).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        stream=sys.stderr,
    )

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    print(f"\nANE-GPU Baseline Benchmark")
    print(f"  model      : {args.model}")
    print(f"  max_tokens : {args.max_tokens}")
    print(f"  prompts    : {len(prompts)}")

    results = run_benchmark(
        model_name=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
