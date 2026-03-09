# Task Checklist

Concrete tasks for Claude Code to execute, in order. Each task should be completable independently. Check off as you go.

---

## Phase 0: Setup and Validation

- [ ] **T0.1** Create project directory structure as specified in README.md
- [ ] **T0.2** Create `pyproject.toml` with project metadata and dependencies:
  - `mlx>=0.22`
  - `mlx-lm>=0.21`
  - `coremltools>=8.0`
  - `transformers>=4.46`
  - `torch>=2.4`
  - `numpy`
  - `huggingface-hub`
- [ ] **T0.3** Create `scripts/setup.sh`:
  - Create venv
  - Install dependencies
  - Verify MLX GPU access
  - Print system info (chip, memory, macOS version)
- [ ] **T0.4** Write `src/utils/config.py`:
  - Dataclass for all configuration (model paths, K, max_tokens, context_length, temperature, etc.)
  - Load from YAML file or CLI args
  - Sensible defaults for M4 Air 24GB
- [ ] **T0.5** Write tokenizer compatibility checker (`scripts/check_tokenizers.py`):
  - Accept two model names as args
  - Load both tokenizers from HF
  - Compare vocab size, vocab contents, special tokens
  - Test encode/decode on 20+ diverse strings
  - Print clear PASS/FAIL report
  - Run it with: `Qwen/Qwen3.5-27B` vs `Qwen/Qwen3.5-0.8B` (and other candidates)
- [ ] **T0.6** Write `benchmarks/baseline_mlx.py`:
  - Load target model via `mlx_lm.load`
  - Generate 200 tokens from 3 different prompts
  - Measure and report: tokens/s (generation), time-to-first-token, peak memory
  - Use `time.perf_counter()` for timing
  - Use `mlx.core.metal.get_active_memory()` and `get_peak_memory()` for memory
- [ ] **T0.7** Write `src/utils/memory.py`:
  - Function to get current memory usage (Python process + MLX metal memory)
  - Function to get system memory pressure (`memory_pressure` command or `psutil`)
  - Warning system: alert when memory usage exceeds threshold (e.g., 22GB)

**Decision gate after T0.5:** Determine which draft model to use based on tokenizer compatibility.

---

## Phase 1: Draft Model on ANE

- [ ] **T1.1** Write `src/draft/convert.py`:
  - Download draft model from HuggingFace
  - Trace with `torch.jit.trace` at fixed sequence length (default 128)
  - Convert with `coremltools`:
    ```python
    ct.convert(
        traced_model,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    ```
  - Save to specified output path
  - Handle errors gracefully with clear messages
  - CLI: `python src/draft/convert.py --model MODEL_NAME --output PATH --seq-len 128`

- [ ] **T1.2** If conversion fails with unsupported ops, investigate:
  - Check which ops are falling back to CPU/GPU
  - Study `apple/ml-ane-transformers` for patterns:
    - Replace `nn.Linear` → `nn.Conv2d(C_in, C_out, 1)`
    - Reshape to (B, C, 1, S) layout
  - Write an ANE-optimized model wrapper if needed
  - Alternatively, investigate using ANEMLL's conversion pipeline
  - Document which approach worked

- [ ] **T1.3** Write `src/draft/engine.py` — `DraftEngine` class:
  ```python
  class DraftEngine:
      def __init__(self, model_path: str, seq_len: int = 128):
          """Load Core ML model."""
      
      def predict_next(self, input_ids: list[int]) -> tuple[int, np.ndarray]:
          """Given token sequence, return (next_token, probability_distribution)."""
      
      def propose(self, context: list[int], k: int) -> tuple[list[int], list[np.ndarray]]:
          """Autoregressively generate k draft tokens with their distributions."""
      
      def reset(self):
          """Reset any internal state."""
  ```
  - Handle padding (input shorter than fixed seq_len)
  - Handle truncation (input longer than seq_len — use last seq_len tokens)
  - Return raw logits/probabilities, not just argmax tokens

- [ ] **T1.4** Write `tests/test_draft_engine.py`:
  - Test: model loads without error
  - Test: single token prediction produces valid distribution (sums to ~1.0)
  - Test: propose(k=5) returns exactly 5 tokens with 5 distributions
  - Test: output tokens are valid token IDs within vocab range
  - Test: padding works correctly for short inputs

- [ ] **T1.5** Write `benchmarks/draft_benchmark.py`:
  - Measure draft model: tokens/s, latency per forward pass
  - Run 100 predictions, report mean/median/p99
  - Monitor which compute unit is being used (ANE vs CPU vs GPU)
  - Use `coremltools` compute plan analysis or Xcode Instruments
  - Compare: draft on ANE vs draft on GPU (via MLX) to quantify ANE benefit

---

## Phase 2: Speculative Decoding Pipeline

- [ ] **T2.1** Write `src/target/engine.py` — `TargetEngine` class:
  ```python
  class TargetEngine:
      def __init__(self, model_name: str):
          """Load MLX model."""
      
      def prefill(self, tokens: list[int]) -> KVCache:
          """Process prompt tokens, return KV cache."""
      
      def verify(self, draft_tokens: list[int], cache: KVCache) -> list[np.ndarray]:
          """Batched forward pass on draft tokens. Return per-position logits."""
      
      def decode_one(self, cache: KVCache) -> tuple[int, np.ndarray]:
          """Standard single-token autoregressive decode (for bonus token)."""
      
      def truncate_cache(self, cache: KVCache, length: int) -> KVCache:
          """Roll back KV cache to given sequence length."""
  ```
  - The `verify` method is the critical one — it must process K tokens in a single forward pass
  - Must return full probability distributions (softmax of logits), not just top tokens
  - KV cache truncation must be exact and correct

- [ ] **T2.2** Write `tests/test_target_engine.py`:
  - Test: model loads and generates text
  - Test: prefill produces valid KV cache
  - Test: verify(K tokens) returns K+1 distributions
  - Test: truncate_cache works (verify output is same after truncate+re-forward)
  - Test: decode_one produces valid token

- [ ] **T2.3** Write `src/orchestrator/sampling.py`:
  ```python
  def rejection_sample(
      target_prob: np.ndarray,  # p(x) from target model at this position
      draft_prob: np.ndarray,   # q(x) from draft model at this position
      draft_token: int,         # The token the draft model proposed
      temperature: float = 1.0,
  ) -> tuple[bool, int]:
      """
      Returns (accepted, token).
      If accepted=True, token=draft_token.
      If accepted=False, token is sampled from residual distribution.
      """
  
  def sample_from_distribution(
      probs: np.ndarray,
      temperature: float = 1.0,
      top_p: float = 1.0,
  ) -> int:
      """Sample a token from a probability distribution with temperature/top-p."""
  
  def compute_residual_distribution(
      target_dist: np.ndarray,
      draft_dist: np.ndarray,
  ) -> np.ndarray:
      """Compute norm(max(0, p(x) - q(x))) for correction sampling."""
  ```
  - Handle numerical edge cases: division by zero, negative probabilities after subtraction
  - Temperature scaling before rejection test
  - Proper normalization of residual distribution

- [ ] **T2.4** Write `tests/test_sampling.py`:
  - Test: if target_prob == draft_prob, acceptance rate should be ~100%
  - Test: if distributions are very different, acceptance rate should be low
  - Test: residual distribution is valid (non-negative, sums to 1)
  - Test: over many samples, output distribution matches target distribution
  - Statistical test with >10,000 samples

- [ ] **T2.5** Write `src/orchestrator/scheduler.py` — main speculative loop:
  ```python
  class SpeculativeScheduler:
      def __init__(self, draft: DraftEngine, target: TargetEngine, k: int = 5):
          ...
      
      def generate(
          self,
          prompt_tokens: list[int],
          max_tokens: int = 200,
          temperature: float = 1.0,
          top_p: float = 1.0,
      ) -> Generator[tuple[list[int], dict], None, None]:
          """
          Yields (new_tokens, stats) at each speculation round.
          stats includes: num_proposed, num_accepted, draft_time, verify_time
          """
  ```
  - Implement the full algorithm from ARCHITECTURE.md
  - Track statistics: acceptance rate, tokens per round, timing per phase
  - Handle EOS token properly (stop if draft or target produces EOS)
  - Handle the "bonus token" case (all K accepted)

- [ ] **T2.6** Write `src/orchestrator/pipeline.py` — CLI entry point:
  - Parse args: `--draft-model`, `--target-model`, `--prompt`, `--K`, `--max-tokens`, `--temperature`
  - Load both engines
  - Run generation with streaming output
  - Print summary statistics at the end:
    - Total tokens generated
    - Wall-clock time
    - Effective tokens/s
    - Average acceptance rate
    - Average tokens per speculation round
    - Peak memory usage

- [ ] **T2.7** Write `tests/test_e2e.py`:
  - End-to-end test with small models (can use tiny test models)
  - Verify generated text is coherent
  - Verify statistics are reasonable
  - Verify no memory leaks over extended generation

- [ ] **T2.8** Write `benchmarks/speculative.py`:
  - Run same prompts as `baseline_mlx.py`
  - Report all metrics
  - Write `benchmarks/compare.py` to produce side-by-side comparison table

---

## Phase 3: Optimization

- [ ] **T3.1** Tune speculation length K:
  - Benchmark K = 3, 4, 5, 6, 8 on a standard prompt set
  - Find the K that maximizes effective tokens/s
  - May vary by task type (code vs prose vs Q&A)

- [ ] **T3.2** Implement async draft pipeline:
  - Draft engine runs on background thread
  - While GPU verifies batch N, ANE drafts batch N+1 (assuming full acceptance)
  - On rejection: discard speculative draft, re-sync
  - Use `threading.Thread` + `queue.Queue` or `asyncio`
  - Benchmark: parallel vs sequential speedup

- [ ] **T3.3** Implement draft model KV caching:
  - Modify Core ML model to accept/output KV cache tensors
  - This requires re-doing the conversion with explicit cache I/O
  - Test: verify cached vs uncached produce identical output
  - Benchmark: speedup from caching (should be significant for longer contexts)

- [ ] **T3.4** Memory optimization:
  - Profile memory at each stage with `src/utils/memory.py`
  - Identify any unnecessary memory duplication
  - Test with context lengths: 1K, 2K, 4K, 8K — find the limit

- [ ] **T3.5** Final benchmarks:
  - Full comparison table: baseline MLX vs sequential speculative vs parallel speculative
  - Metrics: tokens/s, time-to-first-token, peak memory, power (if measurable via `powermetrics`)
  - Multiple prompt types: short Q&A, long-form writing, code generation
  - Write up results in `RESULTS.md`

---

## Notes for Claude Code

### Working with Core ML on macOS
- `coremltools` requires macOS. If running in a Linux container, the conversion step must be done on the Mac directly.
- Core ML models are directories (`.mlpackage`), not single files.
- To check if a model runs on ANE, use `coremltools.models.MLModel.predict()` and monitor with Activity Monitor → GPU History and Xcode Instruments.

### Working with MLX
- MLX operations are lazy — call `mx.eval()` to force computation when timing.
- `mlx-lm` models download to `~/.cache/huggingface/`.
- For the target model, `mlx_lm.load()` returns `(model, tokenizer)`.
- KV cache in MLX is typically a list of `mlx_lm.models.cache.KVCache` objects, one per layer.

### Memory Monitoring
- `mlx.core.metal.get_peak_memory()` — peak Metal (GPU) memory
- `mlx.core.metal.get_active_memory()` — current Metal memory
- `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` — Python process RSS
- `subprocess.run(["memory_pressure"])` — system-wide memory pressure

### Debugging Tips
- If the draft model is slow, it's probably not running on ANE. Check with Instruments.
- If acceptance rate is very low (<30%), tokenizer mismatch is the most likely cause.
- If you get OOM, the first thing to try is reducing context length or target quantization.
- `mlx.core.metal.clear_cache()` can help free up unused Metal memory.
