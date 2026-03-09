# Task Checklist

Tracks project progress. Each task is a self-contained unit of work. Dependencies are noted.
Status: `[x]` = done, `[-]` = in progress, `[ ]` = not started, `[!]` = blocked.

---

## Phase 0: Setup and Validation

- [x] **T0.1** Project directory structure created
- [x] **T0.2** `pyproject.toml` with dependencies (mlx, mlx-lm, coremltools, transformers, torch, numpy)
- [x] **T0.3** Python 3.12 venv with all deps installed and verified (MLX, CoreML, PyTorch all working)
- [x] **T0.4** `src/utils/config.py` — hardware-agnostic config system with YAML support and auto-detection
- [x] **T0.5** **GATE PASSED:** Tokenizer compatibility verified — Qwen3.5-0.8B and 27B share identical 248K tokenizer
- [x] **T0.6** `src/utils/memory.py` — memory monitoring (Metal GPU + process RSS + system pressure)
- [x] **T0.7** `.gitignore`, `config.example.yaml`, test suite (13 tests passing)
- [-] **T0.8** Write `benchmarks/baseline_mlx.py`:
  - Load target model (Qwen3.5-27B-4bit) via `mlx_lm.load`
  - Generate 200 tokens from 3 prompts
  - Measure: tokens/s, time-to-first-token, peak Metal memory
  - This is the baseline all speculative numbers compare against
  - **Note:** Model needs HuggingFace format for MLX. Local GGUF files won't work directly.
    Use `mlx-community/Qwen3.5-27B-4bit` from HF, or convert GGUF → MLX format.
- [ ] **T0.9** GitHub repo setup:
  - Initialize fresh repo (NOT a fork of ANEMLL — see decision log below)
  - Add MIT license, proper README for open source
  - Ensure no secrets committed (.gitignore covers .env, credentials, etc.)

### Decision Log

**ANEMLL fork decision: NO.** ANEMLL is a single-model ANE inference pipeline. Our project
orchestrates two models on two different compute units (ANE + GPU) for speculative decoding.
Forking would add unnecessary baggage (Swift code, iOS targets, chunked execution).
Instead, we reference ANEMLL's Qwen conversion techniques and potentially import as a dependency.

**Model format:** Local models are GGUF (from LM Studio). Core ML needs HuggingFace/PyTorch
format. MLX works best with mlx-community safetensors. We'll download HF models for both engines,
but the config system supports pointing at any model path.

**Python version:** Using 3.12 (not 3.14) because torch and coremltools don't have 3.14 wheels yet.

---

## Phase 1: Draft Model on ANE

**Goal:** Get Qwen3.5-0.8B running on the Neural Engine via Core ML.

- [ ] **T1.1** Write `src/draft/convert.py` — HuggingFace → Core ML conversion:
  - Download Qwen3.5-0.8B from HuggingFace (need PyTorch weights, not GGUF)
  - Trace with `torch.jit.trace` at fixed sequence length (128)
  - Convert with `coremltools.convert(compute_units=CPU_AND_NE, convert_to="mlprogram")`
  - Save as `.mlpackage`
  - CLI: `python src/draft/convert.py --model Qwen/Qwen3.5-0.8B --output models/draft.mlpackage`
  - **Risk:** Conversion may fail on unsupported ops. See T1.2.

- [ ] **T1.2** If T1.1 fails, apply ANE optimizations:
  - Study `apple/ml-ane-transformers` patterns: Linear → Conv2d(1x1), NCHW layout
  - Study ANEMLL's Qwen conversion code for RoPE / GQA handling
  - Write ANE-optimized model wrapper if needed
  - Alternatively, try ANEMLL's full conversion pipeline
  - Document which approach worked and why
  - **Depends on:** T1.1

- [ ] **T1.3** Write `src/draft/engine.py` — `DraftEngine` class:
  - Load Core ML `.mlpackage` model
  - `predict_next(input_ids) → (next_token, probability_distribution)`
  - `propose(context, k) → (draft_tokens, draft_distributions)`
  - Handle padding (input < fixed seq_len) and truncation (input > seq_len)
  - Return full probability distributions (softmax of logits), not just argmax
  - **Depends on:** T1.1 or T1.2 (need a converted model)

- [ ] **T1.4** Write `tests/test_draft_engine.py`:
  - Model loads without error
  - Single prediction produces valid probability distribution (sums to ~1.0)
  - `propose(k=5)` returns exactly 5 tokens with 5 distributions
  - Token IDs are within vocab range (0–248076)
  - Padding works for short inputs
  - **Depends on:** T1.3

- [ ] **T1.5** Write `benchmarks/draft_benchmark.py`:
  - Measure: tokens/s, latency per forward pass (mean/median/p99)
  - Run 100 predictions
  - Verify ANE execution (not CPU/GPU fallback) via coremltools compute plan or Instruments
  - Compare: ANE speed vs same model on GPU via MLX
  - **Depends on:** T1.3
  - **Gate:** Draft must be >100 tok/s on ANE to be beneficial for speculative decoding

---

## Phase 2: Speculative Decoding Pipeline

**Goal:** Full end-to-end speculative decoding with draft on ANE and target on GPU.

- [ ] **T2.1** Write `src/target/engine.py` — `TargetEngine` class:
  - Load model via `mlx_lm.load`
  - `prefill(tokens) → KVCache` — process prompt
  - `verify(draft_tokens, cache) → list[Distribution]` — batched verification (THE critical method)
  - `truncate_cache(cache, length) → KVCache` — roll back after rejection
  - Must return full probability distributions for rejection sampling
  - Can be developed independently of draft engine

- [ ] **T2.2** Write `tests/test_target_engine.py`:
  - Model loads and generates text
  - Prefill produces valid KV cache
  - `verify(K tokens)` returns K+1 distributions
  - Cache truncation is correct (re-forward after truncate matches)
  - **Depends on:** T2.1

- [ ] **T2.3** Write `src/orchestrator/sampling.py`:
  - `rejection_sample(target_prob, draft_prob, draft_token) → (accepted, token)`
  - `compute_residual_distribution(target_dist, draft_dist) → distribution`
  - `sample_from_distribution(probs, temperature, top_p) → token`
  - Handle numerical edge cases: division by zero, negative residuals
  - **No model dependencies — can be developed and tested with synthetic distributions**

- [ ] **T2.4** Write `tests/test_sampling.py`:
  - Identical distributions → ~100% acceptance
  - Very different distributions → low acceptance
  - Residual distribution is valid (non-negative, sums to 1)
  - Statistical test: over 10K samples, output matches target distribution
  - **Depends on:** T2.3

- [ ] **T2.5** Write `src/orchestrator/scheduler.py` — main speculative decoding loop:
  - Implements algorithm from ARCHITECTURE.md
  - Yields `(new_tokens, stats)` per speculation round (streaming)
  - Stats: num_proposed, num_accepted, draft_time_ms, verify_time_ms
  - Handles EOS, bonus token (all K accepted), KV cache rollback
  - **Depends on:** T1.3, T2.1, T2.3

- [ ] **T2.6** Write `src/orchestrator/pipeline.py` / `src/cli.py` — CLI entry point:
  - `--draft-model`, `--target-model`, `--prompt`, `--K`, `--max-tokens`, `--temperature`
  - Streaming output, summary statistics after generation
  - **Depends on:** T2.5

- [ ] **T2.7** Write `tests/test_e2e.py` — end-to-end integration tests:
  - Full pipeline produces coherent text
  - Statistics are reasonable (acceptance rate > 0)
  - No memory leaks over extended generation
  - **Depends on:** T2.6

- [ ] **T2.8** Write `benchmarks/speculative.py` + `benchmarks/compare.py`:
  - Same prompts as baseline benchmark
  - Side-by-side comparison: baseline MLX vs speculative
  - **Depends on:** T0.8, T2.6

---

## Phase 3: Optimization

**Goal:** Maximize speedup through parallelism and tuning.

- [ ] **T3.1** Tune speculation length K:
  - Benchmark K = 3, 4, 5, 6, 8 on standard prompts
  - Find optimal K for Qwen3.5 0.8B/27B pairing
  - May vary by task type

- [ ] **T3.2** Async draft pipeline (ANE + GPU in parallel):
  - While GPU verifies batch N, ANE drafts batch N+1
  - Threading with `queue.Queue` or `asyncio`
  - On rejection: discard speculative draft, re-sync
  - Benchmark parallel vs sequential
  - **Depends on:** T2.5

- [ ] **T3.3** Draft model KV caching on ANE:
  - Modify Core ML model to accept/output KV cache tensors
  - Requires re-conversion with explicit cache I/O
  - Verify: cached output matches uncached
  - Significant speedup for longer contexts
  - **Depends on:** T1.1 or T1.2

- [ ] **T3.4** Memory optimization:
  - Profile memory at each stage
  - Test context lengths: 1K, 2K, 4K, 8K — find the limit
  - Identify unnecessary memory duplication

- [ ] **T3.5** Final benchmarks and `RESULTS.md`:
  - Full comparison: baseline vs sequential speculative vs parallel speculative
  - Metrics: tokens/s, TTFT, peak memory, power (via `powermetrics`)
  - Multiple prompt types: Q&A, long-form, code

---

## Phase 4: Open Source Polish

- [ ] **T4.1** CLI tool (clean argparse/click interface, interactive chat mode)
- [ ] **T4.2** Documentation (usage guide, benchmarks, troubleshooting)
- [ ] **T4.3** Support for other model families (LLaMA, Gemma, etc.)
- [ ] **T4.4** API server mode (OpenAI-compatible, like mlx_lm.server)
- [ ] **T4.5** CI/CD (GitHub Actions for tests — macOS runner)
- [ ] **T4.6** Contributing guide, issue templates, release process

---

## Parallelism Map

Tasks that can be worked on simultaneously:

```
T0.8 (baseline benchmark)  ─┐
T1.1 (draft conversion)     ├── can run in parallel
T2.1 (target engine)         │
T2.3 (sampling math)        ─┘

T1.3 (draft engine)     ── depends on T1.1
T2.5 (scheduler)        ── depends on T1.3 + T2.1 + T2.3
T2.6 (pipeline/CLI)     ── depends on T2.5
```

## Notes for Agents

### Model Formats
- **Draft (ANE):** Needs HuggingFace PyTorch format → Core ML `.mlpackage`. Download `Qwen/Qwen3.5-0.8B`.
- **Target (GPU):** Needs MLX safetensors format. Use `mlx-community/Qwen3.5-27B-4bit` from HF.
- **Local GGUF files** at `/Users/kenfink/.lmstudio/models/unsloth/` are for reference/comparison only.
  Do NOT hard-code these paths. The config system handles model resolution.

### Key Technical Notes
- MLX operations are lazy — call `mx.eval()` to force computation when timing
- Core ML models are directories (`.mlpackage`), not single files
- KV cache in MLX: list of `mlx_lm.models.cache.KVCache` objects, one per layer
- Data transfer between MLX and Core ML goes through NumPy arrays
- ANE prefers: 4D tensors (B,C,1,S), Conv2d instead of Linear, FP16, static shapes

### Debugging
- Draft model slow → probably not on ANE. Check Xcode Instruments (green = ANE)
- Low acceptance rate (<30%) → tokenizer mismatch (already verified: PASS)
- OOM → reduce context length, drop to Q3 quantization, or `mx.metal.clear_cache()`
