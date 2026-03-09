# Implementation Plan

## Guiding Principles

1. **Get something working end-to-end first, then optimize.** A slow working pipeline beats a fast broken one.
2. **Validate assumptions early.** Tokenizer compatibility and ANE execution must be confirmed before writing the orchestrator.
3. **Measure everything.** Every phase should produce benchmark numbers to compare against baseline.
4. **Memory is the constraint.** On 24GB, every MB matters. Monitor continuously.

---

## Phase 0: Environment and Validation (Day 1)

**Goal:** Confirm all dependencies work, tokenizers are compatible, and both engines can produce output independently.

### 0.1 Environment Setup
- Create Python 3.11 virtual environment
- Install dependencies: `mlx`, `mlx-lm`, `coremltools`, `transformers`, `torch`, `numpy`
- Verify MLX can use the GPU: `import mlx.core as mx; mx.eval(mx.ones(3))`
- Verify coremltools version supports macOS 15 / M4

### 0.2 Tokenizer Compatibility Check
- Load tokenizers for candidate draft models and Qwen3.5-27B
- Compare vocabularies: same size? same token→id mapping?
- Test encode/decode round-trip with diverse text samples
- **Decision gate:** Which draft model to use? Prefer smallest model with identical tokenizer.

### 0.3 Baseline: MLX Target Model
- Download `mlx-community/Qwen3.5-27B-4bit` (or Q3 if memory is too tight)
- Run basic generation via `mlx_lm.generate`
- Record: tokens/s, memory usage, time-to-first-token
- Confirm KV cache is accessible and sliceable
- This is the **baseline** all speculative numbers compare against

### 0.4 Baseline: Draft Model on CPU/GPU
- Download the chosen draft model in HF format
- Run inference via `transformers` and via `mlx-lm` (if MLX version available)
- Record: tokens/s, memory usage
- Confirm it produces reasonable text

**Phase 0 deliverables:**
- `requirements.txt` and `scripts/setup.sh`
- `benchmarks/baseline_mlx.py` — target model baseline measurements
- Tokenizer compatibility report (pass/fail + details)
- Draft model selection decision documented

---

## Phase 1: Draft Model on ANE (Days 2–3)

**Goal:** Get the draft model running on the Neural Engine via Core ML and measure its performance.

### 1.1 Core ML Conversion
- Trace the draft model with `torch.jit.trace` at a fixed sequence length (e.g., 128)
- Convert with `coremltools.convert`:
  - `compute_units=ct.ComputeUnit.CPU_AND_NE`
  - `convert_to="mlprogram"`
  - `minimum_deployment_target=ct.target.macOS15`
- Save as `.mlpackage`
- Handle likely conversion issues:
  - Unsupported ops (may need custom op replacements)
  - Dynamic shapes (must be fixed)
  - Attention mask handling

### 1.2 ANE Optimization (if needed)
- If Core ML falls back to GPU/CPU for most ops, apply ANE optimizations:
  - Study `apple/ml-ane-transformers` patterns
  - Replace `nn.Linear` with `nn.Conv2d(C_in, C_out, 1)`
  - Reshape to (B, C, 1, S) channel-first format
  - Ensure all tensors have 64-byte-aligned last axis
- Consider using ANEMLL's conversion pipeline instead of raw coremltools
- Re-convert and verify ANE execution via Xcode Instruments or `coremltools` profiling

### 1.3 Draft Engine Wrapper
- Write `src/draft/engine.py`:
  - Load Core ML model
  - Accept input token ids
  - Return logits (probability distributions)
  - Handle padding for fixed sequence length
- Write `src/draft/convert.py`:
  - Script to convert HF model → Core ML
  - Configurable sequence length, compute units

### 1.4 Draft Engine Benchmarks
- Measure: tokens/s on ANE, latency per forward pass
- Verify with Activity Monitor / `powermetrics` that ANE is actually being used
- Compare: ANE inference speed vs same model on GPU via MLX
- Confirm: draft model is fast enough to be beneficial (should be much faster than target)

**Phase 1 deliverables:**
- `src/draft/convert.py` — conversion script
- `src/draft/engine.py` — ANE inference wrapper
- `models/draft.mlpackage` — compiled draft model
- Benchmark numbers: ANE draft speed, memory usage
- Xcode Instruments profile showing ANE execution

---

## Phase 2: Speculative Decoding — Sequential (Days 4–5)

**Goal:** Implement the full speculative decoding loop with draft on ANE and target on GPU, running sequentially (not yet parallel).

### 2.1 Target Engine Wrapper
- Write `src/target/engine.py`:
  - Load MLX model
  - `prefill(tokens) → kv_cache` — initial prompt processing
  - `verify(draft_tokens, kv_cache) → list[Distribution]` — batched verification
  - `sample(distribution) → token` — sampling from target distribution
  - `truncate_cache(kv_cache, length)` — roll back after rejection

### 2.2 Rejection Sampling
- Write `src/orchestrator/sampling.py`:
  - `rejection_sample(target_prob, draft_prob, draft_token) → (accepted: bool, token: int)`
  - `sample_residual(target_dist, draft_dist) → int` — correction sampling
  - Temperature and top-p support

### 2.3 Orchestrator — Sequential Version
- Write `src/orchestrator/scheduler.py`:
  - Main generation loop as described in ARCHITECTURE.md
  - K (number of speculative tokens) configurable, default 5
  - Proper KV cache management for both draft and target
  - Token-by-token output (streaming)

### 2.4 End-to-End Pipeline
- Write `src/orchestrator/pipeline.py`:
  - CLI interface: `--draft-model`, `--target-model`, `--prompt`, `--K`, `--max-tokens`
  - Loads both engines, runs generation, prints output
  - Reports: total tokens generated, accepted/rejected stats, wall time, tokens/s

### 2.5 Correctness Testing
- Write `tests/test_orchestrator.py`:
  - Verify output distribution matches target-only generation (statistical test)
  - Test edge cases: all tokens rejected, all accepted, single token, EOS handling
  - Test KV cache rollback correctness

### 2.6 Benchmark vs Baseline
- Run `benchmarks/speculative.py`:
  - Same prompts as Phase 0 baseline
  - Record: tokens/s, acceptance rate, average tokens per speculation round
  - Compare against MLX-only baseline

**Phase 2 deliverables:**
- `src/target/engine.py`
- `src/orchestrator/sampling.py`
- `src/orchestrator/scheduler.py`
- `src/orchestrator/pipeline.py`
- `tests/test_orchestrator.py`
- Benchmark comparison: speculative vs baseline

---

## Phase 3: Optimization and Parallelism (Days 6–8)

**Goal:** Make draft and verification run in parallel; optimize for throughput.

### 3.1 Async Draft Pipeline
- Run draft engine on a background thread
- While GPU verifies batch N, ANE speculatively drafts batch N+1
- If batch N has rejections, discard speculative N+1 draft
- Synchronization via threading events or asyncio

### 3.2 Tune K (Speculation Length)
- Benchmark different K values: 3, 4, 5, 6, 8
- Find optimal K for this specific draft/target pair
- K too low → not enough speculation benefit
- K too high → more wasted computation on rejection

### 3.3 Draft KV Caching
- Implement stateful KV caching for the draft model in Core ML
- This means the Core ML model needs KV cache as input/output tensors
- Major complexity increase but significant speed benefit for longer contexts

### 3.4 Memory Optimization
- Profile peak memory usage at each stage
- Implement memory-aware context length limiting
- Consider: can we unload the draft model during target verification and vice versa? (probably not worth the overhead)

### 3.5 Tree-Based Speculation (Advanced)
- Instead of a single draft chain, generate a tree of candidates
- Verify multiple branches in one batched pass
- Similar to Medusa / EAGLE approach
- Higher acceptance rates but more complex implementation

**Phase 3 deliverables:**
- Parallel draft+verify pipeline
- Optimal K determination
- Final benchmark numbers with all optimizations

---

## Phase 4: Polish and Release (Day 9+)

### 4.1 CLI Tool
- Clean CLI with `argparse` or `click`
- Interactive chat mode
- Streaming output
- Configurable: model paths, K, context length, temperature, top-p

### 4.2 Documentation
- Usage guide
- Benchmark results table
- Troubleshooting (memory issues, ANE fallback, tokenizer mismatches)

### 4.3 Extensibility
- Support for other draft/target model pairs
- Configuration file for model combinations
- API server mode (OpenAI-compatible, like mlx_lm.server)

---

## Risk Checkpoints

After each phase, evaluate:

| Checkpoint | Question | If No... |
|---|---|---|
| Phase 0 | Do tokenizers match? | Find a compatible draft model or abandon |
| Phase 0 | Does the target model fit in memory? | Drop to Q3 or a smaller target |
| Phase 1 | Does the draft model actually run on ANE? | Apply ANE optimizations or use ANEMLL pipeline |
| Phase 1 | Is draft inference fast enough? (>100 tok/s) | If draft is slow, spec decoding won't help |
| Phase 2 | Is acceptance rate >50%? | Try a different draft model |
| Phase 2 | Is speculative faster than baseline? | Check overhead; tune K; may need parallel pipeline |
| Phase 3 | Does parallelism actually help? | Threading overhead may negate gains on M4 base chip |
