# Task Checklist

Tracks project progress. Each task is a self-contained unit of work. Dependencies are noted.
Status: `[x]` = done, `[-]` = in progress, `[ ]` = not started, `[!]` = blocked.

---

## Phase 0: Setup and Validation

- [x] **T0.1** Project directory structure created
- [x] **T0.2** `pyproject.toml` with dependencies
- [x] **T0.3** Python 3.12 venv with all deps installed and verified
- [x] **T0.4** `src/utils/config.py` — hardware-agnostic config with YAML and auto-detection
- [x] **T0.5** **GATE PASSED:** Tokenizer compatibility — Qwen3.5-0.8B and 27B identical (248K vocab)
- [x] **T0.6** `src/utils/memory.py` — memory monitoring
- [x] **T0.7** `.gitignore`, `config.example.yaml`, test suite
- [x] **T0.8** `benchmarks/baseline_mlx.py` — written, syntax verified (needs model download to run)
- [x] **T0.9** GitHub repo created: https://github.com/indysandbox/ane-gpu

### Decision Log

**ANEMLL fork: NO.** Different problem (single-model ANE inference vs two-model orchestration).

**Model format:** Local GGUF files (LM Studio) are for reference only. Core ML needs HF/PyTorch.
MLX uses mlx-community safetensors. Config system handles any path.

**Python 3.12** (not 3.14) — torch/coremltools compatibility.

---

## Phase 1: Draft Model

**Goal:** Get a draft model running fast enough to benefit speculative decoding.

### 1A: ANE Path (BLOCKED — hybrid architecture)

- [x] **T1.1** `src/draft/convert.py` — conversion script written with:
  - Custom `new_ones` op converter for coremltools
  - Static causal mask patch (bypasses unsupported mask ops)
  - RoPE analysis (Qwen3.5 already uses real-valued rotate_half — no patch needed)
  - Support for both `torch.export` and `jit.trace` paths

- [!] **T1.2** **BLOCKED:** Core ML conversion fails on Qwen3.5 hybrid architecture
  - Qwen3.5 uses "Gated Delta Networks" — 18 linear_attention + 6 full_attention layers
  - `linear_attention` layers use `causal_conv1d` and recurrent state updates
  - coremltools cannot convert `slice_update` with mismatched state dims
  - **This is architectural, not a simple op patch**
  - See `memory/architecture-notes.md` for detailed analysis

- [-] **T1.3** Research alternative ANE-compatible draft models:
  - Need: same Qwen3.5 tokenizer (248K vocab) + standard transformer architecture
  - Checking: Qwen3 (not 3.5), other small Qwen models, model surgery approaches
  - Checking: ANEMLL's Qwen3.5 support status

### 1B: GPU Fallback (ACTIVE — working path)

- [-] **T1.4** `src/draft/engine.py` — DraftEngine with dual backend:
  - MLX (GPU) backend: loads via `mlx_lm.load()`, fully functional
  - Core ML (ANE) backend: stubbed, ready for when conversion is solved
  - `propose(context, k)` returns draft tokens + probability distributions
  - `predict_next()` for single token prediction

- [ ] **T1.5** `tests/test_draft_engine.py`:
  - Unit tests with mocked model
  - Integration test (marked slow) with real Qwen3.5-0.8B via MLX

- [ ] **T1.6** `benchmarks/draft_benchmark.py`:
  - MLX draft speed: tokens/s, latency per forward pass
  - Compare: draft (0.8B) vs target (27B) speed ratio
  - This ratio determines speculative decoding's potential speedup

---

## Phase 2: Speculative Decoding Pipeline

**Goal:** Full end-to-end speculative decoding.

- [x] **T2.1** `src/target/engine.py` — TargetEngine (MLX/GPU):
  - `prefill()`, `verify()`, `truncate_cache()`
  - Returns full probability distributions for rejection sampling
  - Unit tests passing

- [x] **T2.3** `src/orchestrator/sampling.py` — Rejection sampling:
  - `rejection_sample()`, `compute_residual_distribution()`, `sample_from_distribution()`
  - 39 tests passing, handles all numerical edge cases
  - Statistical correctness verified (chi-squared + KL divergence)

- [-] **T2.5** `src/orchestrator/scheduler.py` — Main speculative decoding loop:
  - Draft → Verify → Accept/Reject → Bonus token → Cache rollback
  - Yields (tokens, stats) per round for streaming
  - Tracks acceptance rate, timing, tokens per round

- [-] **T2.6** `src/cli.py` — CLI entry point:
  - `--draft-model`, `--target-model`, `--prompt`, `--K`, `--max-tokens`
  - Streaming output, summary statistics

- [ ] **T2.7** `tests/test_e2e.py` — End-to-end integration tests

- [ ] **T2.8** `benchmarks/speculative.py` + `benchmarks/compare.py`:
  - Side-by-side: baseline MLX vs speculative decoding

---

## Phase 3: Optimization

- [ ] **T3.1** Tune K (speculation length): benchmark K = 3, 4, 5, 6, 8
- [ ] **T3.2** Async draft pipeline: draft N+1 while verifying N
- [ ] **T3.3** Draft model KV caching (Core ML stateful model or MLX cache)
- [ ] **T3.4** Memory optimization and context length limits
- [ ] **T3.5** Final benchmarks and RESULTS.md

---

## Phase X: ANE Breakthrough (when unblocked)

These tasks activate when we find an ANE-compatible draft model:

- [ ] **TX.1** Convert compatible draft model to Core ML .mlpackage
- [ ] **TX.2** Verify ANE execution with Xcode Instruments (green = ANE)
- [ ] **TX.3** Benchmark ANE draft vs MLX draft (should be faster + separate hardware)
- [ ] **TX.4** Enable true parallel pipeline (ANE drafts while GPU verifies)
- [ ] **TX.5** Benchmark heterogeneous (ANE+GPU) vs homogeneous (GPU+GPU) speculative decoding

---

## Phase 4: Open Source Polish

- [ ] **T4.1** CLI tool (clean interface, interactive chat mode)
- [ ] **T4.2** Documentation (usage guide, benchmarks, troubleshooting)
- [ ] **T4.3** Support for other model families
- [ ] **T4.4** API server mode (OpenAI-compatible)
- [ ] **T4.5** CI/CD (GitHub Actions, macOS runner)
- [ ] **T4.6** Contributing guide, issue templates

---

## Parallelism Map

```
Currently running in parallel:
├── T1.3  Research ANE-compatible draft models
├── T1.4  Build MLX draft engine (GPU fallback)
└── T2.5 + T2.6  Orchestrator + CLI

Next parallel batch:
├── T1.5  Draft engine tests
├── T2.7  E2E tests
└── T2.8  Benchmarks
```

## Notes for Agents

### Key Technical Facts
- Qwen3.5 ALL sizes use hybrid Gated Delta Networks (not standard transformer)
- Tokenizer: 248077 tokens (model embedding: 248320 padded)
- Qwen2.5 tokenizer is INCOMPATIBLE with Qwen3.5 (151K vs 248K)
- MLX operations are lazy — call `mx.eval()` for timing
- Data between MLX ↔ Core ML goes through NumPy arrays
- Config auto-detects hardware — don't hardcode chip/memory assumptions

### File Locations
- Config: `src/utils/config.py` (YAML + auto-detect)
- Memory: `src/utils/memory.py`
- Sampling: `src/orchestrator/sampling.py`
- Target engine: `src/target/engine.py`
- Draft conversion: `src/draft/convert.py` (currently blocked on ANE)
- Draft engine: `src/draft/engine.py` (being built — MLX backend)
- Scheduler: `src/orchestrator/scheduler.py` (being built)
- CLI: `src/cli.py` (being built)
- Tokenizer check: `scripts/check_tokenizers.py`
- Baseline benchmark: `benchmarks/baseline_mlx.py`
