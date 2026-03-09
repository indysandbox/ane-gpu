# Research Background

## Apple Neural Engine (ANE) — M4

### Hardware Specifications
- **Cores:** 16
- **Marketed TOPS:** 38 (INT8) — misleading; actual FP16 throughput is ~19 TFLOPS
- **Peak power:** 2.8W (true hardware shutdown at 0W when idle)
- **Compute efficiency:** ~6.6 TFLOPS/W (vs ~1.0 for M4 GPU, ~0.08 for A100)
- **On-chip SRAM:** ~32MB (performance cliff when working set exceeds this)
- **Native precision:** FP16 compute; INT8 dequantized to FP16 before compute
- **Architecture:** Graph execution engine — submits compiled neural programs, not individual ops
- **Queue depth:** 127 evaluation requests in-flight
- **Access model:** Core ML (public), or private `_ANEClient` API (undocumented)

### ANE Programming Constraints
- Fundamentally a **convolution engine** — 1x1 convolutions are ~3x faster than matmul
- Prefers **4D tensors** in (B, C, 1, S) format — channel-first, sequence on last axis
- Last axis must be **contiguous and 64-byte aligned**
- **Static computation graphs** — compiled once, run many times
- **Weights baked at compile time** — cannot hot-swap weights
- Unsupported ops fall back to CPU/GPU, breaking ANE exclusivity
- CoreML adds **2–4x overhead** for small operations vs direct `_ANEClient` access

### Key Sources
- [maderix/ANE benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) — Most detailed independent ANE benchmarks (March 2026)
- [apple/ml-ane-transformers](https://github.com/apple/ml-ane-transformers) — Apple's reference transformer implementation for ANE
- [Deploying Transformers on ANE (Apple)](https://machinelearning.apple.com/research/neural-engine-transformers) — Optimization principles
- [hollance/neural-engine](https://github.com/hollance/neural-engine) — Community documentation of ANE behavior

## Speculative Decoding — Theory

### Core Papers
1. **Leviathan et al. (2023)** — "Fast Inference from Transformers via Speculative Decoding"
   - Original formulation of speculative decoding
   - Proves output distribution is identical to target model (no quality loss)
   - Key insight: verification of K tokens costs ~same as generating 1 token

2. **Chen et al. (2023)** — "Accelerating Large Language Model Decoding with Speculative Sampling"
   - Independent concurrent work on same idea
   - Detailed rejection sampling algorithm

### Apple's Speculative Decoding Research

3. **Mirror Speculative Decoding (Apple, Jan 2026)**
   - URL: https://machinelearning.apple.com/research/mirror
   - **Directly relevant: explicitly uses GPU+NPU heterogeneous execution**
   - Draft and target run on different accelerators in parallel
   - "Dual pipeline" — target verifies while draft speculatively generates next batch
   - 2.8x–5.8x speedups on 14B–66B models
   - 30% improvement over EAGLE3 baseline

4. **Recurrent Drafter / ReDrafter (Apple)**
   - URL: https://machinelearning.apple.com/research/recurrent-drafter
   - Uses lightweight RNN as draft model conditioned on LLM hidden states
   - Dynamic tree attention over beam search candidates
   - Up to 2.3x speedup on Apple Silicon via MLX on Metal GPU
   - **Note:** This uses GPU for both draft and target; our approach offloads draft to ANE

5. **EAGLE / EAGLE-2 / EAGLE-3**
   - Feature-level draft models that reuse target model's hidden states
   - High acceptance rates but tightly coupled to target model architecture

### Why Heterogeneous (ANE+GPU) is Better Than GPU-Only Speculation

In standard speculative decoding (both models on GPU), the draft and target **contend for the same compute resources**. The draft model's forward passes consume GPU cycles that could be used for verification.

With ANE+GPU, the draft runs on **physically separate hardware**:
- No GPU contention — target model gets full GPU bandwidth
- Draft runs at 2.8W vs 10-15W for GPU equivalent
- True parallelism possible (ANE has independent command queue)
- ANE queue depth of 127 allows pipelining multiple draft evaluations

## MLX Framework

### Relevant Capabilities
- Unified memory: arrays shared between CPU and GPU without copies
- Lazy evaluation: operations fused and optimized before execution
- NumPy-like API with PyTorch-style neural network packages
- `mlx-lm`: high-level package for LLM text generation and fine-tuning
- Supports quantized models (4-bit, 8-bit) from Hugging Face `mlx-community`
- KV cache exposed as sliceable array objects
- **Does NOT support ANE** — GPU and CPU only (confirmed by MLX lead developer)

### Key MLX Functions for This Project
```python
# Loading a model
from mlx_lm import load
model, tokenizer = load("mlx-community/Qwen3.5-27B-4bit")

# Forward pass with cache (for verification)
logits = model(input_ids, cache=kv_cache)

# KV cache is a list of (key, value) tuples per layer
# Can be sliced: cache[layer] = (k[:, :, :seq_len, :], v[:, :, :seq_len, :])
```

## Core ML for ANE

### Conversion Pipeline
```
HuggingFace Model (PyTorch)
    → torch.jit.trace (fixed input shapes)
    → coremltools.convert (to .mlpackage)
    → Core ML runtime (dispatches to ANE)
```

### Key coremltools Options
- `compute_units=ct.ComputeUnit.CPU_AND_NE` — prefer ANE, fall back to CPU
- `compute_units=ct.ComputeUnit.ALL` — let Core ML decide (often routes to GPU)
- `minimum_deployment_target=ct.target.macOS15` — enables latest ANE optimizations
- `convert_to="mlprogram"` — required for ANE (older "neuralnetwork" format doesn't support ANE well)

### Verifying ANE Execution
Use Xcode Instruments → Core ML Instrument to verify the model actually runs on ANE:
- Green = ANE
- Blue = GPU
- Orange = CPU

Or programmatically via `coremltools`:
```python
spec = ct.utils.load_spec("model.mlpackage")
# Check compute plan to see which ops run where
```

## ANEMLL Project

### What It Does
- Open-source pipeline: HuggingFace model → Core ML → ANE inference
- Supports LLaMA, Qwen, Qwen 2.5, Gemma 3 architectures
- Chunked model execution for larger models
- Context lengths up to 4096 (512–2048 recommended for ANE)
- Swift reference implementation for iOS/macOS apps

### Relevance to This Project
- Demonstrates that transformer models CAN run on ANE via Core ML
- Provides model conversion utilities we may be able to reuse or learn from
- Shows practical ANE context length limits (512–2048 optimal)
- **Does NOT do speculative decoding** — only runs a single model on ANE

### Limitations
- Only dense architectures (no MoE support on ANE)
- Context length limited by ANE memory constraints
- Requires careful model chunking for larger models

## Draft Model: Qwen3.5-0.8B

### Why This Model
- **Same family as target:** Both 0.8B and 27B are part of the Qwen3.5 release (Feb 24, 2026)
- **Same tokenizer:** 262K vocabulary, identical token→id mappings, same special tokens
- **Small and dense:** 0.8B params, no MoE routing — perfect for ANE's static graph requirement
- **FP16 size:** ~1.6GB — fits easily alongside the 27B target
- **Trained together:** The small models were distilled from larger Qwen3.5 models, so they share learned representations, which should boost acceptance rate

### Tokenizer Verification
Still verify programmatically in Phase 0:
```python
from transformers import AutoTokenizer
draft_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
target_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
assert draft_tok.get_vocab() == target_tok.get_vocab(), "Tokenizers must match!"
```

### Fallback Options
If 0.8B doesn't convert cleanly to Core ML for ANE:
1. Try Qwen3.5-2B (larger but still reasonable for ANE)
2. Run 0.8B on GPU via MLX as draft (loses ANE parallelism but validates spec decoding logic)

## Target Model: Qwen3.5-27B

### Architecture
- 27B total parameters, **all active per token** (dense, not MoE)
- Hybrid attention: Gated Delta Networks
- Native context: 262,144 tokens (extensible to 1M)
- Multimodal: vision + language (early fusion)
- Tokenizer: 262K vocabulary
- License: Apache 2.0

### Why 27B Instead of 35B-A3B
- **Memory:** ~17GB at Q4 vs ~22GB for 35B MoE. Frees 5GB for KV cache and headroom.
- **Quality:** All 27B params active per token. Better for complex reasoning, coding, creative tasks.
- **Simplicity:** Dense model means simpler target engine — no MoE routing logic.
- **Better for speculative decoding:** The target model's verification step is a straightforward batched forward pass with no dynamic routing.

### Memory at Various Quantizations
- FP16: ~54GB (does not fit)
- Q8: ~27GB (does not fit)
- Q5: ~21GB (fits with some headroom)
- Q4: ~17GB (fits comfortably — **recommended**)
- Q3: ~14GB (fits with lots of headroom, quality tradeoff)

### Performance on Apple Silicon (via MLX)
- M4 Air at Q4: estimated 15–25 tok/s (all 27B params active = slower than 35B MoE's 3B active)
- This is where speculative decoding helps most — the target model is slow, so batching verification of multiple draft tokens yields large gains
