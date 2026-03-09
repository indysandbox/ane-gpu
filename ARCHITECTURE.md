# Architecture: ANE Speculative Decoding

## System Overview

This system implements speculative decoding across two heterogeneous compute units on Apple Silicon:

1. **Draft Engine (ANE):** A small dense transformer (~0.5B params) compiled to Core ML, pinned to the Neural Engine. Proposes candidate token sequences cheaply.
2. **Target Engine (GPU):** The full target model (Qwen3.5-27B) running on MLX via Metal GPU. Verifies draft proposals in batched forward passes.
3. **Orchestrator:** Manages the speculative decoding loop — token proposal, batched verification, rejection sampling, and KV cache rollback.

```
┌─────────────────────────────────────────────────────┐
│                   Orchestrator                       │
│  ┌──────────────┐          ┌──────────────────────┐ │
│  │  Draft Engine │  tokens  │   Target Engine      │ │
│  │  (ANE/CoreML) │────────▶│   (GPU/MLX)          │ │
│  │  Qwen3.5-0.8B │         │   Qwen3.5-27B        │ │
│  │  ~2.8W        │◀────────│   ~10-15W            │ │
│  │  ~1.6GB mem   │ accept/ │   ~17GB mem           │ │
│  └──────────────┘ reject   └──────────────────────┘ │
│         │                           │                │
│         └───────────┬───────────────┘                │
│                     │                                │
│              Shared Unified Memory (24GB)             │
└─────────────────────────────────────────────────────┘
```

## Speculative Decoding Algorithm

We implement standard speculative decoding with rejection sampling (Leviathan et al., 2023; Chen et al., 2023):

### Step-by-step flow:

1. **Draft phase:** Given the current token sequence, the draft model autoregressively generates K candidate tokens (K=4–8 typically). Each draft token also produces a probability distribution q(x).

2. **Verify phase:** The target model processes all K draft tokens in a single batched forward pass, producing probability distributions p(x) for each position. This is much cheaper than K separate autoregressive steps because the target model processes them in parallel.

3. **Accept/reject:** For each draft token i (in order):
   - Sample r ~ Uniform(0, 1)
   - If r < p(x_i) / q(x_i): **accept** token x_i, continue to i+1
   - Else: **reject** token x_i, sample a correction token from norm(max(0, p(x) - q(x))), stop

4. **Bonus token:** If all K drafts are accepted, sample one additional token from the target model's distribution at position K+1 (this comes "free" from the verification pass).

5. **KV cache management:** Roll back the KV cache to the last accepted position. This is critical — the target model's KV cache must be truncatable.

### Expected speedup:

If the draft model's acceptance rate is α (fraction of tokens accepted on average), and generating K draft tokens takes time T_draft, while a single target verification of K tokens takes T_verify ≈ T_single_target (since it's batched):

```
Speedup ≈ (α * K + 1) / (T_draft/T_target + 1)
```

For a well-matched draft model (α ≈ 0.7–0.85), K=5, and fast draft (T_draft << T_target), this yields 2–4x speedup.

## Draft Engine Design (ANE)

### Model Selection

**Primary choice: Qwen3.5-0.8B as draft, Qwen3.5-27B as target.**

Both models are from the same Qwen3.5 family (released Feb 24, 2026), sharing the same 262K tokenizer, chat template, and special tokens. This eliminates tokenizer compatibility risk.

At 0.8B parameters (FP16 ~1.6GB), the draft model fits comfortably alongside the 27B target (~17GB at Q4), leaving ~5GB headroom for KV cache and system overhead on 24GB.

**Verify tokenizer compatibility in Phase 0** (should pass, but always confirm programmatically):
```python
from transformers import AutoTokenizer
draft_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
target_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B")
assert draft_tok.get_vocab() == target_tok.get_vocab(), "Tokenizers must match!"
```

### Core ML Conversion

The draft model is converted from Hugging Face format to Core ML (.mlpackage) using `coremltools`:

```python
import coremltools as ct
import torch

# Key ANE optimization principles from Apple's research:
# 1. Use channel-first layout: (B, C, 1, S) not (B, S, C)
# 2. Express linear layers as 1x1 convolutions
# 3. Keep tensor dimensions aligned to 64 bytes on last axis
# 4. Use FP16 precision (ANE native format)
# 5. Compile the full model as one graph — don't split layers

model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.CPU_AND_NE,  # Prefer ANE
    minimum_deployment_target=ct.target.macOS15,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, seq_len), dtype=np.int32),
    ],
)
```

### ANE-Specific Optimizations

Following Apple's own `ml-ane-transformers` reference:

- **Data format:** Reshape all tensors to (B, C, 1, S) — the ANE's preferred 4D layout. Sequence on last axis must be contiguous and 64-byte aligned.
- **1x1 convolution trick:** Replace `nn.Linear` layers with `nn.Conv2d(in_c, out_c, 1)`. The ANE's convolution datapath is ~3x faster than its matmul path.
- **Avoid unsupported ops:** The ANE cannot run certain operations (e.g., complex attention masks, dynamic shapes). Any unsupported op causes Core ML to fall back to CPU/GPU, breaking the performance story.
- **Fixed sequence length:** Compile the draft model with a fixed input size (e.g., 128 or 256 tokens). Pad shorter inputs.

### KV Caching for Draft Model

The draft model generates K tokens autoregressively. For K=5, this is 5 sequential forward passes through the draft model. Two approaches:

**Option A — Stateless (simpler):** Run the full context through the draft model each time. Feasible for short contexts if the model is small enough. Wasteful but simpler to implement.

**Option B — Cached (faster):** Maintain a separate KV cache for the draft model. This requires the Core ML model to accept and output KV cache tensors. More complex to set up but dramatically faster for longer contexts. This is what ANEMLL does.

Start with Option A, migrate to Option B once the basic pipeline works.

## Target Engine Design (GPU)

### MLX Integration

The target model runs via `mlx-lm` or direct MLX API:

```python
import mlx.core as mx
from mlx_lm import load, generate_step

model, tokenizer = load("mlx-community/Qwen3.5-27B-4bit")
```

Key capabilities needed:
- **Batched forward pass:** Given K draft tokens, compute logits for all positions in one call. MLX supports this natively — it's just a prefill-style forward pass.
- **KV cache access:** Need to read/write/truncate the KV cache. `mlx-lm` exposes cache objects that can be sliced.
- **Probability extraction:** Need raw logits (not just argmax tokens) for the rejection sampling step.

### Verification Forward Pass

The critical insight: verifying K draft tokens is nearly as fast as generating 1 token. The target model processes the K tokens as if they were a prompt (parallel, not sequential). The compute cost scales with K, but since it's parallelized across the GPU, wall-clock time is much less than K × single-token time.

```python
def verify(self, draft_tokens: list[int], kv_cache) -> tuple[list[Distribution], KVCache]:
    """
    Run target model on draft tokens, return per-position distributions.
    """
    input_ids = mx.array([draft_tokens])
    logits = self.model(input_ids, cache=kv_cache)
    # logits shape: (1, K, vocab_size)
    # Return distributions for rejection sampling
    return [softmax(logits[0, i]) for i in range(len(draft_tokens))]
```

## Orchestrator Design

### Main Loop

```python
def generate(prompt_tokens, max_tokens, K=5):
    draft_engine = DraftEngine("models/draft.mlpackage")
    target_engine = TargetEngine("mlx-community/Qwen3.5-27B-4bit")

    tokens = list(prompt_tokens)
    target_cache = target_engine.prefill(tokens)
    generated = 0

    while generated < max_tokens:
        # 1. Draft K tokens on ANE
        draft_tokens, draft_probs = draft_engine.propose(tokens, K)

        # 2. Verify on GPU (single batched forward pass)
        target_probs = target_engine.verify(draft_tokens, target_cache)

        # 3. Accept/reject with rejection sampling
        accepted = []
        for i in range(K):
            r = random.random()
            if r < target_probs[i][draft_tokens[i]] / draft_probs[i][draft_tokens[i]]:
                accepted.append(draft_tokens[i])
            else:
                # Sample correction token
                corrected = sample_from_residual(target_probs[i], draft_probs[i])
                accepted.append(corrected)
                break

        # If all K accepted, get bonus token from target
        if len(accepted) == K:
            bonus = sample(target_probs[K])  # From the K+1 position
            accepted.append(bonus)

        # 4. Update state
        tokens.extend(accepted)
        generated += len(accepted)
        target_cache.truncate(len(tokens))  # Roll back to accepted length
        draft_engine.sync(tokens)            # Reset draft to accepted state

    return tokens
```

### Parallelism (Advanced — Phase 2)

In the basic version above, draft and verify run sequentially. The real speedup comes from **pipelining**: while the GPU verifies batch N, the ANE drafts batch N+1 (speculatively, assuming all of batch N is accepted). If batch N has rejections, we discard the speculative N+1 draft.

This requires:
- Threading or async dispatch for ANE and GPU
- Speculative KV cache branching for the draft model
- Careful synchronization on rejection

Implement this in Phase 2 after the sequential version works.

## Memory Budget (24GB M4 Air)

| Component | Estimated Memory |
|---|---|
| macOS + system overhead | ~3–4 GB |
| Target model (Q4, 27B dense) | ~17 GB |
| Draft model (FP16, 0.8B) | ~1.6 GB |
| Target KV cache (4K context) | ~0.5 GB |
| Draft KV cache | ~0.05 GB |
| Python + framework overhead | ~0.5 GB |
| **Total** | **~22.5 GB** |

This leaves ~1.5GB of breathing room — much more comfortable than the 35B-A3B pairing. This headroom allows for longer context lengths (potentially 8K–16K) or higher quantization (Q5 at ~21GB).

## Failure Modes and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Tokenizer mismatch between draft/target | Spec decoding fails completely | Verify tokenizer compatibility first; use same family |
| Core ML falls back to CPU/GPU | ANE not actually used; no parallelism | Profile with Xcode Instruments; fix unsupported ops |
| Memory pressure causes swapping | Catastrophic slowdown | Monitor memory; use smaller quantization; limit context |
| Low acceptance rate (<50%) | Spec decoding slower than baseline | Try a better-matched draft model; tune K |
| CoreML dispatch overhead dominates | Draft too slow relative to benefit | Batch draft operations; consider stateful caching |
