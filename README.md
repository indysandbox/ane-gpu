# ANE Speculative Decoding for Apple Silicon

**Heterogeneous GPU+ANE inference: use the Apple Neural Engine as a speculative draft model while the GPU runs the target LLM.**

## The Idea

On Apple Silicon, the Neural Engine (ANE) sits idle during LLM inference — all current tools (MLX, llama.cpp, Ollama) use the GPU exclusively. Meanwhile, Apple's own research (Mirror-SD, ReDrafter) demonstrates that running a small draft model on the NPU *in parallel* with GPU verification yields 2.8x–5.8x speedups.

This project builds that pipeline for real consumer use: a small dense model on ANE proposes candidate tokens, while MLX verifies them on the GPU using the full target model. The two hardware units run concurrently with zero contention.

## Why This Works

| Property | ANE (Draft) | GPU via MLX (Target) |
|---|---|---|
| Hardware | 16-core Neural Engine | 10-core Metal GPU |
| Power draw | ~2.8W peak | ~10-15W during inference |
| Efficiency | 6.6 TFLOPS/W | ~1.0 TFLOPS/W |
| Idle state | True hardware shutdown (0W) | Clock gating |
| Best for | Small dense models, static graphs | Large/MoE models, dynamic routing |
| Memory path | Shared unified DRAM | Shared unified DRAM |

The draft model (Qwen3.5-0.8B) runs on ANE via Core ML. The target model (Qwen3.5-27B) runs on GPU via MLX. Both are from the same Qwen3.5 family and share the same tokenizer. Both access the same unified memory — no copies needed.

## Target Hardware

- MacBook Air M4, 24GB unified memory
- macOS 15.x (Sequoia) or later
- Python 3.11+

## Target Configuration

- **Draft model:** Qwen3.5-0.8B (dense, same Qwen3.5 tokenizer), FP16 on ANE via Core ML
- **Target model:** Qwen3.5-27B (dense, all 27B params active), Q4 quantized on GPU via MLX
- **Memory budget:** ~17GB target model + ~1.6GB draft model + ~3GB KV cache/overhead = ~21–22GB
- **Why 27B over 35B-A3B?** The 27B dense model uses ~17GB at Q4 (vs ~22GB for 35B MoE), freeing 3–5GB for KV cache and longer contexts. All 27B params are active per token, giving higher quality output. And as a dense model, the target engine is simpler to implement.

## Project Structure

```
ane-speculative-decoding/
├── README.md                  # This file
├── ARCHITECTURE.md            # Detailed system design
├── RESEARCH.md                # Background research and references
├── IMPLEMENTATION_PLAN.md     # Phased build plan for Claude Code
├── TASKS.md                   # Concrete task checklist
├── src/
│   ├── draft/                 # ANE draft model (Core ML)
│   │   ├── convert.py         # HF model → Core ML conversion
│   │   ├── engine.py          # ANE inference wrapper
│   │   └── tokenizer.py       # Shared tokenizer utilities
│   ├── target/                # GPU target model (MLX)
│   │   ├── engine.py          # MLX inference wrapper
│   │   └── kv_cache.py        # KV cache management
│   ├── orchestrator/          # Speculative decoding logic
│   │   ├── scheduler.py       # Token proposal + verification loop
│   │   ├── sampling.py        # Rejection sampling / acceptance
│   │   └── pipeline.py        # End-to-end generation pipeline
│   └── utils/
│       ├── memory.py          # Memory monitoring
│       ├── benchmark.py       # Throughput / latency measurement
│       └── config.py          # Configuration management
├── tests/
│   ├── test_draft_engine.py
│   ├── test_target_engine.py
│   ├── test_orchestrator.py
│   └── test_e2e.py
├── benchmarks/
│   ├── baseline_mlx.py        # MLX-only baseline (no speculation)
│   ├── speculative.py         # Full speculative pipeline benchmark
│   └── compare.py             # Side-by-side comparison
├── scripts/
│   ├── setup.sh               # Environment setup
│   ├── convert_draft.sh       # Draft model conversion helper
│   └── run.sh                 # Quick-start runner
├── pyproject.toml
└── requirements.txt
```

## Quick Start (once built)

```bash
# 1. Setup
./scripts/setup.sh

# 2. Convert draft model to Core ML for ANE
python src/draft/convert.py \
  --model Qwen/Qwen3.5-0.8B \
  --output models/draft.mlpackage \
  --compute-unit ANE

# 3. Run speculative generation
python -m src.orchestrator.pipeline \
  --draft-model models/draft.mlpackage \
  --target-model mlx-community/Qwen3.5-27B-4bit \
  --prompt "Explain speculative decoding in simple terms" \
  --num-speculative-tokens 5
```

## Key References

- [Mirror Speculative Decoding (Apple, Jan 2026)](https://machinelearning.apple.com/research/mirror) — GPU+NPU heterogeneous spec decoding
- [Recurrent Drafter / ReDrafter (Apple)](https://machinelearning.apple.com/research/recurrent-drafter) — RNN draft model, MLX implementation
- [Deploying Transformers on ANE (Apple)](https://machinelearning.apple.com/research/neural-engine-transformers) — ANE optimization principles
- [ANEMLL](https://github.com/Anemll/Anemll) — Open-source LLM→ANE conversion pipeline
- [maderix/ANE](https://github.com/maderix/ANE) — Reverse-engineered ANE benchmarks and training
- [apple/ml-ane-transformers](https://github.com/apple/ml-ane-transformers) — Apple's reference ANE transformer implementation

## License

MIT
