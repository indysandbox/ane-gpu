# CLAUDE.md — Instructions for Claude Code

## Project Summary

This project implements speculative decoding on Apple Silicon by using the Neural Engine (ANE) to run a small draft model while the GPU runs the large target model via MLX. The two compute units operate on physically separate hardware sharing unified memory — no contention.

**Read these files first, in this order:**
1. `README.md` — Overview and structure
2. `ARCHITECTURE.md` — System design, algorithm, memory budget
3. `RESEARCH.md` — Background on ANE, speculative decoding, MLX, Core ML
4. `IMPLEMENTATION_PLAN.md` — Phased build plan
5. `TASKS.md` — Concrete task checklist (work through these in order)

## Key Constraints

### Hardware: MacBook Air M4, 24GB
- Memory is THE bottleneck. Budget: ~20GB target model + ~1GB draft + ~3GB overhead.
- Monitor memory continuously. Use `mlx.core.metal.get_peak_memory()` and `resource.getrusage`.
- If anything OOMs, first try: smaller quantization (Q3), shorter context, smaller draft model.

### This runs on macOS only
- Core ML and the Neural Engine require macOS 15+ on Apple Silicon.
- MLX requires Apple Silicon.
- All code targets macOS exclusively. No Linux/Windows compatibility needed.
- Python 3.11+ (native arm64 build, NOT Rosetta/x86).

### Two separate ML runtimes
- **MLX** for the target model (GPU). Import: `import mlx.core as mx`, `from mlx_lm import load`.
- **Core ML** for the draft model (ANE). Import: `import coremltools as ct`, `import CoreMLTools`.
- These are completely separate frameworks. They share unified memory at the hardware level but have no software integration. Data transfer between them goes through NumPy arrays.

## Critical Path Decisions

### Task T0.5 is a hard gate
The tokenizer compatibility check determines whether this project is viable. If the draft and target models use different tokenizers, speculative decoding produces garbage. Run this FIRST before investing in conversion/optimization.

**If tokenizers don't match:** Look for a model in the same family. The Qwen3.5 small models (0.8B, 2B) should share the tokenizer with 27B. If even those don't work, consider distilling a small model from the target family (out of scope for v1).

### Core ML conversion may require iteration
Converting a HuggingFace transformer to Core ML for ANE execution is not always straightforward. Common issues:
- Some ops aren't supported by ANE and fall back to CPU/GPU
- Dynamic shapes aren't supported (must trace with fixed input size)
- The model may need architectural changes (Linear → Conv2d, reshape to NCHW)

If basic `coremltools.convert()` doesn't work, try in this order:
1. Fix unsupported ops by replacing them in the PyTorch model before tracing
2. Use Apple's `ml-ane-transformers` patterns as a guide
3. Try ANEMLL's conversion pipeline (it handles Qwen architectures)
4. As a fallback, run the draft model on GPU via MLX (loses the ANE benefit but still enables speculative decoding)

## Code Style

- Python 3.11+ with type hints everywhere
- Dataclasses for configuration
- No global state — everything passed explicitly
- Logging via `logging` module (not print statements) for anything that's not user-facing output
- NumPy as the interchange format between MLX and Core ML
- Tests with `pytest`

## What Success Looks Like

1. **Minimum viable:** Speculative decoding works end-to-end, producing coherent text, and is measurably faster than baseline MLX generation. Even 1.2x speedup counts.
2. **Good:** ANE is confirmed running the draft model (not falling back to GPU). Speedup >1.5x.
3. **Great:** Parallel draft+verify pipeline working. Speedup >2x. Publishable benchmark results.

## What to Do If Stuck

- **Core ML conversion fails:** Try a different/smaller draft model. Or fall back to running draft on GPU via MLX (still validates the speculative decoding logic).
- **ANE isn't being used:** Check Xcode Instruments → Core ML Instrument. Green = ANE. If everything is blue/orange, the model needs ANE optimization.
- **OOM:** Drop target to Q3 quantization. Reduce context length. Use 0.5B draft instead of 0.8B.
- **Low acceptance rate:** This means draft/target disagree too often. Try: (a) verify tokenizers match exactly, (b) try a larger draft model, (c) reduce K.
- **Speculative decoding is slower than baseline:** The overhead (draft inference + rejection sampling) exceeds the savings. Check: is the draft model actually fast? Is K too high? Is the acceptance rate too low?
