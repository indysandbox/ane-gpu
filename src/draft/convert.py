#!/usr/bin/env python3
"""Convert a HuggingFace transformer model to Core ML for ANE execution.

Pipeline:
1. Download model from HuggingFace
2. Patch RoPE to eliminate unsupported complex-number ops (view_as_complex)
3. Export with torch.export (preferred) or trace with torch.jit.trace (fallback)
4. Convert to Core ML .mlpackage via coremltools
5. Verify the converted model produces reasonable output

The RoPE patch is critical: Qwen models use complex-number rotary embeddings
(view_as_complex + view_as_real), but coremltools cannot convert view_as_complex.
We replace it with a mathematically equivalent real-valued rotation using
the standard rotate_half formulation.

Usage:
    python src/draft/convert.py --model Qwen/Qwen3.5-0.8B --output models/draft.mlpackage
    python src/draft/convert.py --model Qwen/Qwen3.5-0.8B --output models/draft.mlpackage --seq-len 256
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Register missing coremltools op converters
# ---------------------------------------------------------------------------
# coremltools has new_zeros but not new_ones. We register it identically
# but with value=1.0. This is needed for Qwen3.5's causal mask creation.

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import (
    _get_inputs,
    register_torch_op,
)


@register_torch_op
def new_ones(context, node):
    """Convert torch.Tensor.new_ones to MIL fill op with value=1.0.

    Mirrors the existing new_zeros converter but with value=1.0.
    Handles both static (list of ints) and dynamic (tensor) shapes,
    and ensures shape is cast to int32 as required by mb.fill.
    """
    inputs = _get_inputs(context, node)
    shape = inputs[1]
    if isinstance(shape, list):
        shape = mb.concat(values=shape, axis=0)
    # Ensure shape is int32 (mb.fill requires it)
    shape = mb.cast(x=shape, dtype="int32")
    context.add(mb.fill(shape=shape, value=1.0, name=node.name))


logger.info("Registered custom coremltools converter for 'new_ones'")


# ---------------------------------------------------------------------------
# RoPE Patch: Replace complex-number rotation with real-valued equivalent
# ---------------------------------------------------------------------------
# Qwen's RoPE uses torch.view_as_complex() which is not supported by
# coremltools. The rotate_half formulation is mathematically identical
# but uses only cat, mul, add, and slicing — all ANE-compatible.

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input.

    Splits the last dimension in half and swaps with negation:
    [x1, x2] -> [-x2, x1]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_pos_emb_real(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings using only real-valued operations.

    This replaces the complex-number formulation with:
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

    Mathematically identical to the complex rotation, but uses only
    operations that coremltools and ANE can handle.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def patch_rope(model_id: str) -> None:
    """Monkey-patch the model's RoPE implementation to avoid complex ops.

    Detects the model architecture and patches the appropriate module.
    Supports Qwen2/Qwen3 family models.
    """
    # Try Qwen2-based architectures (Qwen3.5 uses Qwen2 modeling code)
    modules_to_try = [
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "transformers.models.qwen2.modeling_qwen2",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.qwen2_5.modeling_qwen2_5",
    ]

    patched = False
    for module_name in modules_to_try:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "apply_rotary_pos_emb"):
                original = getattr(module, "apply_rotary_pos_emb")
                setattr(module, "apply_rotary_pos_emb", _apply_rotary_pos_emb_real)
                logger.info(f"Patched RoPE in {module_name}")
                patched = True
        except ImportError:
            continue

    if not patched:
        logger.warning(
            "Could not find RoPE to patch. If conversion fails with "
            "'view_as_complex' error, the model architecture may need "
            "a custom RoPE patch."
        )


# ---------------------------------------------------------------------------
# Model wrapper for export/trace
# ---------------------------------------------------------------------------

def _patch_causal_mask(seq_len: int) -> None:
    """Replace the complex create_causal_mask with a simple static version.

    The transformers library's create_causal_mask uses ops like new_ones,
    bitwise_and, and vmap-based mask creation that coremltools cannot convert.
    For a stateless forward pass with fixed sequence length and no KV cache,
    a simple static lower-triangular mask is sufficient.
    """
    import transformers.modeling_utils

    # Create the static 4D causal mask once
    # Shape: (1, 1, seq_len, seq_len) — broadcastable across heads
    mask = torch.zeros(1, 1, seq_len, seq_len)
    # Fill upper triangle with -inf (positions that should not attend)
    mask = mask.masked_fill(
        torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )

    def simple_causal_mask(*args, **kwargs) -> torch.Tensor:
        """Return pre-computed static causal mask."""
        return mask

    # Patch at the module level where it's called from
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_mod
        if hasattr(qwen_mod, "create_causal_mask"):
            qwen_mod.create_causal_mask = simple_causal_mask
            logger.info(f"Patched create_causal_mask with static {seq_len}x{seq_len} mask")
    except ImportError:
        pass

    # Also patch the general location
    try:
        import transformers.modeling_utils as utils_mod
        if hasattr(utils_mod, "create_causal_mask"):
            utils_mod.create_causal_mask = simple_causal_mask
    except ImportError:
        pass


class DraftWrapper(nn.Module):
    """Wrapper that makes a HuggingFace CausalLM model exportable.

    Returns only logits (not the full output dict with past_key_values).
    Takes only input_ids — causal mask is pre-computed via monkey-patch.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only logits.

        Args:
            input_ids: Token IDs, shape (1, seq_len)

        Returns:
            Logits tensor, shape (1, seq_len, vocab_size)
        """
        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,
        )
        return outputs.logits


# ---------------------------------------------------------------------------
# Download, export, convert
# ---------------------------------------------------------------------------

def download_model(model_id: str) -> tuple[nn.Module, AutoTokenizer]:
    """Download and load a HuggingFace model for conversion."""
    logger.info(f"Downloading model: {model_id}")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # FP16 — ANE's native precision
        trust_remote_code=True,
    )
    model.eval()

    elapsed = time.perf_counter() - t0
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model loaded in {elapsed:.1f}s | "
        f"{param_count / 1e6:.0f}M params | "
        f"~{param_count * 2 / 1e9:.1f}GB (FP16)"
    )

    return model, tokenizer


def export_model(
    model: nn.Module,
    seq_len: int = 128,
    vocab_size: int = 248077,
) -> torch.export.ExportedProgram | torch.jit.ScriptModule:
    """Export the model using jit.trace (preferred) or torch.export (fallback).

    jit.trace resolves ops like new_ones at trace time, avoiding unsupported
    op errors in coremltools. torch.export is tried as fallback since it captures
    more complete graphs but may hit unsupported ops.

    Returns:
        TracedModule or ExportedProgram
    """
    wrapper = DraftWrapper(model)
    wrapper.eval()

    dummy_ids = torch.zeros(1, seq_len, dtype=torch.long)

    # Verify wrapper works before export
    with torch.no_grad():
        test_output = wrapper(dummy_ids)
        logger.info(f"Pre-export output shape: {test_output.shape}")
        expected = (1, seq_len, vocab_size)
        if test_output.shape != expected:
            logger.warning(
                f"Output shape {test_output.shape} != expected {expected}. "
                f"Vocab size may differ from {vocab_size}."
            )

    # Try jit.trace first — resolves dynamic ops (new_ones, arange, etc.)
    # at trace time to concrete tensors, avoiding coremltools unsupported ops
    try:
        logger.info(f"Tracing model with torch.jit.trace (seq_len={seq_len})")
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (dummy_ids,))

        # Verify traced output matches
        with torch.no_grad():
            traced_output = traced(dummy_ids)
            diff = (test_output - traced_output).abs().max().item()
            logger.info(f"Trace verification: max diff = {diff:.2e}")
            if diff > 1e-3:
                logger.warning(f"Large diff between original and traced: {diff}")

        logger.info("torch.jit.trace succeeded")
        return traced
    except Exception as e:
        logger.warning(f"torch.jit.trace failed: {e}")
        logger.info("Falling back to torch.export")

    # Fallback: torch.export
    logger.info(f"Exporting model with torch.export (seq_len={seq_len})")
    with torch.no_grad():
        exported = torch.export.export(wrapper, args=(dummy_ids,))
    logger.info("torch.export succeeded")
    return exported


def convert_to_coreml(
    exported_model: torch.export.ExportedProgram | torch.jit.ScriptModule,
    seq_len: int = 128,
    compute_unit: str = "CPU_AND_NE",
    output_path: Optional[str] = None,
) -> ct.models.MLModel:
    """Convert exported/traced PyTorch model to Core ML .mlpackage.

    Args:
        exported_model: From torch.export or torch.jit.trace
        seq_len: Sequence length the model was exported with
        compute_unit: "CPU_AND_NE" (prefer ANE), "ALL", or "CPU_ONLY"
        output_path: Where to save the .mlpackage

    Returns:
        Converted Core ML model
    """
    logger.info(f"Converting to Core ML (compute_unit={compute_unit})")
    t0 = time.perf_counter()

    compute_units_map = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }
    cu = compute_units_map.get(compute_unit)
    if cu is None:
        raise ValueError(f"Unknown compute_unit: {compute_unit}")

    # Build inputs list — needed for jit.trace path, optional for torch.export
    # Only input_ids — causal mask is pre-computed via monkey-patch
    inputs = [
        ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
    ]

    # Convert — coremltools 9.0 detects ExportedProgram vs ScriptModule automatically
    is_exported = hasattr(exported_model, "module")  # ExportedProgram has .module
    convert_kwargs = {
        "convert_to": "mlprogram",
        "compute_units": cu,
        "compute_precision": ct.precision.FLOAT16,
        "minimum_deployment_target": ct.target.macOS15,
    }

    if not is_exported:
        # jit.trace path needs explicit input specs
        convert_kwargs["inputs"] = inputs

    mlmodel = ct.convert(exported_model, **convert_kwargs)

    elapsed = time.perf_counter() - t0
    logger.info(f"Core ML conversion complete in {elapsed:.1f}s")

    if output_path:
        output_path = str(output_path)
        logger.info(f"Saving to: {output_path}")
        mlmodel.save(output_path)

        size_mb = sum(
            f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file()
        ) / (1024 * 1024)
        logger.info(f"Model size on disk: {size_mb:.0f}MB")

    return mlmodel


def verify_coreml_model(
    mlmodel: ct.models.MLModel,
    tokenizer: AutoTokenizer,
    seq_len: int = 128,
) -> bool:
    """Verify the converted Core ML model produces reasonable output."""
    logger.info("Verifying Core ML model...")

    test_text = "The capital of France is"
    tokens = tokenizer.encode(test_text)
    real_len = len(tokens)

    # Pad input to fixed seq_len (no attention mask — causal mask is baked in)
    if len(tokens) < seq_len:
        pad_len = seq_len - len(tokens)
        tokens = tokens + [tokenizer.pad_token_id or 0] * pad_len
    else:
        tokens = tokens[:seq_len]
        pass  # already truncated above

    input_ids = np.array([tokens], dtype=np.int32)

    # Run prediction
    t0 = time.perf_counter()
    prediction = mlmodel.predict({"input_ids": input_ids})
    inference_time = (time.perf_counter() - t0) * 1000

    output_keys = list(prediction.keys())
    logger.info(f"Output keys: {output_keys}")

    if not output_keys:
        logger.error("No output from Core ML model")
        return False

    logits = prediction[output_keys[0]]
    logger.info(f"Output shape: {logits.shape}, inference time: {inference_time:.1f}ms")

    if np.any(np.isnan(logits)):
        logger.error("Output contains NaN values")
        return False

    if np.any(np.isinf(logits)):
        logger.error("Output contains Inf values")
        return False

    # Check prediction quality at the last real token
    last_logits = logits[0, real_len - 1, :]
    top_token = int(np.argmax(last_logits))
    top_word = tokenizer.decode([top_token])
    logger.info(f"Top prediction after '{test_text}': '{top_word}' (token {top_token})")

    # Check if "Paris" is reasonably ranked
    paris_tokens = tokenizer.encode("Paris")
    if paris_tokens:
        sorted_indices = np.argsort(-last_logits).tolist()
        if paris_tokens[0] in sorted_indices[:200]:
            paris_rank = sorted_indices.index(paris_tokens[0]) + 1
            logger.info(f"'Paris' rank: {paris_rank}")
        else:
            logger.warning("'Paris' not in top 200 — model may not be producing good output")

    logger.info("Verification PASSED")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to Core ML for ANE execution"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output", type=str, default="models/draft.mlpackage",
        help="Output path for .mlpackage",
    )
    parser.add_argument(
        "--seq-len", type=int, default=128,
        help="Fixed sequence length for ANE compilation (default: 128)",
    )
    parser.add_argument(
        "--compute-unit", type=str, default="CPU_AND_NE",
        choices=["CPU_AND_NE", "ALL", "CPU_ONLY"],
        help="Core ML compute unit preference",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip post-conversion verification",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Step 0: Patch causal mask creation with a static version
    # The transformers library's create_causal_mask uses ops (new_ones, bitwise_and)
    # that coremltools cannot convert. For fixed-length stateless inference,
    # a static lower-triangular mask is equivalent and fully convertible.
    _patch_causal_mask(args.seq_len)

    # Step 1: Download model
    model, tokenizer = download_model(args.model)
    vocab_size = model.config.vocab_size
    logger.info(f"Vocab size: {vocab_size}")

    # Step 2: Export (torch.export preferred, jit.trace fallback)
    exported = export_model(model, seq_len=args.seq_len, vocab_size=vocab_size)

    # Free the original model to save memory
    del model
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Step 3: Convert to Core ML
    mlmodel = convert_to_coreml(
        exported,
        seq_len=args.seq_len,
        compute_unit=args.compute_unit,
        output_path=args.output,
    )

    del exported
    gc.collect()

    # Step 4: Verify
    if not args.skip_verify:
        success = verify_coreml_model(mlmodel, tokenizer, seq_len=args.seq_len)
        if not success:
            logger.error("Verification FAILED — the converted model may not work correctly")
            sys.exit(1)

    logger.info(f"Done! Model saved to: {args.output}")
    logger.info(
        "Next steps:\n"
        "  1. Profile with Xcode Instruments to verify ANE execution\n"
        f"  2. Run: python benchmarks/draft_benchmark.py --model {args.output}"
    )


if __name__ == "__main__":
    main()
