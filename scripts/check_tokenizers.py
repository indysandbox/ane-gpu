#!/usr/bin/env python3
"""Tokenizer compatibility checker for speculative decoding.

Verifies that draft and target models share the exact same tokenizer —
a hard requirement for speculative decoding to produce correct output.

Usage:
    python scripts/check_tokenizers.py Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-27B
    python scripts/check_tokenizers.py --all-qwen  # Check all Qwen3.5 sizes
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from transformers import AutoTokenizer


# Diverse test strings covering edge cases
TEST_STRINGS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "こんにちは世界",  # Japanese
    "مرحبا بالعالم",  # Arabic
    "🎉 Emoji test 🚀 with 💻 multiple 🎯 symbols",
    "x = 3.14159; y = -2.71828e10",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",  # Chat template
    "   \t\n\r   ",  # Whitespace
    "",  # Empty string
    "a" * 500,  # Long repetition
    'She said "hello" and he said \'goodbye\'',
    "C:\\Users\\path\\to\\file.txt",  # Windows path
    "https://example.com/path?query=value&other=123#fragment",
    "SELECT * FROM users WHERE id = 1; DROP TABLE users;--",
    "$$\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$",  # LaTeX
    "The temperature is -40°C which equals -40°F",
    "Line 1\nLine 2\nLine 3\n\n\nLine 6",
    "Mixed CaSe AnD UPPERCASE and lowercase",
    "Token1 Token2\tToken3\nToken4",
]


def compare_tokenizers(
    model_a: str,
    model_b: str,
    verbose: bool = False,
) -> bool:
    """Compare two tokenizers for exact compatibility.

    Returns True if tokenizers are compatible for speculative decoding.
    """
    print(f"\n{'='*60}")
    print(f"Comparing tokenizers:")
    print(f"  A: {model_a}")
    print(f"  B: {model_b}")
    print(f"{'='*60}\n")

    # Load tokenizers
    print("Loading tokenizers...")
    try:
        tok_a = AutoTokenizer.from_pretrained(model_a, trust_remote_code=True)
    except Exception as e:
        print(f"FAIL: Could not load tokenizer for {model_a}: {e}")
        return False

    try:
        tok_b = AutoTokenizer.from_pretrained(model_b, trust_remote_code=True)
    except Exception as e:
        print(f"FAIL: Could not load tokenizer for {model_b}: {e}")
        return False

    all_pass = True

    # 1. Vocabulary size
    vocab_a = tok_a.get_vocab()
    vocab_b = tok_b.get_vocab()
    if len(vocab_a) == len(vocab_b):
        print(f"  PASS  Vocab size: {len(vocab_a)}")
    else:
        print(f"  FAIL  Vocab size mismatch: {len(vocab_a)} vs {len(vocab_b)}")
        all_pass = False

    # 2. Vocabulary contents
    if vocab_a == vocab_b:
        print(f"  PASS  Vocab contents: identical")
    else:
        only_a = set(vocab_a.keys()) - set(vocab_b.keys())
        only_b = set(vocab_b.keys()) - set(vocab_a.keys())
        diff_ids = {t for t in set(vocab_a.keys()) & set(vocab_b.keys()) if vocab_a[t] != vocab_b[t]}
        print(f"  FAIL  Vocab contents differ:")
        if only_a:
            print(f"         Only in A: {len(only_a)} tokens (first 5: {list(only_a)[:5]})")
        if only_b:
            print(f"         Only in B: {len(only_b)} tokens (first 5: {list(only_b)[:5]})")
        if diff_ids:
            print(f"         Different IDs: {len(diff_ids)} tokens")
        all_pass = False

    # 3. Special tokens
    specials_match = True
    for attr in ["bos_token", "eos_token", "pad_token", "unk_token"]:
        val_a = getattr(tok_a, attr, None)
        val_b = getattr(tok_b, attr, None)
        if val_a == val_b:
            if verbose:
                print(f"  PASS  {attr}: {repr(val_a)}")
        else:
            print(f"  FAIL  {attr}: {repr(val_a)} vs {repr(val_b)}")
            specials_match = False
            all_pass = False
    if specials_match:
        print(f"  PASS  Special tokens: identical")

    # 4. Encode/decode round-trip
    encode_pass = True
    for i, text in enumerate(TEST_STRINGS):
        ids_a = tok_a.encode(text)
        ids_b = tok_b.encode(text)
        if ids_a != ids_b:
            print(f"  FAIL  Encode mismatch on test string {i}: {repr(text[:50])}")
            if verbose:
                print(f"         A: {ids_a[:20]}...")
                print(f"         B: {ids_b[:20]}...")
            encode_pass = False
            all_pass = False

        # Also check decode
        decoded_a = tok_a.decode(ids_a)
        decoded_b = tok_b.decode(ids_b)
        if decoded_a != decoded_b:
            print(f"  FAIL  Decode mismatch on test string {i}")
            encode_pass = False
            all_pass = False

    if encode_pass:
        print(f"  PASS  Encode/decode: all {len(TEST_STRINGS)} test strings match")

    # Summary
    print(f"\n{'='*60}")
    if all_pass:
        print(f"  RESULT: PASS — tokenizers are compatible for speculative decoding")
    else:
        print(f"  RESULT: FAIL — tokenizers are NOT compatible")
    print(f"{'='*60}\n")

    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Check tokenizer compatibility for speculative decoding")
    parser.add_argument("model_a", nargs="?", help="First model (draft)")
    parser.add_argument("model_b", nargs="?", help="Second model (target)")
    parser.add_argument("--all-qwen", action="store_true", help="Check all Qwen3.5 sizes against 27B")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    if args.all_qwen:
        target = "Qwen/Qwen3.5-27B"
        drafts = [
            "Qwen/Qwen3.5-0.8B",
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-9B",
        ]
        results = {}
        for draft in drafts:
            results[draft] = compare_tokenizers(draft, target, verbose=args.verbose)

        print("\n\nSummary:")
        print("-" * 40)
        for draft, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {status}  {draft} <-> {target}")

        sys.exit(0 if all(results.values()) else 1)

    elif args.model_a and args.model_b:
        passed = compare_tokenizers(args.model_a, args.model_b, verbose=args.verbose)
        sys.exit(0 if passed else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
