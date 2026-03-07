#!/usr/bin/env python3
"""
Test: E-Commerce Instruction Output Quality
============================================
Verifies that the pipeline produces clean, reliable instruction-output
pairs from e-commerce product data — no promotional noise, no raw dumps.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "✅"
FAIL = "❌"
results = []


def report(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed))
    print(f"  {status} {name}")
    if detail:
        print(f"      {detail}")


# ─── Mock E-Commerce Data ────────────────────────────────────────────────
# Simulates what the scraper produces after scraping an Amazon product page

MOCK_PRODUCT_DATA = {
    "title": "Apple 2026 MacBook Neo 13\" Laptop with A18 Pro chip",
    "brand": "Apple",
    "price": 69900.0,
    "original_price": 79900.0,
    "discount": "12% off",
    "currency": "INR",
    "rating": 4.5,
    "reviews_count": 1234,
    "availability": "In Stock",
    "category": "Computers > Laptops",
    "description": "Built for AI and Apple Intelligence, Liquid Retina Display, 8GB Unified Memory, 256GB SSD Storage.",
    "features": [
        "A18 Pro chip: Built for AI and Apple Intelligence",
        "Liquid Retina Display with True Tone",
        "8GB Unified Memory",
        "256GB SSD Storage",
        "1080p FaceTime HD Camera",
        "Touch ID",
    ],
    "reviews": [
        "Amazing laptop, fast and lightweight!",
        "Great display quality, worth the price.",
        "Battery life could be better for the price point.",
    ],
    "url": "https://www.amazon.in/dp/B0EXAMPLE",
    "platform": "amazon",
    "seller": "Apple India",
}

MOCK_PRODUCT_TEXT = """Product: Apple 2026 MacBook Neo 13" Laptop with A18 Pro chip
Brand: Apple
Price: INR 69,900.00 (Original: INR 79,900.00) — 12% off
Rating: 4.5/5 (1,234 reviews)
Availability: In Stock
Category: Computers > Laptops

Description:
Built for AI and Apple Intelligence, Liquid Retina Display, 8GB Unified Memory, 256GB SSD Storage.

Key Features:
  • A18 Pro chip: Built for AI and Apple Intelligence
  • Liquid Retina Display with True Tone
  • 8GB Unified Memory
  • 256GB SSD Storage
  • 1080p FaceTime HD Camera
  • Touch ID

Customer Reviews:
  [1] Amazing laptop, fast and lightweight!
  [2] Great display quality, worth the price.
  [3] Battery life could be better for the price point."""

# Simulate raw scraped text WITH noise (what was happening before)
RAW_NOISY_TEXT = """Apple
2026 MacBook Neo 13″ Laptop with A18 Pro chip: Built for AI and Apple Intelligence, Liquid Retina Display, 8GB Unified Memory, 256GB SSD Storage, 1080p FaceTime HD Camera; Silver
Price, product page
₹69,900
₹
69,900
Up to 5% back with Amazon Pay ICICI card
FREE delivery
13 - 21 Mar
This item will be released on March 12, 2026. Apple
2026 MacBook Neo 13" Laptop"""


def test_ecommerce_text_cleaning():
    """Test that promotional noise is stripped from e-commerce text."""
    print("\n── Test: E-Commerce Text Cleaning ────────────────────────")
    from data_pipeline.instruct_formatter import InstructFormatter

    formatter = InstructFormatter(template="alpaca")

    cleaned = formatter._clean_ecommerce_text(RAW_NOISY_TEXT)

    # Should NOT contain promotional noise
    noise_patterns = [
        "Up to 5% back",
        "Amazon Pay ICICI",
        "FREE delivery",
        "13 - 21 Mar",
        "This item will be released",
        "Price, product page",
    ]

    all_clean = True
    for noise in noise_patterns:
        if noise.lower() in cleaned.lower():
            report(f"Noise removed: '{noise}'", False, f"Still found in output")
            all_clean = False
        else:
            report(f"Noise removed: '{noise}'", True)

    # Check duplicate price removal
    price_count = cleaned.count("69,900")
    report(
        "Duplicate prices removed",
        price_count <= 1,
        f"Found {price_count} occurrences of price",
    )

    report("Cleaned text > 10 chars", len(cleaned.strip()) > 10, f"Length: {len(cleaned)}")

    assert all_clean


def test_ecommerce_instruction_generation():
    """Test that e-commerce chunks produce structured instruction pairs."""
    print("\n── Test: E-Commerce Instruction Generation ───────────────")
    from data_pipeline.instruct_formatter import InstructFormatter

    formatter = InstructFormatter(template="alpaca")

    # Simulate an e-commerce chunk (as produced by the pipeline after chunking)
    ecommerce_chunk = {
        "text": MOCK_PRODUCT_TEXT,
        "source": "https://www.amazon.in/dp/B0EXAMPLE",
        "source_type": "ecommerce",
        "doc_id": "test123",
        "chunk_index": 0,
        "metadata": {"product_data": MOCK_PRODUCT_DATA},
    }

    pairs = formatter.format_chunks(
        [ecommerce_chunk], domain="ecommerce", generate_qa=True, pairs_per_chunk=3
    )

    report("Pairs generated", len(pairs) > 0, f"Got {len(pairs)} pairs")

    # Check that instructions are product-specific and answerable
    for i, pair in enumerate(pairs[:5]):
        instruction = pair.get("instruction", "")
        output = pair.get("output", "")

        # Instruction should reference the product or be product-aware
        is_product_aware = any(
            kw in instruction.lower()
            for kw in [
                "macbook",
                "apple",
                "specification",
                "feature",
                "price",
                "review",
                "summary",
                "brand",
                "pros",
                "category",
                "product",
                "display",
                "retina",
                "chip",
                "memory",
                "storage",
                "laptop",
                "speaker",
                "headphone",
                "intelligence",
                "description",
            ]
        )
        report(
            f"Pair {i+1}: Instruction is product-aware",
            is_product_aware,
            f'"{instruction[:80]}"',
        )

        # Output should NOT contain promotional noise
        noise_words = [
            "Amazon Pay ICICI",
            "FREE delivery",
            "product page",
            "Up to 5% back",
        ]
        has_noise = any(nw.lower() in output.lower() for nw in noise_words)
        report(
            f"Pair {i+1}: Output is clean (no promo noise)",
            not has_noise,
            f"Output length: {len(output)} chars",
        )

        # Output should have meaningful content
        report(
            f"Pair {i+1}: Output has substance",
            len(output) > 30,
            f"Output: {output[:100]}...",
        )

    assert len(pairs) > 0


def test_generic_chunks_unaffected():
    """Test that non-ecommerce chunks still use generic instruction generation."""
    print("\n── Test: Generic Chunks Unaffected ────────────────────────")
    from data_pipeline.instruct_formatter import InstructFormatter

    formatter = InstructFormatter(template="alpaca")

    generic_chunk = {
        "text": (
            "Machine learning is a subset of artificial intelligence that enables systems "
            "to learn from data. Deep learning uses neural networks with multiple layers. "
            "Transformers have revolutionized natural language processing since 2017. "
            "The attention mechanism allows models to focus on relevant parts of input."
        ),
        "source": "textbook.pdf",
        "source_type": "pdf",
        "doc_id": "generic123",
        "chunk_index": 0,
    }

    pairs = formatter.format_chunks(
        [generic_chunk], domain="AI", generate_qa=True, pairs_per_chunk=2
    )

    report("Generic pairs generated", len(pairs) > 0, f"Got {len(pairs)} pairs")

    # Check it uses generic templates (not e-commerce ones)
    for i, pair in enumerate(pairs[:3]):
        instruction = pair.get("instruction", "")
        is_generic = any(
            kw in instruction.lower()
            for kw in ["explain", "summarize", "main points", "overview", "analyze", "what is"]
        )
        report(
            f"Generic pair {i+1}: Uses generic template",
            is_generic,
            f'"{instruction[:80]}"',
        )

    assert len(pairs) > 0


def test_alpha_ratio_filter():
    """Test that noisy QA answers (mostly numbers/symbols) are filtered out."""
    print("\n── Test: Alpha Ratio Filter ───────────────────────────────")
    from data_pipeline.instruct_formatter import InstructFormatter

    formatter = InstructFormatter(template="alpaca")

    # Text with capitalized phrases but mostly numeric content
    noisy_text = "Product X123. ₹69,900 ₹79,900 ₹89,900 ₹59,900. Product X123 costs ₹69,900 originally ₹79,900."

    qa_pairs = formatter._extract_qa_pairs(noisy_text, "ecommerce")

    # Any QA pairs generated should have alpha ratio > 0.4
    all_clean = True
    for pair in qa_pairs:
        output = pair.get("output", "")
        alpha_count = sum(c.isalpha() for c in output)
        ratio = alpha_count / max(len(output), 1)
        if ratio < 0.4:
            report(f"QA output alpha ratio", False, f"Ratio: {ratio:.2f}")
            all_clean = False

    report("All QA outputs pass alpha ratio check", all_clean or len(qa_pairs) == 0)

    assert all_clean or len(qa_pairs) == 0


def main():
    print("\n" + "=" * 60)
    print("  E-COMMERCE INSTRUCTION QUALITY TESTS")
    print("=" * 60)

    test_ecommerce_text_cleaning()
    test_ecommerce_instruction_generation()
    test_generic_chunks_unaffected()
    test_alpha_ratio_filter()

    # Summary
    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
