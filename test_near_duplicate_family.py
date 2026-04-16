import json

from data_pipeline.quality_scorer import TwoLayerQualityScorer


def _extraction_output(model: str, price: str, storage: str) -> str:
    payload = {
        "brand": "Apple",
        "category": "smartphone",
        "model": model,
        "price": price,
        "storage": storage,
        "display": "17.53 cm (6.9 inch) Super Retina XDR Display",
        "rear_camera": "48MP + 48MP + 48MP",
        "front_camera": "18MP",
        "processor": "A19 Chip",
        "warranty": "(1) Year Limited Warranty",
    }
    return json.dumps(payload, ensure_ascii=False)


def test_family_level_near_duplicate_filter():
    scorer = TwoLayerQualityScorer()

    # Same family (iPhone 17 Pro Max), extraction task + same intent -> keep one.
    sample_a = {
        "instruction": "Provide a structured JSON output of the features for Apple iPhone 17 Pro Max (Silver, 256 GB).",
        "input": (
            "Product: Apple iPhone 17 Pro Max (Silver, 256 GB)\n"
            "Price: INR 149,900.00\n"
            "Rating: 4.8/5 (552 reviews)\n\n"
            "Description:\n"
            "256 GB ROM\n"
            "17.53 cm (6.9 inch) Super Retina XDR Display\n"
            "48MP + 48MP + 48MP | 18MP Front Camera\n"
            "A19 Chip, 6 Core Processor Processor\n"
            "Apple One (1) Year Limited Warranty"
        ),
        "output": _extraction_output(
            "iPhone 17 Pro Max (Silver, 256 GB)", "INR 149,900.00", "256 GB ROM"
        ),
    }
    sample_b = {
        "instruction": "Provide a structured JSON output of the features for Apple iPhone 17 Pro Max (Deep Blue, 512 GB).",
        "input": (
            "Product: Apple iPhone 17 Pro Max (Deep Blue, 512 GB)\n"
            "Price: INR 169,900.00\n"
            "Rating: 4.8/5 (552 reviews)\n\n"
            "Description:\n"
            "512 GB ROM\n"
            "17.53 cm (6.9 inch) Super Retina XDR Display\n"
            "48MP + 48MP + 48MP | 18MP Front Camera\n"
            "A19 Chip, 6 Core Processor Processor\n"
            "Apple One (1) Year Limited Warranty"
        ),
        "output": _extraction_output(
            "iPhone 17 Pro Max (Deep Blue, 512 GB)", "INR 169,900.00", "512 GB ROM"
        ),
    }

    # Same family but different task/intent -> keep.
    sample_c = {
        "instruction": "What brand is Apple iPhone 17 Pro Max (Silver, 256 GB) and what category does it belong to?",
        "input": sample_a["input"],
        "output": "Brand: Apple\nCategory: smartphone",
    }

    # Different family, same extraction intent -> keep.
    sample_d = {
        "instruction": "Provide a structured JSON output of the features for Apple iPhone 16 (Teal, 256 GB).",
        "input": (
            "Product: Apple iPhone 16 (Teal, 256 GB)\n"
            "Price: INR 74,900.00\n"
            "Rating: 4.6/5 (213,340 reviews)\n\n"
            "Description:\n"
            "256 GB ROM\n"
            "15.49 cm (6.1 inch) Super Retina XDR Display\n"
            "48MP + 12MP | 12MP Front Camera\n"
            "A18 Chip, 6 Core Processor Processor\n"
            "1 year warranty for phone and 1 year warranty for in Box Accessories."
        ),
        "output": json.dumps(
            {
                "brand": "Apple",
                "category": "smartphone",
                "model": "iPhone 16 (Teal, 256 GB)",
                "price": "INR 74,900.00",
                "storage": "256 GB ROM",
                "display": "15.49 cm (6.1 inch) Super Retina XDR Display",
                "rear_camera": "48MP + 12MP",
                "front_camera": "12MP",
                "processor": "A18 Chip",
                "warranty": "1 year warranty for phone and 1 year warranty",
            },
            ensure_ascii=False,
        ),
    }

    passed = scorer.score_quality([sample_a, sample_b, sample_c, sample_d])

    # sample_b is removed as near-duplicate family extraction intent.
    assert len(passed) == 3
    instructions = [p["instruction"] for p in passed]
    assert sample_a["instruction"] in instructions
    assert sample_c["instruction"] in instructions
    assert sample_d["instruction"] in instructions
    assert sample_b["instruction"] not in instructions

    stats = scorer.get_stats()
    assert stats["rejected_near_duplicate_family"] == 1


def test_family_total_cap_limits_mixed_task_variants():
    scorer = TwoLayerQualityScorer()
    base_input = (
        "Product: Apple iPhone 17 Pro Max (Silver, 256 GB)\n"
        "Price: INR 149,900.00\n"
        "Rating: 4.8/5 (552 reviews)\n\n"
        "Description:\n"
        "256 GB ROM\n"
        "17.53 cm (6.9 inch) Super Retina XDR Display\n"
        "48MP + 48MP + 48MP | 18MP Front Camera\n"
        "A19 Chip, 6 Core Processor Processor\n"
        "Apple One (1) Year Limited Warranty"
    )

    extraction = {
        "instruction": "Provide a structured JSON output of the features for Apple iPhone 17 Pro Max (Silver, 256 GB).",
        "input": base_input,
        "output": _extraction_output(
            "iPhone 17 Pro Max (Silver, 256 GB)", "INR 149,900.00", "256 GB ROM"
        ),
    }
    qa = {
        "instruction": "What brand is Apple iPhone 17 Pro Max (Silver, 256 GB) and what category does it belong to?",
        "input": base_input,
        "output": "Brand: Apple\nCategory: smartphone",
    }
    summary = {
        "instruction": "Give me a quick 2-line summary of Apple iPhone 17 Pro Max (Silver, 256 GB).",
        "input": base_input,
        "output": "Apple iPhone 17 Pro Max (Silver, 256 GB) by Apple\nPrice: INR 149,900.00 | Rating: 4.8/5",
    }
    reasoning = {
        "instruction": "Think step-by-step and tell me if Apple iPhone 17 Pro Max (Silver, 256 GB) is a good value.",
        "input": base_input,
        "output": (
            "<thought>Storage is 256 GB ROM, which supports heavier usage. "
            "Processor is A19 Chip, suggesting performance potential. "
            "Rating is 4.8/5, indicating user satisfaction signals. "
            "Price is INR 149,900.00, relevant for value evaluation.</thought>\n"
            "Conclusion: Based on available evidence, this product appears to offer good value."
        ),
    }

    passed = scorer.score_quality([extraction, qa, summary, reasoning])
    assert len(passed) == 3
    instructions = [p["instruction"] for p in passed]
    assert extraction["instruction"] in instructions
    assert qa["instruction"] in instructions
    assert summary["instruction"] in instructions
    assert reasoning["instruction"] not in instructions

    stats = scorer.get_stats()
    assert stats["rejected"] >= 1
