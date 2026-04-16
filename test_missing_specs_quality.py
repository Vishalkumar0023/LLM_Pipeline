from data_pipeline.quality_scorer import TwoLayerQualityScorer
from data_pipeline.verification_agent import DatasetVerificationAgent


IPHONE_17_PRO_MAX_INPUT = (
    "Product: Apple iPhone 17 Pro Max (Silver, 512 GB)\n"
    "Price: INR 169,900.00\n"
    "Rating: 4.8/5 (552 reviews)\n\n"
    "Description:\n"
    "512 GB ROM\n"
    "17.53 cm (6.9 inch) Super Retina XDR Display\n"
    "48MP + 48MP + 48MP | 18MP Front Camera\n"
    "A19 Chip, 6 Core Processor Processor\n"
    "Apple One (1) Year Limited Warranty"
)


def test_vague_missing_specs_is_corrected_and_rejected_by_scorer():
    sample = {
        "instruction": "What are the main specifications missing for Apple iPhone 17 Pro Max (Silver, 512 GB)?",
        "input": IPHONE_17_PRO_MAX_INPUT,
        "output": "The context does not provide information about several specifications.",
    }

    verifier = DatasetVerificationAgent()
    report = verifier.verify_sample(sample)
    assert report["status"] == "corrected"
    corrected = str(report["corrected_output"])
    assert "The context does not provide information about" in corrected
    assert "several specifications" not in corrected.lower()

    scorer = TwoLayerQualityScorer()
    passed = scorer.score_quality([sample])
    assert len(passed) == 0
    assert scorer.get_stats()["rejected_instruction_mismatch"] >= 1


def test_brand_category_qa_is_valid_without_missing_field_phrase():
    sample = {
        "instruction": "What brand is Apple iPhone 17 Pro Max (Silver, 512 GB) and what category does it belong to?",
        "input": IPHONE_17_PRO_MAX_INPUT,
        "output": "Brand: Apple\nCategory: smartphone",
    }

    verifier = DatasetVerificationAgent()
    report = verifier.verify_sample(sample)
    assert report["status"] == "valid"
    assert report["issues"] == []


def test_blank_processor_gets_corrected_when_a18_pro_present():
    sample = {
        "instruction": "Please extract all technical specifications for the item Apple iPhone 16 Pro (Black Titanium, 256 GB) as JSON.",
        "input": (
            "Product: Apple iPhone 16 Pro (Black Titanium, 256 GB)\n"
            "Price: INR 119,900.00\n"
            "Rating: 4.7/5 (29,869 reviews)\n\n"
            "Description:\n"
            "256 GB ROM\n"
            "16.0 cm (6.3 inch) Super Retina XDR Display\n"
            "48MP + 48MP + 12MP | 12MP Front Camera\n"
            "A18 Pro Chip, 6 Core Processor Processor\n"
            "1 year warranty for phone and 1 year warranty for in Box Accessories."
        ),
        "output": (
            "{\"brand\": \"Apple\", \"category\": \"smartphone\", "
            "\"model\": \"iPhone 16 Pro (Black Titanium, 256 GB)\", "
            "\"price\": \"INR 119,900.00\", \"storage\": \"256 GB ROM\", "
            "\"display\": \"16.0 cm (6.3 inch) Super Retina XDR Display\", "
            "\"rear_camera\": \"48MP + 48MP + 12MP\", \"front_camera\": \"12MP\", "
            "\"processor\": \"\", \"warranty\": \"1 year warranty for phone and 1 year warranty\"}"
        ),
    }
    verifier = DatasetVerificationAgent()
    report = verifier.verify_sample(sample)
    assert report["status"] == "corrected"
    corrected = str(report["corrected_output"])
    assert "\"processor\": \"A18 Pro Chip\"" in corrected


def test_subset_extraction_instruction_blanks_unrequested_schema_fields():
    sample = {
        "instruction": "Extract brand, category, and price for Apple iPhone 17 Pro Max (Silver, 256 GB) if available.",
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
        "output": (
            "{\"brand\": \"Apple\", \"category\": \"smartphone\", "
            "\"model\": \"iPhone 17 Pro Max (Silver, 256 GB)\", "
            "\"price\": \"INR 149,900.00\", \"storage\": \"256 GB ROM\", "
            "\"display\": \"17.53 cm (6.9 inch) Super Retina XDR Display\", "
            "\"rear_camera\": \"48MP + 48MP + 48MP\", \"front_camera\": \"18MP\", "
            "\"processor\": \"A19 Chip\", \"warranty\": \"(1) Year Limited Warranty\"}"
        ),
    }
    verifier = DatasetVerificationAgent()
    report = verifier.verify_sample(sample)
    assert report["status"] == "corrected"
    corrected = str(report["corrected_output"])
    assert "\"brand\": \"Apple\"" in corrected
    assert "\"category\": \"smartphone\"" in corrected
    assert "\"price\": \"INR 149,900.00\"" in corrected
    assert "\"model\": \"iPhone 17 Pro Max (Silver, 256 GB)\"" in corrected
    assert "\"storage\": \"256 GB ROM\"" in corrected
    assert "\"display\": \"17.53 cm (6.9 inch) Super Retina XDR Display\"" in corrected
    assert "\"rear_camera\": \"48MP + 48MP + 48MP\"" in corrected
    assert "\"front_camera\": \"18MP\"" in corrected
    assert "\"processor\": \"A19 Chip\"" in corrected
    assert "\"price_inr\": 149900" in corrected
    assert "\"rating\": 4.8" in corrected
    assert "\"review_count\": 552" in corrected
    assert "\"ram\": null" in corrected
    assert "\"os\": null" in corrected

    scorer = TwoLayerQualityScorer()
    passed = scorer.score_quality([sample])
    assert len(passed) == 1


def test_reasoning_camera_battery_prompt_requires_camera_and_battery_mention():
    sample = {
        "instruction": "Provide a reasoned evaluation of Apple iPhone 16 (Teal, 256 GB)'s camera and battery.",
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
        "output": (
            "<thought>Storage is 256 GB ROM, which supports heavier usage. "
            "Processor is A18 Chip, suggesting performance potential. "
            "Rating is 4.6/5, indicating user satisfaction signals. "
            "Price is INR 74,900.00, relevant for value evaluation.</thought>\n"
            "Conclusion: Based on available evidence, this product appears to offer good value."
        ),
    }
    verifier = DatasetVerificationAgent()
    report = verifier.verify_sample(sample)
    assert report["status"] == "corrected"
    corrected = str(report["corrected_output"]).lower()
    assert "analysis:" in corrected and "recommendation:" in corrected
    assert "camera" in corrected
    assert "battery" in corrected

    scorer = TwoLayerQualityScorer()
    passed = scorer.score_quality([sample])
    assert len(passed) == 0


def test_reasoning_pros_cons_prompt_requires_pros_cons_language():
    sample = {
        "instruction": "Compare the pros and cons of Apple iPhone 16 (Teal, 128 GB).",
        "input": (
            "Product: Apple iPhone 16 (Teal, 128 GB)\n"
            "Price: INR 64,900.00\n"
            "Rating: 4.6/5 (213,340 reviews)\n\n"
            "Description:\n"
            "128 GB ROM\n"
            "15.49 cm (6.1 inch) Super Retina XDR Display\n"
            "48MP + 12MP | 12MP Front Camera\n"
            "A18 Chip, 6 Core Processor Processor\n"
            "1 year warranty for phone and 1 year warranty for in Box Accessories."
        ),
        "output": (
            "<thought>Storage is 128 GB ROM, which supports heavier usage. "
            "Processor is A18 Chip, suggesting performance potential. "
            "Rating is 4.6/5, indicating user satisfaction signals. "
            "Price is INR 64,900.00, relevant for value evaluation.</thought>\n"
            "Conclusion: Based on available evidence, this product appears to offer good value."
        ),
    }

    verifier = DatasetVerificationAgent()
    report = verifier.verify_sample(sample)
    assert report["status"] == "corrected"
    corrected = str(report["corrected_output"]).lower()
    assert "pros:" in corrected
    assert "cons:" in corrected

    scorer = TwoLayerQualityScorer()
    passed = scorer.score_quality([sample])
    assert len(passed) == 0


def test_scorer_rejects_extra_storage_for_discount_instruction():
    sample = {
        "instruction": "Is there a discount available for Apple MacBook Pro (M5 Pro, 2026) M5 Pro - (24 GB/1 TB SSD/Tahoe) MGDN4HN/A?",
        "input": (
            "Product: Apple MacBook Pro (M5 Pro, 2026) M5 Pro - (24 GB/1 TB SSD/Tahoe) MGDN4HN/A\n"
            "Price: INR 249,900.00\n\n"
            "Description:\n"
            "Apple M5 Pro Processor\n"
            "24 GB Unified Memory RAM\n"
            "Mac OS Operating System\n"
            "1 TB SSD\n"
            "35.56 cm (14 inch) Display"
        ),
        "output": "Storage: 1 TB SSD\nThe context does not provide information about discount.",
    }

    scorer = TwoLayerQualityScorer()
    passed = scorer.score_quality([sample])
    assert len(passed) == 0
    assert scorer.get_stats()["rejected_instruction_mismatch"] >= 1
