from data_pipeline.evidence_builder import EvidenceBuilder
from data_pipeline.verification_agent import DatasetVerificationAgent
from data_pipeline.dataset_generator import DatasetGenerator


def test_evidence_builder_extracts_laptop_processor_and_memory():
    text = (
        "Product: Apple MacBook Air M4 - (24 GB/512 GB SSD/macOS Sequoia) MC6V4HN/A\n"
        "Price: INR 132,990.00\n"
        "Rating: 4.8/5 (5 reviews)\n\n"
        "Description:\n"
        "Apple M4 Processor\n"
        "24 GB Unified Memory RAM\n"
        "Mac OS Operating System\n"
        "512 GB SSD\n"
        "34.54 cm (13.6 Inch) Display"
    )

    evidence = EvidenceBuilder.parse_evidence_from_text(text)

    assert evidence["brand"] == "Apple"
    assert evidence["chip"] == "Apple M4 Processor"
    assert evidence["unified_memory"] == "24 GB Unified Memory RAM"
    assert evidence["ram"] == "24 GB Unified Memory RAM"


def test_evidence_builder_extracts_microsoft_brand_and_snapdragon_processor():
    text = (
        "Product: MICROSOFT Surface Pro 12\" with Type Cover Core 8 Snapdragon X Plus - "
        "(16 GB/512 GB SSD/Windows 11 Home)\n"
        "Price: INR 108,990.00\n\n"
        "Description:\n"
        "Snapdragon X Plus Processor\n"
        "16 GB LPDDR5X RAM\n"
        "64 bit Windows 11 Operating System\n"
        "512 GB SSD\n"
        "30.48 cm (12 inch) Touchscreen Display"
    )

    evidence = EvidenceBuilder.parse_evidence_from_text(text)

    assert evidence["brand"] == "Microsoft"
    assert evidence["chip"] == "Snapdragon X Plus Processor"


def test_verifier_corrects_discount_answer_with_extra_missing_fields():
    sample = {
        "instruction": "Is there a discount available for Apple iPhone 17 (White, 256 GB)?",
        "input": (
            "Product: Apple iPhone 17 (White, 256 GB)\n"
            "Price: INR 82,900.00\n"
            "Rating: 4.6/5 (6,250 reviews)\n\n"
            "Description:\n"
            "256 GB ROM\n"
            "16.0 cm (6.3 inch) Super Retina XDR Display\n"
            "48MP + 48MP | 18MP Front Camera\n"
            "A19 Chip, 6 Core Processor Processor\n"
            "Apple One (1) Year Limited Warranty"
        ),
        "output": "The context does not provide information about availability, discount.",
    }

    verifier = DatasetVerificationAgent()
    report = verifier.verify_sample(sample)

    assert report["status"] == "corrected"
    assert (
        str(report["corrected_output"]).strip()
        == "The context does not provide information about discount."
    )


def test_verifier_returns_ram_when_unified_memory_exists():
    sample = {
        "instruction": "Can you tell me the RAM and storage for Apple MacBook Air M4 - (24 GB/512 GB SSD/macOS Sequoia) MC6V4HN/A?",
        "input": (
            "Product: Apple MacBook Air M4 - (24 GB/512 GB SSD/macOS Sequoia) MC6V4HN/A\n"
            "Price: INR 132,990.00\n"
            "Rating: 4.8/5 (5 reviews)\n\n"
            "Description:\n"
            "Apple M4 Processor\n"
            "24 GB Unified Memory RAM\n"
            "Mac OS Operating System\n"
            "512 GB SSD\n"
            "34.54 cm (13.6 Inch) Display"
        ),
        "output": "Storage: 512 GB SSD\nThe context does not provide information about ram.",
    }

    verifier = DatasetVerificationAgent()
    report = verifier.verify_sample(sample)

    assert report["status"] == "corrected"
    corrected = str(report["corrected_output"])
    assert "Ram: 24 GB Unified Memory RAM" in corrected
    assert "Storage: 512 GB SSD" in corrected
    assert "does not provide information about ram" not in corrected.lower()


def test_verifier_treats_assess_quality_instruction_as_reasoning():
    sample = {
        "instruction": "Assess the overall quality of Apple Macbook Neo A18 Pro(2026) A18 Pro - (8 GB/512 GB SSD/Tahoe) MHFC4HN/A using the provided context.",
        "input": (
            "Product: Apple Macbook Neo A18 Pro(2026) A18 Pro - (8 GB/512 GB SSD/Tahoe) MHFC4HN/A\n"
            "Price: INR 79,900.00\n\n"
            "Description:\n"
            "Apple A18 Pro Processor\n"
            "8 GB Unified Memory RAM\n"
            "Mac OS Operating System\n"
            "512 GB SSD\n"
            "33.02 cm (13 inch) Display"
        ),
        "output": "Storage: 512 GB SSD",
    }

    verifier = DatasetVerificationAgent()
    report = verifier.verify_sample(sample)

    assert report["status"] == "corrected"
    corrected = str(report["corrected_output"])
    assert "Analysis:" in corrected
    assert "Recommendation:" in corrected


def test_verifier_rejects_extra_storage_for_discount_question_with_parenthetical_product():
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

    verifier = DatasetVerificationAgent()
    report = verifier.verify_sample(sample)

    assert report["status"] == "corrected"
    assert (
        str(report["corrected_output"]).strip()
        == "The context does not provide information about discount."
    )


def test_verifier_ignores_unparenthesized_storage_tokens_in_product_title():
    sample = {
        "instruction": "Is there a discount available for Apple MacBook Pro M3 8 GB/1 TB SSD macOS Sonoma MR7K3HN/A?",
        "input": (
            "Product: Apple MacBook Pro M3 8 GB/1 TB SSD macOS Sonoma MR7K3HN/A\n"
            "Price: INR 160,990.00\n"
            "Rating: 4.3/5 (76 reviews)\n\n"
            "Description:\n"
            "Apple M3 Processor\n"
            "8 GB Unified Memory RAM\n"
            "Mac OS Operating System\n"
            "1 TB SSD\n"
            "35.56 cm (14 Inch) Display"
        ),
        "output": "Storage: 1 TB SSD\nThe context does not provide information about discount.",
    }

    verifier = DatasetVerificationAgent()
    report = verifier.verify_sample(sample)

    assert report["status"] == "corrected"
    assert (
        str(report["corrected_output"]).strip()
        == "The context does not provide information about discount."
    )


def test_dataset_generator_requested_qa_fields_ignore_product_variant_noise():
    instruction = (
        "What is the price and warranty for Apple MacBook Pro M3 8 GB/1 TB SSD macOS Sonoma MR7K3HN/A?"
    )
    fields = DatasetGenerator._requested_qa_fields(instruction)
    assert "storage" not in fields
    assert fields == ["price", "warranty"]


def test_verifier_corrects_summary_missing_display_for_quick_summary_prompt():
    sample = {
        "instruction": "Give me a quick 2-line summary of Samsung Galaxy S26 Ultra 5G (Cobalt Violet, 256 GB).",
        "input": (
            "Product: Samsung Galaxy S26 Ultra 5G (Cobalt Violet, 256 GB)\n"
            "Price: INR 139,999.00\n"
            "Rating: 4.8/5 (31 reviews)\n\n"
            "Description:\n"
            "12 GB RAM | 256 GB ROM\n"
            "17.53 cm (6.9 inch) Quad HD+ Display\n"
            "200MP + 50MP | 12MP Front Camera\n"
            "5000 mAh Li-ion Battery\n"
            "Snapdragon 8 Elite Gen 5 Processor"
        ),
        "output": (
            "Samsung Galaxy S26 Ultra 5G (Cobalt Violet, 256 GB) by Samsung • 256 GB, 12MP Front Camera\n"
            "Price: INR 139,999.00 | Rating: 4.8/5"
        ),
    }

    verifier = DatasetVerificationAgent()
    report = verifier.verify_sample(sample)

    assert report["status"] == "corrected"
    corrected = str(report["corrected_output"])
    assert "17.53 cm (6.9 inch) Quad HD+ Display" in corrected
