import json

from data_pipeline.instruct_formatter import InstructFormatter


def test_subset_extraction_prompt_blanks_non_requested_fields():
    formatter = InstructFormatter(template="alpaca")
    structured = {
        "product_name": "Apple iPhone 17 Pro Max (Silver, 256 GB)",
        "brand": "Apple",
        "category": "smartphone",
        "price": "INR 149,900.00",
        "storage": "256 GB ROM",
        "display": "17.53 cm (6.9 inch) Super Retina XDR Display",
        "camera_rear": "48MP + 48MP + 48MP",
        "camera_front": "18MP",
        "processor": "A19 Chip",
        "warranty": "(1) Year Limited Warranty",
    }

    payload = formatter._structured_to_extraction_payload(structured)
    requested = formatter._requested_extraction_fields_from_instruction(
        "Extract brand, category, and price for Apple iPhone 17 Pro Max (Silver, 256 GB) if available."
    )
    subset = formatter._apply_extraction_subset(payload, requested)

    assert subset["brand"] == "Apple"
    assert subset["category"] == "smartphone"
    assert subset["price"] == "INR 149,900.00"
    assert subset["model"] == ""
    assert subset["storage"] == ""
    assert subset["display"] == ""
    assert subset["rear_camera"] == ""
    assert subset["front_camera"] == ""
    assert subset["processor"] == ""
    assert subset["warranty"] == ""


def test_full_extraction_prompt_keeps_full_schema_values():
    formatter = InstructFormatter(template="alpaca")
    structured = {
        "product_name": "Apple iPhone 16 (Ultramarine, 128 GB)",
        "brand": "Apple",
        "category": "smartphone",
        "price": "INR 64,900.00",
        "storage": "128 GB ROM",
        "display": "15.49 cm (6.1 inch) Super Retina XDR Display",
        "camera_rear": "48MP + 12MP",
        "camera_front": "12MP",
        "processor": "A18 Chip",
        "warranty": "1 year warranty for phone and 1 year warranty",
    }

    payload = formatter._structured_to_extraction_payload(structured)
    requested = formatter._requested_extraction_fields_from_instruction(
        "Provide a structured JSON output of the features for Apple iPhone 16 (Ultramarine, 128 GB)."
    )
    subset = formatter._apply_extraction_subset(payload, requested)
    out = json.dumps(subset, ensure_ascii=False)

    assert "\"brand\": \"Apple\"" in out
    assert "\"model\": \"iPhone 16 (Ultramarine, 128 GB)\"" in out
    assert "\"price\": \"INR 64,900.00\"" in out
    assert "\"storage\": \"128 GB ROM\"" in out
    assert "\"display\": \"15.49 cm (6.1 inch) Super Retina XDR Display\"" in out
    assert "\"rear_camera\": \"48MP + 12MP\"" in out
    assert "\"front_camera\": \"12MP\"" in out
    assert "\"processor\": \"A18 Chip\"" in out
    assert "\"warranty\": \"1 year warranty for phone and 1 year warranty\"" in out
