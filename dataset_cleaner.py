#!/usr/bin/env python3
"""
dataset_cleaner.py

Clean and normalize instruction-tuning datasets with schema:
{
  "instruction": "...",
  "input": "...",
  "output": "..." | {...}
}

Features:
1. Load JSON/JSONL files.
2. Validate and repair samples.
3. Normalize outputs to structured JSON.
4. Remove noisy scraped tokens.
5. Deduplicate records.
6. Export clean JSONL and print stats.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


STANDARD_INSTRUCTION = "Extract product specifications"
REQUIRED_OUTPUT_KEYS = ["brand", "model", "ram", "storage", "memory_type"]

# Exact noisy phrases requested by user + common scraped variants.
NOISE_PATTERNS = [
    r"\bBuy\s*Now\b",
    r"\bWithout\s*Charger\b",
    r"\bLimited\s*Offer\b",
    r"\bFree\s*Delivery\b",
    r"\|+",
]

BRAND_CANDIDATES = [
    "Apple",
    "Samsung",
    "Lenovo",
    "Dell",
    "HP",
    "Asus",
    "Acer",
    "MSI",
    "Sony",
    "LG",
    "Xiaomi",
    "OnePlus",
    "Realme",
    "Vivo",
    "Oppo",
]


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load JSON or JSONL into a pandas DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    suffix = path.suffix.lower()
    records: List[Dict[str, Any]] = []

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            records = data["data"]
        else:
            raise ValueError("JSON file must contain a list of records or {'data': [...]} structure.")
    else:
        raise ValueError("Unsupported file format. Use .json or .jsonl")

    df = pd.DataFrame(records)
    for col in ["instruction", "input", "output"]:
        if col not in df.columns:
            df[col] = ""
    return df[["instruction", "input", "output"]].copy()


def clean_text(text: Any) -> str:
    """Remove noisy tokens and normalize whitespace."""
    if text is None:
        return ""
    value = str(text)
    for pattern in NOISE_PATTERNS:
        value = re.sub(pattern, " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def maybe_parse_json_string(value: str) -> Optional[Dict[str, Any]]:
    """Parse JSON string output if possible and it resolves to dict."""
    if not value:
        return None

    raw = value.strip()
    # Remove markdown code fences if present.
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    if not (raw.startswith("{") and raw.endswith("}")):
        return None

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def extract_specs_from_text(text: str) -> Dict[str, str]:
    """Extract product spec fields from plain text using regex heuristics."""
    text = clean_text(text)
    specs = {k: "" for k in REQUIRED_OUTPUT_KEYS}
    if not text:
        return specs

    # Brand
    for brand in BRAND_CANDIDATES:
        if re.search(rf"\b{re.escape(brand)}\b", text, flags=re.IGNORECASE):
            specs["brand"] = brand
            break

    # Model (common laptop patterns first)
    model_patterns = [
        r"\b(MacBook\s+Air(?:\s+M\d+)?)\b",
        r"\b(MacBook\s+Pro(?:\s+M\d+)?)\b",
        r"\b(Inspiron\s+\d+)\b",
        r"\b(ThinkPad\s+[A-Za-z0-9]+)\b",
        r"\b(Pavilion\s+[A-Za-z0-9]+)\b",
        r"\b(Vivobook\s+[A-Za-z0-9]+)\b",
    ]
    for pattern in model_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            specs["model"] = match.group(1).strip()
            break

    # RAM and memory type
    ram_match = re.search(
        r"\b(\d+\s?(?:GB|TB))\s*(Unified\s*Memory|RAM|Memory)?\b",
        text,
        flags=re.IGNORECASE,
    )
    if ram_match:
        specs["ram"] = ram_match.group(1).upper().replace(" ", "")
        memory_label = (ram_match.group(2) or "").strip()
        if memory_label:
            if re.search(r"unified", memory_label, flags=re.IGNORECASE):
                specs["memory_type"] = "Unified Memory"
            elif re.search(r"ram|memory", memory_label, flags=re.IGNORECASE):
                specs["memory_type"] = "RAM"

    # Storage
    storage_match = re.search(
        r"\b(\d+\s?(?:GB|TB)\s*(?:SSD|HDD|Storage)?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if storage_match:
        raw_storage = storage_match.group(1).strip()
        # Avoid assigning same token as RAM when no storage keyword is present.
        if raw_storage.upper().replace(" ", "") != specs["ram"]:
            specs["storage"] = raw_storage.upper().replace(" STORAGE", "")

    # Fallback memory type inference.
    if not specs["memory_type"] and specs["ram"]:
        if re.search(r"unified\s*memory", text, flags=re.IGNORECASE):
            specs["memory_type"] = "Unified Memory"
        else:
            specs["memory_type"] = "RAM"

    return specs


def normalize_output_dict(output_obj: Dict[str, Any], input_text: str) -> Dict[str, str]:
    """Normalize arbitrary output dict to required schema."""
    normalized = {k: "" for k in REQUIRED_OUTPUT_KEYS}

    alias_map = {
        "brand": ["brand", "manufacturer", "make"],
        "model": ["model", "product_name", "product", "title", "device"],
        "ram": ["ram", "memory", "unified_memory"],
        "storage": ["storage", "ssd", "hdd", "disk"],
        "memory_type": ["memory_type", "memorytype", "ram_type"],
    }

    lower_map = {str(k).lower(): v for k, v in output_obj.items()}
    for target, aliases in alias_map.items():
        for alias in aliases:
            if alias in lower_map and lower_map[alias] is not None:
                normalized[target] = clean_text(lower_map[alias])
                break

    # Fill missing fields from input text using regex extraction.
    from_text = extract_specs_from_text(input_text)
    for key in REQUIRED_OUTPUT_KEYS:
        if not normalized[key] and from_text[key]:
            normalized[key] = from_text[key]

    # Cleanup repeated "off off" style errors if any leaked into fields.
    for key, val in normalized.items():
        val = re.sub(r"\b(off)(?:\s+off)+\b", r"\1", val, flags=re.IGNORECASE)
        normalized[key] = val.strip()

    return normalized


def output_has_spec_signal(output_dict: Dict[str, str]) -> bool:
    """Check if output has enough product-spec information to be useful."""
    filled = sum(1 for v in output_dict.values() if str(v).strip())
    return filled >= 2  # Keep records with at least two meaningful fields.


def instruction_matches_output(instruction: str, output_dict: Dict[str, str]) -> bool:
    """Heuristic intent check between instruction and output."""
    inst = instruction.lower().strip()
    has_spec_output = output_has_spec_signal(output_dict)
    if not inst:
        return False
    if any(k in inst for k in ["extract", "spec", "specification", "feature", "json", "product"]):
        return has_spec_output
    # Non-extraction prompts are treated as mismatch for this normalized dataset format.
    return False


def create_input_from_output(output_dict: Dict[str, str]) -> str:
    """Build a compact input sentence from output fields when input is missing."""
    parts = []
    if output_dict["brand"]:
        parts.append(output_dict["brand"])
    if output_dict["model"]:
        parts.append(output_dict["model"])
    if output_dict["ram"]:
        parts.append(output_dict["ram"])
    if output_dict["memory_type"]:
        parts.append(output_dict["memory_type"])
    if output_dict["storage"]:
        parts.append(output_dict["storage"])
    return ", ".join([p for p in parts if p])


def clean_record(row: pd.Series) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Validate and fix one sample.
    Returns (clean_record_or_none, was_fixed).
    """
    fixed = False
    instruction = clean_text(row.get("instruction", ""))
    input_text = clean_text(row.get("input", ""))
    output_raw = row.get("output", "")

    # Parse/normalize output into dict.
    output_dict: Dict[str, str]
    if isinstance(output_raw, dict):
        output_dict = normalize_output_dict(output_raw, input_text)
        fixed = True
    else:
        output_text = clean_text(output_raw)
        parsed_json = maybe_parse_json_string(output_text)
        if parsed_json is not None:
            output_dict = normalize_output_dict(parsed_json, input_text)
            fixed = True
        else:
            # Convert plain text output into structured spec dict.
            output_dict = normalize_output_dict({}, f"{input_text} {output_text}".strip())
            fixed = True

    # Fix empty input by deriving context from output.
    if not input_text:
        rebuilt_input = create_input_from_output(output_dict)
        if rebuilt_input:
            input_text = rebuilt_input
            fixed = True

    # If still empty input after repair, drop as invalid.
    if not input_text:
        return None, fixed

    # Enforce instruction/output intent consistency.
    if not instruction_matches_output(instruction, output_dict):
        if output_has_spec_signal(output_dict):
            instruction = STANDARD_INSTRUCTION
            fixed = True
        else:
            return None, fixed

    # Final output validity check.
    if not output_has_spec_signal(output_dict):
        return None, fixed

    # Ensure JSON serializable output.
    try:
        json.loads(json.dumps(output_dict, ensure_ascii=False))
    except (TypeError, ValueError):
        return None, fixed

    cleaned = {
        "instruction": instruction or STANDARD_INSTRUCTION,
        "input": input_text,
        "output": output_dict,
    }
    return cleaned, fixed


def deduplicate_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Deduplicate records using canonical instruction+input+output hash keys."""
    if not records:
        return records, 0

    df = pd.DataFrame(records)
    df["output_canonical"] = df["output"].apply(
        lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False)
    )
    before = len(df)
    df = df.drop_duplicates(subset=["instruction", "input", "output_canonical"]).copy()
    removed = before - len(df)
    return df[["instruction", "input", "output"]].to_dict(orient="records"), removed


def write_jsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    """Write list of dict records to JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def clean_dataset(input_file: str, output_file: str = "clean_dataset.jsonl") -> None:
    """Main cleaning pipeline."""
    df = load_dataset(input_file)
    total_samples = len(df)

    cleaned_records: List[Dict[str, Any]] = []
    fixed_samples = 0
    invalid_removed = 0

    for _, row in df.iterrows():
        cleaned, was_fixed = clean_record(row)
        if cleaned is None:
            invalid_removed += 1
            continue
        if was_fixed:
            fixed_samples += 1
        cleaned_records.append(cleaned)

    deduped_records, removed_duplicates = deduplicate_records(cleaned_records)
    write_jsonl(deduped_records, output_file)

    print(f"total samples: {total_samples}")
    print(f"removed duplicates: {removed_duplicates}")
    print(f"fixed samples: {fixed_samples}")
    print(f"invalid samples removed: {invalid_removed}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Clean and normalize LLM fine-tuning datasets.")
    parser.add_argument("input_file", help="Path to input .json or .jsonl dataset")
    parser.add_argument(
        "--output_file",
        default="clean_dataset.jsonl",
        help="Path to output cleaned JSONL file (default: clean_dataset.jsonl)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_dataset(args.input_file, args.output_file)
