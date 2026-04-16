#!/usr/bin/env python3
"""
dataset_verifier.py

Dataset verification and correction utility for instruction-tuning data.

For each sample:
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}

The script performs:
1. Instruction alignment
2. Grounding check (output facts must exist in input)
3. Missing information logic enforcement
4. Format validation by task type
5. JSON validation for extraction tasks
6. Automatic correction when possible

Output per sample:
{
  "status": "valid | corrected | invalid",
  "issues": [...],
  "corrected_output": "..."
}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


# ------------------------------
# Task classification
# ------------------------------

EXTRACTION_HINTS = [
    "extract",
    "json",
    "structured",
    "parse",
    "identify",
    "spec",
    "attribute",
]

REASONING_HINTS = [
    "reason",
    "step-by-step",
    "recommend",
    "why",
    "good for",
    "suitable",
    "explain",
]


# ------------------------------
# Product parser (extended)
# ------------------------------

def parse_product(text: str) -> Dict[str, str]:
    """Extract product attributes from context text via regex."""
    value = text or ""
    data: Dict[str, str] = {}

    name = re.search(r"Product:\s*(.*)", value, flags=re.IGNORECASE)
    price = re.search(r"Price:\s*(INR\s*[\d,\.]+)", value, flags=re.IGNORECASE)
    rating = re.search(r"Rating:\s*([\d\.]+/5)", value, flags=re.IGNORECASE)
    storage = re.search(r"(\d+\s*(GB|TB))\s*ROM", value, flags=re.IGNORECASE)
    display = re.search(r"(\d+\.?\d*\s*cm.*Display)", value, flags=re.IGNORECASE)
    chip = re.search(r"([A-Z]\d+\s*Chip|Snapdragon\s*[\w\s]+|Dimensity\s*[\w\s]+)", value, flags=re.IGNORECASE)
    warranty = re.search(r"(\d+\s*year.*warranty)", value, flags=re.IGNORECASE)

    if name:
        data["product_name"] = name.group(1).strip()
        # Optional brand extraction from product_name.
        brand_match = re.match(r"([A-Za-z]+)", data["product_name"])
        if brand_match:
            data["brand"] = brand_match.group(1)

    if price:
        data["price"] = price.group(1).strip()

    if rating:
        data["rating"] = rating.group(1).strip()

    if storage:
        data["storage"] = storage.group(1).strip()

    if display:
        data["display"] = display.group(1).strip()

    if chip:
        data["chip"] = chip.group(1).strip()

    if warranty:
        data["warranty"] = warranty.group(1).strip()

    return data


def normalize_for_match(text: str) -> str:
    """Normalize text for rough containment comparisons."""
    t = (text or "").lower()
    t = re.sub(r"[^\w\s/\.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def contains_fact(input_text: str, fact_value: str) -> bool:
    """Check if a fact appears grounded in input."""
    if not fact_value:
        return True
    i_norm = normalize_for_match(input_text)
    f_norm = normalize_for_match(fact_value)
    if not f_norm:
        return True

    # Direct contain
    if f_norm in i_norm:
        return True

    # Secondary numeric-only check for formatted currency like INR 74,900.00
    i_num = re.sub(r"[^\d\.]", "", i_norm)
    f_num = re.sub(r"[^\d\.]", "", f_norm)
    if f_num and f_num in i_num:
        return True

    return False


# ------------------------------
# Instruction semantics
# ------------------------------

FIELD_MAP = {
    "brand": ["brand"],
    "price": ["price", "cost"],
    "rating": ["rating", "review"],
    "storage": ["storage", "rom", "ssd"],
    "display": ["display", "screen"],
    "chip": ["chip", "processor", "soc"],
    "warranty": ["warranty"],
    "product_name": ["product", "name", "model"],
}


def detect_task_type(instruction: str) -> str:
    """Classify instruction into extraction | qa | reasoning."""
    inst = (instruction or "").lower()
    if any(h in inst for h in EXTRACTION_HINTS):
        return "extraction"
    if any(h in inst for h in REASONING_HINTS):
        return "reasoning"
    if "?" in inst or inst.startswith("what") or inst.startswith("tell me"):
        return "qa"
    # Fallback: treat unknown as QA.
    return "qa"


def requested_fields(instruction: str) -> Set[str]:
    """Infer requested fields from instruction text."""
    inst = (instruction or "").lower()
    inst = re.sub(r"\([^)]*\)", " ", inst)
    for sep in (" for ", " of ", " about ", " regarding "):
        if sep in inst:
            inst = inst.split(sep, 1)[0]
            break
    inst = re.sub(r"\s+", " ", inst).strip()
    req: Set[str] = set()
    for field, hints in FIELD_MAP.items():
        for h in hints:
            if re.search(r"\b" + re.escape(h) + r"\b", inst):
                req.add(field)
                break
    # Generic extraction prompts should include full parsed fields.
    if any(k in inst for k in ["spec", "key specifications", "attributes", "details"]):
        req.update({"product_name", "price", "rating", "storage", "display", "chip", "warranty"})
    return req


# ------------------------------
# Output parsing/validation
# ------------------------------

def parse_output_json(output: Any) -> Optional[Dict[str, Any]]:
    """Parse output as JSON object if possible."""
    if isinstance(output, dict):
        return output
    if not isinstance(output, str):
        return None

    raw = output.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    if not raw.startswith("{"):
        return None

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def output_specs_from_text(output_text: str) -> Dict[str, str]:
    """Extract structured-like facts from non-JSON output text."""
    return parse_product(output_text)


def has_missing_info_phrase(output_text: str) -> bool:
    """Check required missing-information phrase pattern."""
    pattern = r"The context does not provide information about"
    return bool(re.search(pattern, output_text or "", flags=re.IGNORECASE))


# ------------------------------
# Correction generators
# ------------------------------

def build_missing_info_message(fields: List[str]) -> str:
    """Build canonical missing-information sentence."""
    if not fields:
        return "The context does not provide this information."
    return f"The context does not provide information about {', '.join(fields)}."


def generate_extraction_output(specs: Dict[str, str], req_fields: Set[str]) -> str:
    """Generate corrected extraction JSON output."""
    if req_fields:
        available = {k: v for k, v in specs.items() if k in req_fields and v}
        missing = [f for f in sorted(req_fields) if f not in available]
    else:
        available = {k: v for k, v in specs.items() if v}
        missing = []

    payload: Dict[str, Any] = dict(available)
    if missing:
        payload["missing_information"] = build_missing_info_message(missing)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def generate_qa_output(specs: Dict[str, str], req_fields: Set[str]) -> str:
    """Generate corrected QA natural-language output."""
    if not req_fields:
        # If no specific field requested, return concise summary from available specs.
        if not specs:
            return "The context does not provide this information."
        lines = [f"{k.replace('_', ' ').title()}: {v}" for k, v in specs.items() if v]
        return "\n".join(lines[:4])

    lines = []
    missing = []
    for field in sorted(req_fields):
        val = specs.get(field, "")
        if val:
            lines.append(f"{field.replace('_', ' ').title()}: {val}")
        else:
            missing.append(field)

    if lines and not missing:
        return "\n".join(lines)
    if lines and missing:
        lines.append(build_missing_info_message(missing))
        return "\n".join(lines)
    return build_missing_info_message(missing or sorted(req_fields))


def generate_reasoning_output(specs: Dict[str, str]) -> str:
    """Generate corrected reasoning output with <thought> section."""
    thoughts = []
    if specs.get("storage"):
        thoughts.append(
            f"The device has {specs['storage']} storage, which supports heavier app and media usage."
        )
    if specs.get("chip"):
        thoughts.append(
            f"It uses {specs['chip']}, suggesting stronger performance for demanding tasks."
        )
    if specs.get("rating"):
        thoughts.append(
            f"The rating of {specs['rating']} indicates positive user feedback."
        )
    if not thoughts:
        thoughts.append("The available context is limited, so performance confidence is moderate.")

    return "<thought>" + " ".join(thoughts) + "</thought>\nRecommendation: Suitable for power users."


def correct_output(task: str, instruction: str, input_text: str) -> str:
    """Create corrected output following required format and grounding rules."""
    specs = parse_product(input_text)
    req_fields = requested_fields(instruction)

    if task == "extraction":
        return generate_extraction_output(specs, req_fields)
    if task == "reasoning":
        return generate_reasoning_output(specs)
    return generate_qa_output(specs, req_fields)


# ------------------------------
# Verifier core
# ------------------------------

def verify_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Verify a single sample and return verification result payload."""
    instruction = str(sample.get("instruction", "")).strip()
    input_text = str(sample.get("input", "")).strip()
    output_raw = sample.get("output", "")
    output_text = output_raw if isinstance(output_raw, str) else json.dumps(output_raw, ensure_ascii=False)

    issues: List[str] = []

    if not instruction:
        return {
            "status": "invalid",
            "issues": ["Missing instruction"],
            "corrected_output": "",
        }
    if not input_text:
        return {
            "status": "invalid",
            "issues": ["Missing input"],
            "corrected_output": "",
        }

    task = detect_task_type(instruction)
    req_fields = requested_fields(instruction)
    input_specs = parse_product(input_text)

    # 1) Format validation + 5) JSON validation
    output_json = parse_output_json(output_raw)
    if task == "extraction":
        if output_json is None:
            issues.append("Extraction task output is not valid JSON")
        else:
            # Key match sanity: requested fields should be represented if available.
            for f in req_fields:
                if f in input_specs and f not in output_json:
                    issues.append(f"JSON output missing requested field: {f}")
    elif task == "reasoning":
        if "<thought>" not in output_text or "</thought>" not in output_text:
            issues.append("Reasoning task output missing <thought> section")
    else:  # QA
        if output_json is not None:
            issues.append("QA task output should be natural language, not JSON")

    # 2) Grounding check
    if task == "extraction" and output_json is not None:
        for k, v in output_json.items():
            if k == "missing_information":
                continue
            if isinstance(v, (str, int, float)) and not contains_fact(input_text, str(v)):
                issues.append(f"Hallucination: output field '{k}' not grounded in input")
    else:
        out_specs = output_specs_from_text(output_text)
        for k, v in out_specs.items():
            if v and not contains_fact(input_text, v):
                issues.append(f"Hallucination: output field '{k}' not grounded in input")

    # 3) Missing information logic
    missing_requested = [f for f in sorted(req_fields) if f not in input_specs]
    if missing_requested:
        if not has_missing_info_phrase(output_text):
            issues.append(
                "Missing information logic violation: output should state missing context fields"
            )

    # 1) Instruction alignment (field-level)
    if req_fields:
        if task == "extraction" and output_json is not None:
            covered = set(output_json.keys())
            needed = {f for f in req_fields if f in input_specs}
            if not needed.issubset(covered):
                issues.append("Instruction/output mismatch: requested fields not fully covered")
        else:
            # For QA/reasoning, verify requested available fields are mentioned.
            needed = {f for f in req_fields if f in input_specs}
            if needed:
                lower_out = output_text.lower()
                for f in needed:
                    if f == "product_name":
                        val = input_specs.get("product_name", "")
                        if val and normalize_for_match(val) not in normalize_for_match(output_text):
                            issues.append("Instruction/output mismatch: product_name not addressed")
                    else:
                        val = input_specs.get(f, "")
                        if val and normalize_for_match(val) not in normalize_for_match(output_text):
                            issues.append(f"Instruction/output mismatch: '{f}' not addressed")

    if not issues:
        return {
            "status": "valid",
            "issues": [],
            "corrected_output": "",
        }

    # 6) Correction
    corrected = correct_output(task, instruction, input_text)
    if not corrected.strip():
        return {
            "status": "invalid",
            "issues": issues,
            "corrected_output": "",
        }

    return {
        "status": "corrected",
        "issues": issues,
        "corrected_output": corrected,
    }


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load JSON or JSONL dataset into list of records."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        raise ValueError("JSON file must be a list or {'data': [...]} structure")

    raise ValueError("Unsupported input format. Use .json or .jsonl")


def verify_dataset(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Verify all records and return report DataFrame."""
    report_rows = []
    for idx, sample in enumerate(records):
        result = verify_sample(sample)
        report_rows.append(
            {
                "index": idx,
                "status": result["status"],
                "issues": result["issues"],
                "corrected_output": result["corrected_output"],
            }
        )
    return pd.DataFrame(report_rows)


def build_corrected_dataset(records: List[Dict[str, Any]], report_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Create corrected dataset using valid and corrected samples only."""
    corrected_records: List[Dict[str, Any]] = []
    for i, row in report_df.iterrows():
        status = row["status"]
        if status == "invalid":
            continue

        src = dict(records[int(row["index"])])
        if status == "corrected":
            src["output"] = row["corrected_output"]
        corrected_records.append(src)
    return corrected_records


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    """Write records to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and correct LLM instruction-tuning dataset samples.")
    parser.add_argument("input_file", help="Path to dataset (.json or .jsonl)")
    parser.add_argument(
        "--report_file",
        default="verification_report.jsonl",
        help="Path to verification report JSONL",
    )
    parser.add_argument(
        "--corrected_file",
        default="verified_corrected_dataset.jsonl",
        help="Path to corrected dataset JSONL (valid + corrected only)",
    )
    args = parser.parse_args()

    records = load_dataset(args.input_file)
    report_df = verify_dataset(records)

    # Write per-sample verification report in requested schema.
    report_records = report_df[["status", "issues", "corrected_output"]].to_dict(orient="records")
    write_jsonl(report_records, args.report_file)

    corrected_dataset = build_corrected_dataset(records, report_df)
    write_jsonl(corrected_dataset, args.corrected_file)

    # Minimal summary
    print(f"total samples: {len(records)}")
    print(f"valid: {int((report_df['status'] == 'valid').sum())}")
    print(f"corrected: {int((report_df['status'] == 'corrected').sum())}")
    print(f"invalid: {int((report_df['status'] == 'invalid').sum())}")
    print(f"report written: {args.report_file}")
    print(f"corrected dataset written: {args.corrected_file}")


if __name__ == "__main__":
    main()
