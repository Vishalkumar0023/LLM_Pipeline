"""
Dataset Generator (Layer B)
===========================
Converts canonical evidence records into instruction-tuning samples.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Sequence, Tuple

from .instruction_templates import (
    EXTRACTION_INSTRUCTIONS,
    QA_INSTRUCTIONS,
    REASONING_INSTRUCTIONS,
    SUMMARIZATION_INSTRUCTIONS,
    TASK_DISTRIBUTION,
)


class DatasetGenerator:
    """Generate instruction/input/output samples from evidence records."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    @staticmethod
    def _norm(value: Any) -> str:
        text = "" if value is None else str(value)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _norm_input(value: Any) -> str:
        """Normalize input while preserving line boundaries for parser grounding."""
        text = "" if value is None else str(value)
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines).strip()

    def _allocate_tasks(self, n: int) -> List[str]:
        """Allocate task labels with exact target proportions as closely as possible."""
        if n <= 0:
            return []

        counts = {
            task: int(TASK_DISTRIBUTION[task] * n)
            for task in ["extraction", "qa", "summarization", "reasoning"]
        }
        assigned = sum(counts.values())
        remainder = n - assigned

        # Distribute remainder in deterministic order by largest fractional weight.
        order = ["extraction", "qa", "summarization", "reasoning"]
        for i in range(remainder):
            counts[order[i % len(order)]] += 1

        tasks: List[str] = []
        for task in order:
            tasks.extend([task] * counts[task])
        self.rng.shuffle(tasks)
        return tasks

    @staticmethod
    def _qa_intent_text(instruction: str) -> str:
        """Keep only the question-intent phrase; drop product variant noise."""
        inst = (instruction or "").lower()
        inst = re.sub(r"\([^)]*\)", " ", inst)
        for sep in (" for ", " of ", " about ", " regarding "):
            if sep in inst:
                inst = inst.split(sep, 1)[0]
                break
        inst = re.sub(r"\s+", " ", inst).strip()
        return inst

    @staticmethod
    def _requested_qa_fields(instruction: str) -> List[str]:
        inst = DatasetGenerator._qa_intent_text(instruction)
        mapping = {
            "price": "price",
            "warranty": "warranty",
            "storage": "storage",
            "rating": "rating",
            "display": "display",
            "chip": "chip",
            "name": "product_name",
            "brand": "brand",
            "rear": "camera_rear",
            "front": "camera_front",
            "camera": "camera_rear",
        }
        fields = []
        for token, field in mapping.items():
            if re.search(r"\b" + re.escape(token) + r"\b", inst) and field not in fields:
                fields.append(field)
        return fields

    @staticmethod
    def _infer_category(evidence: Dict[str, Any]) -> str:
        text = " ".join(
            [
                str(evidence.get("product_name") or ""),
                str(evidence.get("raw_text") or ""),
            ]
        ).lower()
        if any(k in text for k in ["iphone", "galaxy", "pixel", "oneplus", "smartphone", "mobile"]):
            return "smartphone"
        if any(k in text for k in ["macbook", "thinkpad", "inspiron", "laptop", "notebook", "vivobook", "pavilion"]):
            return "laptop"
        if any(k in text for k in ["tablet", "ipad"]):
            return "tablet"
        return "electronics"

    @staticmethod
    def _derive_model(evidence: Dict[str, Any]) -> str:
        product_name = str(evidence.get("product_name") or "").strip()
        brand = str(evidence.get("brand") or "").strip()
        if not product_name:
            return ""
        if brand and product_name.lower().startswith(brand.lower() + " "):
            return product_name[len(brand) :].strip()
        return product_name

    def _extraction_payload(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Build extraction output schema expected by fine-tune dataset consumers."""
        def _to_optional_text(v: Any) -> Any:
            if v in (None, ""):
                return None
            return str(v).strip() or None

        price = _to_optional_text(evidence.get("price"))
        price_inr = None
        if price:
            cleaned = re.sub(r"[^\d.]", "", str(price))
            if cleaned:
                try:
                    price_inr = int(round(float(cleaned)))
                except Exception:
                    price_inr = None

        rating = None
        if evidence.get("rating") not in (None, ""):
            try:
                rating = round(float(str(evidence.get("rating"))), 1)
            except Exception:
                rating = None

        review_count = evidence.get("review_count", evidence.get("reviews_count"))
        if review_count not in (None, ""):
            try:
                review_count = int(str(review_count).replace(",", ""))
            except Exception:
                review_count = None
        else:
            review_count = None

        return {
            "brand": _to_optional_text(evidence.get("brand")),
            "model": _to_optional_text(self._derive_model(evidence)),
            "category": _to_optional_text(self._infer_category(evidence)),
            "price": price,
            "price_inr": price_inr,
            "ram": _to_optional_text(evidence.get("ram") or evidence.get("unified_memory")),
            "storage": _to_optional_text(evidence.get("storage")),
            "processor": _to_optional_text(evidence.get("chip") or evidence.get("processor")),
            "display": _to_optional_text(evidence.get("display")),
            "os": _to_optional_text(evidence.get("os")),
            "rating": rating,
            "review_count": review_count,
            "rear_camera": _to_optional_text(evidence.get("camera_rear")),
            "front_camera": _to_optional_text(evidence.get("camera_front")),
            "warranty": _to_optional_text(evidence.get("warranty")),
        }

    def _make_extraction_sample(self, evidence: Dict[str, Any]) -> Tuple[str, str]:
        instruction = self.rng.choice(EXTRACTION_INSTRUCTIONS)
        output = json.dumps(self._extraction_payload(evidence), ensure_ascii=False)
        return instruction, output

    def _make_qa_sample(self, evidence: Dict[str, Any]) -> Tuple[str, str]:
        instruction = self.rng.choice(QA_INSTRUCTIONS)
        requested = self._requested_qa_fields(instruction)

        lines = []
        missing = []
        for field in requested:
            val = evidence.get(field)
            if val:
                pretty = field.replace("_", " ").title()
                lines.append(f"{pretty}: {self._norm(val)}")
            else:
                missing.append(field.replace("_", " "))

        if missing:
            lines.append(
                f"The context does not provide information about {', '.join(missing)}."
            )

        output = "\n".join(lines).strip() or "The context does not provide this information."
        return instruction, output

    def _make_summarization_sample(self, evidence: Dict[str, Any]) -> Tuple[str, str]:
        instruction = self.rng.choice(SUMMARIZATION_INSTRUCTIONS)

        parts = []
        if evidence.get("product_name"):
            parts.append(f"{self._norm(evidence['product_name'])}.")
        if evidence.get("brand"):
            parts.append(f"Brand: {self._norm(evidence['brand'])}.")
        if evidence.get("storage"):
            parts.append(f"Storage: {self._norm(evidence['storage'])}.")
        if evidence.get("display"):
            parts.append(f"Display: {self._norm(evidence['display'])}.")
        if evidence.get("chip"):
            parts.append(f"Chip: {self._norm(evidence['chip'])}.")
        if evidence.get("price"):
            parts.append(f"Price: {self._norm(evidence['price'])}.")
        if evidence.get("rating"):
            parts.append(f"Rating: {self._norm(evidence['rating'])}.")

        output = " ".join(parts).strip()
        if not output:
            output = "The context does not provide enough product details for summarization."
        return instruction, output

    def _make_reasoning_sample(self, evidence: Dict[str, Any]) -> Tuple[str, str]:
        instruction = self.rng.choice(REASONING_INSTRUCTIONS)

        thoughts = []
        if evidence.get("price"):
            thoughts.append(f"Price is {self._norm(evidence['price'])}.")
        if evidence.get("storage"):
            thoughts.append(f"Storage is {self._norm(evidence['storage'])}, which matters for heavy usage.")
        if evidence.get("chip"):
            thoughts.append(f"Chip is {self._norm(evidence['chip'])}, indicating performance capability.")
        if evidence.get("rating"):
            thoughts.append(f"Rating is {self._norm(evidence['rating'])}, reflecting user feedback.")
        if evidence.get("warranty"):
            thoughts.append(f"Warranty is {self._norm(evidence['warranty'])}, which helps long-term value.")

        if not thoughts:
            thoughts.append("Limited specifications are available, reducing confidence in value assessment.")

        # Deterministic conclusion rule.
        score = 0
        if evidence.get("storage"):
            score += 1
        if evidence.get("chip"):
            score += 1
        if evidence.get("rating"):
            score += 1
        if evidence.get("warranty"):
            score += 1
        if evidence.get("price"):
            score += 1
        recommendation = (
            "Recommendation: This appears to be good value based on the available evidence."
            if score >= 3
            else "Recommendation: Value is uncertain because evidence is incomplete."
        )

        output = "Analysis: " + " ".join(thoughts) + "\n" + recommendation
        return instruction, output

    def generate_instruction_pairs(self, evidence_records: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert evidence records into instruction-tuning samples."""
        records = list(evidence_records or [])
        if not records:
            return []

        tasks = self._allocate_tasks(len(records))
        dataset: List[Dict[str, str]] = []

        for evidence, task in zip(records, tasks):
            input_text = self._norm_input(evidence.get("raw_text", ""))
            if not input_text:
                # Keep contract strict: synthesis layer requires text context.
                continue

            if task == "extraction":
                instruction, output = self._make_extraction_sample(evidence)
            elif task == "qa":
                instruction, output = self._make_qa_sample(evidence)
            elif task == "summarization":
                instruction, output = self._make_summarization_sample(evidence)
            else:
                instruction, output = self._make_reasoning_sample(evidence)

            dataset.append(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                }
            )

        return dataset
