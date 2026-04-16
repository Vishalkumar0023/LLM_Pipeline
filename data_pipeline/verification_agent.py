"""
Dataset Verification Agent
==========================
Validates instruction-tuning samples and auto-corrects invalid outputs using
only evidence grounded in the input context.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .evidence_builder import EvidenceBuilder


ALLOWED_EXTRACTION_KEYS = [
    "brand",
    "category",
    "model",
    "price",
    "price_inr",
    "ram",
    "storage",
    "processor",
    "display",
    "os",
    "rating",
    "review_count",
    "rear_camera",
    "front_camera",
    "warranty",
]


class DatasetVerificationAgent:
    """Validation + correction layer for generated instruction datasets."""

    def __init__(self):
        self._builder = EvidenceBuilder()

    @staticmethod
    def _task_type(instruction: str) -> str:
        inst = (instruction or "").lower()
        if any(
            k in inst
            for k in [
                "json",
                "extract",
                "parse",
                "structured",
                "machine-readable",
                "json object",
                "json dictionary",
                "product data",
            ]
        ):
            return "extraction"
        if any(
            k in inst
            for k in [
                "summarize",
                "summary",
                "summarise",
                "condense",
                "overview",
                "concise",
                "in a nutshell",
                "quick summary",
                "briefly describe",
            ]
        ):
            return "summarization"
        if any(
            k in inst
            for k in [
                "reason",
                "step-by-step",
                "evaluate",
                "assess",
                "overall quality",
                "pros and cons",
                "trade-off",
                "trade-offs",
                "tradeoff",
                "recommend",
                "power user",
                "good value",
                "why someone might buy",
                "explain why",
                "advantages and disadvantages",
                "break down",
            ]
        ):
            return "reasoning"
        return "qa"

    @staticmethod
    def _normalize(text: str) -> str:
        t = (text or "").lower()
        t = re.sub(r"[^\w\s/.\-+]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _has_hint(inst: str, hint: str) -> bool:
        """Match hint as a full term/phrase, not arbitrary substring."""
        phrase = (hint or "").strip().lower()
        if not phrase:
            return False
        pattern = r"\b" + r"\s+".join(re.escape(tok) for tok in phrase.split()) + r"\b"
        return re.search(pattern, inst) is not None

    @staticmethod
    def _intent_text(instruction: str) -> str:
        """
        Keep only intent-bearing phrase and drop product variant details.
        Example noise: '(16 GB/1 TB SSD/...)' or title fragments after 'for/of/about'.
        """
        inst = (instruction or "").lower()
        inst = re.sub(r"\([^)]*\)", " ", inst)
        for sep in (" for ", " of ", " about ", " regarding "):
            if sep in inst:
                inst = inst.split(sep, 1)[0]
                break
        inst = re.sub(r"\s+", " ", inst).strip()
        return inst

    def _normalized_field_name(self, field: str) -> str:
        return self._normalize((field or "").replace("_", " "))

    def _extract_missing_fields_from_output(self, text: str) -> List[str]:
        m = re.search(
            r"the context does not provide information about\s+([^.\n]+)",
            text or "",
            flags=re.IGNORECASE,
        )
        if not m:
            return []
        raw = m.group(1)
        parts = re.split(r",| and ", raw)
        fields = []
        for part in parts:
            p = self._normalize(part)
            if p:
                fields.append(p)
        return fields

    def _line_value_pairs(self, output: str) -> Dict[str, str]:
        pairs: Dict[str, str] = {}
        for line in (output or "").splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            pairs[self._normalize(key)] = val.strip()
        return pairs

    def _expand_missing_aliases(self, fields: Sequence[str]) -> Set[str]:
        expanded: Set[str] = set()
        for field in fields or []:
            f = self._normalize(field)
            if not f:
                continue
            expanded.add(f)
            if f == "camera":
                expanded.add("rear camera")
                expanded.add("front camera")
        return expanded

    def _requested_fields(self, instruction: str) -> Set[str]:
        inst = self._intent_text(instruction)
        mapping = {
            "model": ["product name", "name", "model"],
            "brand": ["brand"],
            "price": ["price", "cost"],
            "price_inr": ["price inr", "numeric price"],
            "rating": ["rating", "review"],
            "review_count": ["review count", "number of reviews"],
            "storage": ["storage", "rom", "ssd", "hdd"],
            "display": ["display", "screen"],
            "rear_camera": ["rear camera", "camera"],
            "front_camera": ["front camera", "selfie camera"],
            "processor": ["chip", "processor", "soc"],
            "os": ["operating system", "os"],
            "warranty": ["warranty"],
            "category": ["category", "type"],
            "discount": ["discount", "offer", "off"],
            "seller": ["seller"],
            "availability": ["availability", "delivery", "in stock", "out of stock"],
            "battery": ["battery"],
            "touch_id": ["touch id", "touchid"],
            "ram": ["ram"],
            "unified_memory": ["unified memory", "memory capacity"],
            "weight": ["weight"],
        }
        fields = set()
        for field, hints in mapping.items():
            for hint in hints:
                if self._has_hint(inst, hint):
                    fields.add(field)
                    break
        has_camera = self._has_hint(inst, "camera")
        has_rear = self._has_hint(inst, "rear camera")
        has_front = self._has_hint(inst, "front camera")
        if has_camera and not has_rear and not has_front:
            fields.add("rear_camera")
            fields.add("front_camera")
        if (
            "spec" in inst
            or "all available" in inst
            or "json" in inst
            or "structured" in inst
            or "machine-readable" in inst
            or "product info" in inst
        ):
            fields.update(ALLOWED_EXTRACTION_KEYS)
        return fields

    def _is_missing_specs_instruction(self, instruction: str) -> bool:
        inst = (instruction or "").lower()
        return self._has_hint(inst, "missing") and (
            self._has_hint(inst, "specifications")
            or self._has_hint(inst, "specification")
            or self._has_hint(inst, "specs")
        )

    def _parse_output_json(self, output: Any) -> Optional[Dict[str, Any]]:
        if isinstance(output, dict):
            return output
        raw = str(output or "").strip()
        if not raw:
            return None
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        if not raw.startswith("{"):
            return None
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def _grounded_value(self, input_text: str, expected: Any, candidate: Any) -> bool:
        if candidate in (None, ""):
            return True
        if expected not in (None, ""):
            exp = self._normalize(str(expected))
            cand = self._normalize(str(candidate))
            if exp == cand:
                return True
            if exp and cand and (exp in cand or cand in exp):
                return True
            exp_digits = re.sub(r"\D", "", str(expected))
            cand_digits = re.sub(r"\D", "", str(candidate))
            if exp_digits and cand_digits and (
                exp_digits == cand_digits
                or exp_digits in cand_digits
                or cand_digits in exp_digits
            ):
                return True
            return False
        # If no expected value, candidate must still be present in input text.
        return self._normalize(str(candidate)) in self._normalize(input_text)

    @staticmethod
    def _parse_price_inr(price_text: Any) -> Optional[int]:
        raw = str(price_text or "").strip()
        if not raw:
            return None
        cleaned = re.sub(r"[^0-9.]", "", raw)
        if not cleaned:
            return None
        try:
            return int(round(float(cleaned)))
        except ValueError:
            return None

    @staticmethod
    def _parse_rating_float(rating_text: Any) -> Optional[float]:
        raw = str(rating_text or "").strip()
        if not raw:
            return None
        m = re.search(r"([\d.]+)", raw)
        if not m:
            return None
        try:
            return round(float(m.group(1)), 1)
        except ValueError:
            return None

    def _infer_category(self, evidence: Dict[str, Any], input_text: str) -> str:
        text = " ".join(
            [
                str(evidence.get("product_name") or ""),
                str(input_text or ""),
            ]
        ).lower()
        if any(k in text for k in ["ipad", "tablet", "surface pro"]):
            return "tablet"
        if any(
            k in text
            for k in [
                "macbook",
                "thinkpad",
                "inspiron",
                "laptop",
                "notebook",
                "vivobook",
                "pavilion",
                "motobook",
                "galaxy book",
                "book4",
                "book 4",
                "intel core",
                "windows 11 operating system",
                "ssd",
            ]
        ):
            return "laptop"
        if any(k in text for k in ["iphone", "smartphone", "mobile", "pixel", "oneplus", "galaxy s", "galaxy z"]):
            return "smartphone"
        return "electronics"

    def _derive_model(self, evidence: Dict[str, Any]) -> Optional[str]:
        product_name = str(evidence.get("product_name") or "").strip()
        brand = str(evidence.get("brand") or "").strip()
        if not product_name:
            return None
        if brand and product_name.lower().startswith(brand.lower() + " "):
            product_name = product_name[len(brand) :].strip()
        product_name = re.sub(r"[.]{3,}$", "", product_name).strip()
        product_name = re.sub(r"[…]+$", "", product_name).strip()
        product_name = product_name.rstrip("- ").strip()
        return product_name or None

    def _expected_extraction_payload(self, evidence: Dict[str, Any], input_text: str) -> Dict[str, Any]:
        rating_float = self._parse_rating_float(evidence.get("rating"))
        return {
            "brand": evidence.get("brand"),
            "category": self._infer_category(evidence, input_text),
            "model": self._derive_model(evidence),
            "price": evidence.get("price"),
            "price_inr": self._parse_price_inr(evidence.get("price")),
            "ram": evidence.get("unified_memory") or evidence.get("ram"),
            "storage": evidence.get("storage"),
            "processor": evidence.get("chip"),
            "display": evidence.get("display"),
            "os": evidence.get("os"),
            "rating": rating_float,
            "review_count": evidence.get("review_count"),
            "rear_camera": evidence.get("camera_rear"),
            "front_camera": evidence.get("camera_front"),
            "warranty": evidence.get("warranty"),
        }

    def _build_extraction_output(self, evidence: Dict[str, Any], requested: Set[str]) -> str:
        # Enforce a stable full-schema JSON output with null fallbacks.
        # This keeps dataset rows consistent for downstream training consumers.
        base = self._expected_extraction_payload(evidence, evidence.get("raw_text", ""))
        req = set(requested or set())
        payload: Dict[str, Any] = {}
        for key in ALLOWED_EXTRACTION_KEYS:
            val = base.get(key)
            if val in ("",):
                val = None
            payload[key] = val
        return json.dumps(payload, ensure_ascii=False)

    def _missing_specs_lists(
        self, evidence: Dict[str, Any], input_text: str
    ) -> Tuple[List[str], List[str]]:
        extraction_payload = self._expected_extraction_payload(evidence, input_text)
        core_order = [
            "brand",
            "category",
            "model",
            "price",
            "price_inr",
            "ram",
            "storage",
            "processor",
            "display",
            "os",
            "rating",
            "review_count",
            "rear_camera",
            "front_camera",
            "warranty",
        ]
        core_missing = [f.replace("_", " ") for f in core_order if not extraction_payload.get(f)]

        optional_missing: List[str] = []
        if not evidence.get("ram") and not evidence.get("unified_memory"):
            optional_missing.append("ram")
        if not evidence.get("battery"):
            optional_missing.append("battery")
        if not evidence.get("touch_id"):
            optional_missing.append("touch id")
        if not evidence.get("seller"):
            optional_missing.append("seller")
        if not evidence.get("availability"):
            optional_missing.append("availability")
        if not evidence.get("discount"):
            optional_missing.append("discount")
        if not evidence.get("weight"):
            optional_missing.append("weight")
        return core_missing, optional_missing

    def _build_qa_output(
        self, evidence: Dict[str, Any], requested: Set[str], instruction: str = ""
    ) -> str:
        extraction_payload = self._expected_extraction_payload(
            evidence, evidence.get("raw_text", "")
        )
        inst = (instruction or "").lower()
        has_missing_specs = self._is_missing_specs_instruction(instruction)
        if not requested and has_missing_specs:
            requested = {
                "brand",
                "category",
                "model",
                "price",
                "storage",
                "display",
                "rear_camera",
                "front_camera",
                "processor",
                "warranty",
            }
        if has_missing_specs:
            missing_core, optional_missing = self._missing_specs_lists(
                evidence, evidence.get("raw_text", "")
            )
            if missing_core:
                return (
                    "The context does not provide information about "
                    + ", ".join(missing_core)
                    + "."
                )
            if optional_missing:
                return (
                    "The context does not provide information about "
                    + ", ".join(optional_missing)
                    + "."
                )
            return "The context provides the main specifications available in the input."
        if not requested:
            requested = {"price", "rating"}

        lines: List[str] = []
        missing: List[str] = []
        for field in sorted(requested):
            if field not in ALLOWED_EXTRACTION_KEYS:
                val = evidence.get(field)
            else:
                val = extraction_payload.get(field)
            if val:
                lines.append(f"{field.replace('_', ' ').title()}: {val}")
            else:
                missing.append(field.replace("_", " "))
        if missing:
            lines.append(
                f"The context does not provide information about {', '.join(missing)}."
            )
        return "\n".join(lines) if lines else "The context does not provide this information."

    def _build_summary_output(self, evidence: Dict[str, Any]) -> str:
        extraction_payload = self._expected_extraction_payload(
            evidence, evidence.get("raw_text", "")
        )
        model = str(extraction_payload.get("model") or "").strip()
        brand = str(extraction_payload.get("brand") or "").strip()
        ram = str(extraction_payload.get("ram") or "").strip()
        storage = str(extraction_payload.get("storage") or "").strip()
        processor = str(extraction_payload.get("processor") or "").strip()
        display = str(extraction_payload.get("display") or "").strip()
        os_name = str(extraction_payload.get("os") or "").strip()
        price = str(extraction_payload.get("price") or "").strip()
        rating = extraction_payload.get("rating")

        if not any([model, brand, ram, storage, processor, display, os_name, price, rating]):
            return "The context does not provide enough product information for summary."

        headline = model or "This product"
        if brand:
            headline += f" by {brand}"
        if ram:
            headline += f" • {ram}"
        if storage:
            headline += f" • {storage}"

        second_parts: List[str] = []
        if processor:
            second_parts.append(processor)
        if display:
            second_parts.append(display)
        if os_name:
            second_parts.append(os_name)
        if price:
            second_parts.append(f"Price: {price}")
        if rating not in (None, ""):
            second_parts.append(f"Rating: {rating}/5")

        if second_parts:
            return headline + "\n" + " | ".join(second_parts)
        return headline

    def _build_reasoning_output(self, evidence: Dict[str, Any], instruction: str = "") -> str:
        extraction_payload = self._expected_extraction_payload(
            evidence, evidence.get("raw_text", "")
        )
        requested = self._requested_fields(instruction)
        inst = (instruction or "").lower()
        thoughts: List[str] = []

        asks_camera = (
            "rear_camera" in requested
            or "front_camera" in requested
            or self._has_hint(inst, "camera")
        )
        asks_battery = "battery" in requested or self._has_hint(inst, "battery")
        asks_pros_cons = (
            self._has_hint(inst, "pros and cons")
            or self._has_hint(inst, "trade-off")
            or self._has_hint(inst, "trade-offs")
            or self._has_hint(inst, "tradeoff")
            or self._has_hint(inst, "trade offs")
            or self._has_hint(inst, "advantages and disadvantages")
        )

        if asks_camera:
            rear = extraction_payload.get("rear_camera")
            front = extraction_payload.get("front_camera")
            if rear and front:
                thoughts.append(f"Camera setup is {rear} rear and {front} front.")
            elif rear:
                thoughts.append(f"Rear camera is {rear}.")
            elif front:
                thoughts.append(f"Front camera is {front}.")
            else:
                thoughts.append("The context does not provide information about camera.")

        if asks_battery:
            battery = evidence.get("battery")
            if battery:
                thoughts.append(f"Battery is {battery}, which affects endurance.")
            else:
                thoughts.append("The context does not provide information about battery.")

        if asks_pros_cons:
            pros: List[str] = []
            cons: List[str] = []
            if extraction_payload.get("storage"):
                pros.append(f"storage is {extraction_payload['storage']}")
            if extraction_payload.get("processor"):
                pros.append(f"processor is {extraction_payload['processor']}")
            if evidence.get("rating"):
                pros.append(f"rating is {evidence['rating']}")
            if extraction_payload.get("display"):
                pros.append("display specs are clearly provided")

            if not evidence.get("battery"):
                cons.append("battery details are missing")
            if not evidence.get("discount"):
                cons.append("discount information is missing")
            if extraction_payload.get("price"):
                cons.append(f"price is {extraction_payload['price']}")

            if pros:
                thoughts.append("Pros: " + "; ".join(pros) + ".")
            if cons:
                thoughts.append("Cons: " + "; ".join(cons) + ".")

        if extraction_payload.get("storage"):
            thoughts.append(
                f"Storage is {extraction_payload['storage']}, which supports heavier usage."
            )
        if extraction_payload.get("processor"):
            thoughts.append(
                f"Processor is {extraction_payload['processor']}, suggesting performance potential."
            )
        if evidence.get("rating"):
            thoughts.append(f"Rating is {evidence['rating']}, indicating user satisfaction signals.")
        if extraction_payload.get("price"):
            thoughts.append(f"Price is {extraction_payload['price']}, relevant for value evaluation.")

        if not thoughts:
            thoughts.append("Available evidence is limited, so confidence is moderate.")

        score = sum(
            1
            for k in ["price", "storage", "processor", "warranty"]
            if extraction_payload.get(k)
        ) + (1 if extraction_payload.get("rating") else 0)
        if score >= 3:
            conclusion = "Recommendation: Based on available evidence, this product appears to offer good value."
        else:
            conclusion = "Recommendation: Value is uncertain due to missing evidence."
        return "Analysis: " + " ".join(thoughts) + "\n" + conclusion

    def _validate_extraction(
        self,
        sample: Dict[str, str],
        evidence: Dict[str, Any],
        issues: List[str],
    ) -> None:
        output_json = self._parse_output_json(sample["output"])
        if output_json is None:
            issues.append("invalid_json")
            return

        extra_keys = [k for k in output_json.keys() if k not in ALLOWED_EXTRACTION_KEYS]
        if extra_keys:
            issues.append("hallucinated_fields")
        missing_schema = [k for k in ALLOWED_EXTRACTION_KEYS if k not in output_json]
        if missing_schema:
            issues.append("schema_mismatch")

        requested = self._requested_fields(sample["instruction"])
        requested = {f for f in requested if f in ALLOWED_EXTRACTION_KEYS}
        if requested and not requested.issubset(set(output_json.keys())):
            issues.append("instruction_mismatch")

        expected_payload = self._expected_extraction_payload(evidence, sample["input"])
        for key, value in output_json.items():
            expected = expected_payload.get(key) if key in ALLOWED_EXTRACTION_KEYS else None
            if expected not in (None, "") and value in (None, ""):
                issues.append("instruction_mismatch")
                break
            if not self._grounded_value(sample["input"], expected, value):
                issues.append("not_grounded")
                break

        model_val = str(output_json.get("model") or "")
        if "..." in model_val or "…" in model_val:
            issues.append("instruction_mismatch")

    def _validate_qa(
        self,
        sample: Dict[str, str],
        evidence: Dict[str, Any],
        issues: List[str],
    ) -> None:
        requested = self._requested_fields(sample["instruction"])
        text = sample["output"]
        pairs = self._line_value_pairs(text)
        if not text.strip():
            issues.append("empty_output")
            return

        # QA tasks should be natural language, not JSON blobs.
        if self._parse_output_json(sample.get("output")) is not None:
            issues.append("qa_json_output")

        expected_payload = self._expected_extraction_payload(evidence, sample["input"])
        missing_requested = []
        for field in requested:
            val = expected_payload.get(field) if field in ALLOWED_EXTRACTION_KEYS else evidence.get(field)
            if not val:
                missing_requested.append(field)
        missing_phrase_present = "The context does not provide information about" in text
        if missing_requested and not missing_phrase_present:
            issues.append("missing_field_logic")

        listed_missing = self._extract_missing_fields_from_output(text)
        if listed_missing and not self._is_missing_specs_instruction(sample.get("instruction", "")):
            requested_norm = {self._normalized_field_name(f) for f in requested}
            listed_set = self._expand_missing_aliases(listed_missing)
            if any(field not in requested_norm for field in listed_set):
                issues.append("instruction_mismatch")
            if missing_requested:
                expected_missing_norm = self._expand_missing_aliases(
                    [self._normalized_field_name(f) for f in missing_requested]
                )
                if not expected_missing_norm.issubset(listed_set):
                    issues.append("missing_field_logic")

        # Each requested available value should appear.
        is_missing_specs = self._is_missing_specs_instruction(sample.get("instruction", ""))
        if requested and pairs and not is_missing_specs:
            requested_norm = {self._normalized_field_name(f) for f in requested}
            for key in pairs.keys():
                if key not in requested_norm:
                    issues.append("instruction_mismatch")
                    break

        if is_missing_specs:
            low = text.lower()
            if "several specifications" in low or "missing specifications" in low:
                issues.append("missing_field_logic")
            core_missing, optional_missing = self._missing_specs_lists(evidence, sample["input"])
            expected_missing = core_missing or optional_missing
            if expected_missing:
                if "the context does not provide information about" not in low:
                    issues.append("missing_field_logic")
                else:
                    listed_set = self._expand_missing_aliases(
                        self._extract_missing_fields_from_output(text)
                    )
                    expected_set = self._expand_missing_aliases(expected_missing)
                    if not expected_set.issubset(listed_set):
                        issues.append("missing_field_logic")
                    if listed_set and any(field not in expected_set for field in listed_set):
                        issues.append("instruction_mismatch")
            elif "the context does not provide information about" in low:
                issues.append("instruction_mismatch")
        else:
            for field in requested:
                val = expected_payload.get(field) if field in expected_payload else evidence.get(field)
                if val and self._normalize(str(val)) not in self._normalize(text):
                    issues.append("instruction_mismatch")
                    break

        # Disallow hallucinated facts not in input using simple grounding by lines.
        for line in text.splitlines():
            if ":" in line:
                lhs, rhs = line.split(":", 1)
                candidate = rhs.strip()
                expected = None
                lhs_norm = self._normalize(lhs)
                for field in ALLOWED_EXTRACTION_KEYS:
                    if lhs_norm == self._normalize(field.replace("_", " ")):
                        expected = expected_payload.get(field)
                        break
                if candidate and not self._grounded_value(sample["input"], expected, candidate):
                    issues.append("not_grounded")
                    break

    def _validate_summarization(
        self,
        sample: Dict[str, str],
        evidence: Dict[str, Any],
        issues: List[str],
    ) -> None:
        text = sample["output"].strip()
        if not text:
            issues.append("empty_output")
            return
        # Summary must reference at least one evidence field value.
        expected_payload = self._expected_extraction_payload(evidence, sample["input"])
        evidence_vals = [str(v) for _, v in expected_payload.items() if v]
        if evidence_vals:
            if not any(self._normalize(v) in self._normalize(text) for v in evidence_vals):
                issues.append("instruction_mismatch")
        inst = (sample.get("instruction") or "").lower()
        if any(k in inst for k in ["key features", "selling points", "brief overview", "summary", "summarize"]):
            tech_vals = [
                expected_payload.get("storage"),
                expected_payload.get("display"),
                expected_payload.get("processor"),
                expected_payload.get("rear_camera"),
                expected_payload.get("front_camera"),
            ]
            tech_vals = [str(v) for v in tech_vals if v]
            if tech_vals and not any(self._normalize(v) in self._normalize(text) for v in tech_vals):
                issues.append("instruction_mismatch")
            # Stronger check for concise product summaries: if display is known, include it.
            display_val = expected_payload.get("display")
            if display_val and self._normalize(str(display_val)) not in self._normalize(text):
                issues.append("instruction_mismatch")

    def _validate_reasoning(
        self,
        sample: Dict[str, str],
        evidence: Dict[str, Any],
        issues: List[str],
    ) -> None:
        text = sample["output"]
        if "analysis:" not in text.lower():
            issues.append("missing_analysis_section")
        if "recommendation:" not in text.lower():
            issues.append("missing_conclusion")

        inst = (sample.get("instruction") or "").lower()
        requested = self._requested_fields(sample.get("instruction", ""))
        text_low = text.lower()

        asks_camera = (
            "rear_camera" in requested
            or "front_camera" in requested
            or self._has_hint(inst, "camera")
        )
        asks_battery = "battery" in requested or self._has_hint(inst, "battery")
        asks_pros_cons = (
            self._has_hint(inst, "pros and cons")
            or self._has_hint(inst, "trade-off")
            or self._has_hint(inst, "trade-offs")
            or self._has_hint(inst, "tradeoff")
            or self._has_hint(inst, "trade offs")
            or self._has_hint(inst, "advantages and disadvantages")
        )

        if asks_camera and not any(tok in text_low for tok in ["camera", "rear", "front"]):
            issues.append("instruction_mismatch")
        if asks_battery and ("battery" not in text_low):
            issues.append("instruction_mismatch")
        if asks_pros_cons and not any(tok in text_low for tok in ["pros", "cons", "advantage", "disadvantage", "trade-off", "tradeoff"]):
            issues.append("instruction_mismatch")
        if self._has_hint(inst, "power user"):
            if not any(tok in text_low for tok in ["storage", "processor", "chip"]):
                issues.append("instruction_mismatch")

        expected_payload = self._expected_extraction_payload(evidence, sample["input"])
        evidence_vals = [str(v) for _, v in expected_payload.items() if v]
        if evidence_vals and not any(self._normalize(v) in self._normalize(text) for v in evidence_vals):
            issues.append("not_grounded")

    def _correct_output(self, sample: Dict[str, str], evidence: Dict[str, Any], task: str) -> Any:
        requested = self._requested_fields(sample["instruction"])
        if task == "extraction":
            return self._build_extraction_output(evidence, requested)
        if task == "qa":
            return self._build_qa_output(evidence, requested, sample["instruction"])
        if task == "summarization":
            return self._build_summary_output(evidence)
        return self._build_reasoning_output(evidence, sample["instruction"])

    def verify_sample(self, sample: Dict[str, str]) -> Dict[str, Any]:
        """
        Verify one sample and return:
        {
          "status": "valid|corrected|invalid",
          "issues": [...],
          "corrected_output": "..."
        }
        """
        instruction = (sample.get("instruction") or "").strip()
        input_text = (sample.get("input") or "").strip()
        out_val = sample.get("output")
        output_text = (
            json.dumps(out_val, ensure_ascii=False)
            if isinstance(out_val, dict)
            else str(out_val or "").strip()
        )

        if not instruction or not input_text:
            return {
                "status": "invalid",
                "issues": ["missing_instruction_or_input"],
                "corrected_output": "",
            }

        task = self._task_type(instruction)
        evidence = self._builder.parse_evidence_from_text(input_text)
        issues: List[str] = []

        if not output_text:
            issues.append("empty_output")

        if task == "extraction":
            self._validate_extraction(sample, evidence, issues)
        elif task == "qa":
            self._validate_qa(sample, evidence, issues)
        elif task == "summarization":
            self._validate_summarization(sample, evidence, issues)
        else:
            self._validate_reasoning(sample, evidence, issues)

        # Deduplicate issue codes while preserving order.
        seen = set()
        unique_issues = []
        for issue in issues:
            if issue not in seen:
                seen.add(issue)
                unique_issues.append(issue)

        if not unique_issues:
            return {"status": "valid", "issues": [], "corrected_output": ""}

        corrected = self._correct_output(sample, evidence, task)
        if not corrected:
            return {"status": "invalid", "issues": unique_issues, "corrected_output": ""}
        return {"status": "corrected", "issues": unique_issues, "corrected_output": corrected}

    def validate_and_correct(
        self, samples: Sequence[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """
        Validate/correct all samples.

        Returns:
        - corrected_samples (invalid removed)
        - verification_reports (status/issues/corrected_output)
        """
        verified_samples: List[Dict[str, str]] = []
        reports: List[Dict[str, Any]] = []

        for sample in samples or []:
            report = self.verify_sample(sample)
            reports.append(report)

            if report["status"] == "invalid":
                continue
            if report["status"] == "corrected":
                fixed = dict(sample)
                fixed["output"] = report["corrected_output"]
                if isinstance(fixed.get("output"), (dict, list)):
                    fixed["output"] = json.dumps(fixed["output"], ensure_ascii=False)
                verified_samples.append(fixed)
            else:
                fixed = dict(sample)
                if isinstance(fixed.get("output"), (dict, list)):
                    fixed["output"] = json.dumps(fixed["output"], ensure_ascii=False)
                verified_samples.append(fixed)

        return verified_samples, reports
