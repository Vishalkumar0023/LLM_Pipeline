import json
import re
from hashlib import md5
from random import shuffle
from typing import Any, Dict, List, Optional

from .evidence_builder import EvidenceBuilder


class LLMDataProcessor:
    """
    Deterministic data processor for instruction dataset generation.

    Pipeline role:
    - extract grounded evidence from raw text
    - normalize into canonical fields
    - generate diverse instruction tasks without external LLM calls
    - apply lightweight local verification before global validation layer
    """

    EXTRACTION_INSTRUCTIONS = [
        "Extract the product specifications in JSON.",
        "Parse the given text and return a JSON containing product specs.",
        "Provide a structured JSON output of the product features.",
        "Return the product data in a machine-readable JSON format.",
    ]
    QA_PRICE_INSTRUCTIONS = [
        "What is the price of this product?",
        "How much does this product cost?",
        "Tell me the current listed price for this item.",
    ]
    QA_RAM_STORAGE_INSTRUCTIONS = [
        "Can you tell me the RAM and storage for this product?",
        "What are the memory and storage specifications?",
    ]
    SUMMARY_INSTRUCTIONS = [
        "Summarize the key features of this product.",
        "Give a concise 2-line summary of this product.",
        "Provide a brief overview of the key specs and price.",
    ]
    REASONING_INSTRUCTIONS = [
        "Based on the specs, explain why someone might buy this product.",
        "Evaluate whether this product looks like good value from the provided details.",
    ]

    def __init__(self, client=None, model=None):
        # Kept for backward compatibility with existing route signatures.
        self._builder = EvidenceBuilder()
        self._stats = {
            "processed": 0,
            "evidence_extracted": 0,
            "tasks_generated": 0,
            "validated_valid": 0,
            "validated_invalid": 0,
            "errors": 0,
        }

    def process_raw_text(self, raw_text: str) -> List[Dict[str, Any]]:
        """Run local stages (extract -> normalize -> generate -> verify)."""
        self._stats["processed"] += 1
        valid_samples: List[Dict[str, Any]] = []

        try:
            evidence = self.extract_evidence(raw_text)
            if not evidence:
                return []
            self._stats["evidence_extracted"] += 1

            record = self.normalize_record(evidence)
            raw_tasks = self.generate_tasks(record, raw_text)
            self._stats["tasks_generated"] += len(raw_tasks)

            for task in raw_tasks:
                if self.verify_sample(task, record):
                    valid_samples.append(task)
                    self._stats["validated_valid"] += 1
                else:
                    self._stats["validated_invalid"] += 1
        except Exception as e:
            self._stats["errors"] += 1
            print(f"Error processing text: {e}")

        return valid_samples

    # ==========================================
    # STAGE 1: Evidence Extraction
    # ==========================================
    @staticmethod
    def _derive_model(product_name: str, brand: str) -> str:
        name = (product_name or "").strip()
        b = (brand or "").strip()
        if not name:
            return ""
        if b and name.lower().startswith(b.lower() + " "):
            return name[len(b) :].strip()
        return name

    @staticmethod
    def _infer_category(text: str, product_name: str = "") -> str:
        low = f"{product_name} {text}".lower()

        tablet_terms = [
            "ipad",
            "tablet",
            "surface pro",
        ]
        laptop_terms = [
            "laptop",
            "notebook",
            "macbook",
            "galaxy book",
            "book4",
            "book 4",
            "motobook",
            "vivobook",
            "zenbook",
            "thinkpad",
            "inspiron",
            "pavilion",
            "ideapad",
            "ultrabook",
            "chromebook",
            "intel core",
            "windows 11 operating system",
            "ssd",
        ]
        phone_terms = [
            "iphone",
            "smartphone",
            "mobile",
            "oneplus",
            "pixel",
            "galaxy s",
            "galaxy z",
        ]

        if any(t in low for t in tablet_terms):
            return "tablet"
        if any(t in low for t in laptop_terms):
            return "laptop"
        if any(t in low for t in phone_terms):
            return "smartphone"
        return "electronics"

    @staticmethod
    def _pick_instruction(pool: List[str], seed_text: str) -> str:
        if not pool:
            return ""
        seed = md5((seed_text or "").encode("utf-8")).hexdigest()
        idx = int(seed[:8], 16) % len(pool)
        return pool[idx]

    @staticmethod
    def _price_to_inr_int(price_text: str) -> Optional[int]:
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
    def _is_truncated_model(model: str) -> bool:
        m = str(model or "").strip()
        if not m:
            return True
        if "..." in m or "…" in m:
            return True
        if m.endswith("-"):
            return True
        return False

    def extract_evidence(self, text: str) -> Dict[str, Any]:
        """
        Parse raw product text into a grounded evidence dictionary.
        Uses EvidenceBuilder first, then fallback regex fills for robustness.
        """
        evidence: Dict[str, Any] = {}
        parsed = self._builder.parse_evidence_from_text(text or "")

        product_name = str(parsed.get("product_name") or "").strip()
        brand = str(parsed.get("brand") or "").strip()

        if brand:
            evidence["brand"] = brand
        if product_name:
            evidence["model"] = self._derive_model(product_name, brand)
            evidence["product_name"] = product_name

        inferred_category = self._infer_category(text or "", product_name)
        if inferred_category:
            evidence["category"] = inferred_category

        # Canonical parser values
        if parsed.get("price"):
            evidence["price_raw"] = str(parsed.get("price")).strip()
        if parsed.get("rating"):
            rating_value = str(parsed.get("rating")).strip()
            evidence["rating_raw"] = rating_value.split("/", 1)[0].strip()
        if parsed.get("storage"):
            evidence["storage_raw"] = str(parsed.get("storage")).strip()
        if parsed.get("chip"):
            evidence["processor_raw"] = str(parsed.get("chip")).strip()
        if parsed.get("camera_rear"):
            evidence["rear_camera_raw"] = str(parsed.get("camera_rear")).strip()
        if parsed.get("camera_front"):
            evidence["front_camera_raw"] = str(parsed.get("camera_front")).strip()
        if parsed.get("display"):
            evidence["display_raw"] = str(parsed.get("display")).strip()
        if parsed.get("os"):
            evidence["os_raw"] = str(parsed.get("os")).strip()
        if parsed.get("warranty"):
            evidence["warranty_raw"] = str(parsed.get("warranty")).strip()
        if parsed.get("ram"):
            evidence["ram_raw"] = str(parsed.get("ram")).strip()
        if parsed.get("review_count") is not None:
            evidence["reviews_raw"] = str(parsed.get("review_count"))

        # Reviews count (supports comma-formatted values)
        reviews_match = re.search(r"\(([\d,]+)\s+reviews?\)", text or "", re.IGNORECASE)
        if reviews_match:
            evidence["reviews_raw"] = reviews_match.group(1).replace(",", "")

        # Fallback fills when fields are still missing
        if "brand" not in evidence or "model" not in evidence:
            product_match = re.search(r"Product:\s*(.+)", text or "", flags=re.IGNORECASE)
            if product_match:
                product_str = product_match.group(1).strip()
                parts = product_str.split()
                if parts and "brand" not in evidence:
                    evidence["brand"] = parts[0]
                if len(parts) > 1 and "model" not in evidence:
                    evidence["model"] = " ".join(parts[1:])

        if "price_raw" not in evidence:
            price_match = re.search(
                r"Price:\s*((?:INR|Rs\.?|₹|\$|USD)\s*[\d,]+(?:\.\d+)?)",
                text or "",
                re.IGNORECASE,
            )
            if price_match:
                evidence["price_raw"] = price_match.group(1).strip()

        if "storage_raw" not in evidence:
            storage_match = re.search(
                r"(\d+\s*(?:GB|TB)\s*(?:ROM|SSD|HDD|Storage))",
                text or "",
                re.IGNORECASE,
            )
            if storage_match:
                evidence["storage_raw"] = storage_match.group(1).strip()

        # Return only when we have enough usable information.
        signal_keys = {
            "brand",
            "model",
            "price_raw",
            "storage_raw",
            "processor_raw",
            "display_raw",
            "rear_camera_raw",
            "front_camera_raw",
        }
        if any(k in evidence and str(evidence[k]).strip() for k in signal_keys):
            return evidence
        return {}

    # ==========================================
    # STAGE 2: Canonical Record Builder
    # ==========================================
    def normalize_record(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize evidence into a canonical record."""
        record: Dict[str, Any] = {}

        brand = evidence.get("brand")
        if isinstance(brand, str) and brand.strip():
            record["brand"] = brand.strip().title()

        model = evidence.get("model")
        if isinstance(model, str) and model.strip():
            clean_model = model.strip().rstrip("- ").strip()
            clean_model = re.sub(r"[.]{3,}$", "", clean_model).strip()
            clean_model = re.sub(r"[…]+$", "", clean_model).strip()
            record["model"] = clean_model

        category = evidence.get("category")
        if isinstance(category, str) and category.strip():
            record["category"] = category.strip().lower()

        price_raw = evidence.get("price_raw")
        if isinstance(price_raw, str) and price_raw.strip():
            record["price"] = price_raw.strip()
            price_inr = self._price_to_inr_int(price_raw)
            if price_inr is not None:
                record["price_inr"] = price_inr

        rating_raw = evidence.get("rating_raw")
        if isinstance(rating_raw, str) and rating_raw.strip():
            try:
                rating_val = float(rating_raw)
                record["rating"] = round(rating_val, 1)
            except ValueError:
                pass

        reviews_raw = evidence.get("reviews_raw")
        if isinstance(reviews_raw, str) and reviews_raw.isdigit():
            record["reviews"] = int(reviews_raw)
            record["review_count"] = int(reviews_raw)

        for key_raw, key_norm in [
            ("storage_raw", "storage"),
            ("ram_raw", "ram"),
            ("processor_raw", "processor"),
            ("rear_camera_raw", "rear_camera"),
            ("front_camera_raw", "front_camera"),
            ("display_raw", "display"),
            ("os_raw", "os"),
            ("warranty_raw", "warranty"),
        ]:
            val = evidence.get(key_raw)
            if isinstance(val, str) and val.strip():
                record[key_norm] = val.strip()

        return record

    # ==========================================
    # STAGE 3: Task Generator
    # ==========================================
    def generate_tasks(self, record: Dict[str, Any], raw_text: str) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        if not record:
            return tasks
        if self._is_truncated_model(record.get("model", "")):
            return tasks

        tasks.append(self.make_extraction_task(record, raw_text))

        # Keep one QA task, but avoid overfitting to price-only.
        if record.get("price") and record.get("rating"):
            tasks.append(self.make_price_rating_qa_task(record, raw_text))
        elif record.get("ram") and record.get("storage"):
            tasks.append(self.make_storage_qa_task(record, raw_text))
        elif record.get("price"):
            tasks.append(self.make_price_qa_task(record, raw_text))
        elif record.get("storage"):
            tasks.append(self.make_storage_qa_task(record, raw_text))

        tasks.append(self.make_summarization_task(record, raw_text))

        # Add reasoning when enough evidence exists.
        if record.get("processor") or record.get("storage"):
            tasks.append(self.make_reasoning_task(record, raw_text))

        return tasks

    def make_extraction_task(self, record: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        seed = f"{record.get('model','')}|extraction"
        instruction = self._pick_instruction(self.EXTRACTION_INSTRUCTIONS, seed)
        output = {
            "brand": record.get("brand"),
            "model": record.get("model"),
            "category": record.get("category"),
            "price": record.get("price"),
            "price_inr": record.get("price_inr"),
            "ram": record.get("ram"),
            "storage": record.get("storage"),
            "processor": record.get("processor"),
            "display": record.get("display"),
            "os": record.get("os"),
            "rating": record.get("rating"),
            "review_count": record.get("review_count"),
            "rear_camera": record.get("rear_camera"),
            "front_camera": record.get("front_camera"),
            "warranty": record.get("warranty"),
        }
        return {
            "instruction": instruction,
            "input": raw_text,
            "output": json.dumps(output, ensure_ascii=False),
            "task_type": "extraction",
        }

    def make_price_rating_qa_task(self, record: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        instruction = "What is the price and rating of this product?"
        lines = []
        if record.get("price"):
            lines.append(f"Price: {record['price']}")
        if record.get("rating"):
            lines.append(f"Rating: {record['rating']}")
        if not lines:
            lines.append("The context does not provide information about price, rating.")
        return {
            "instruction": instruction,
            "input": raw_text,
            "output": "\n".join(lines),
            "task_type": "qa",
        }

    def make_price_qa_task(self, record: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        seed = f"{record.get('model','')}|qa_price"
        instruction = self._pick_instruction(self.QA_PRICE_INSTRUCTIONS, seed)
        output = record.get("price", "Not provided")
        return {
            "instruction": instruction,
            "input": raw_text,
            "output": output,
            "task_type": "qa",
        }

    def make_storage_qa_task(self, record: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        seed = f"{record.get('model','')}|qa_storage"
        instruction = self._pick_instruction(self.QA_RAM_STORAGE_INSTRUCTIONS, seed)
        if record.get("ram") and record.get("storage"):
            output = f"Ram: {record['ram']}\nStorage: {record['storage']}"
        elif record.get("storage"):
            output = f"Storage: {record['storage']}"
        else:
            output = "The context does not provide information about ram, storage."
        return {
            "instruction": instruction,
            "input": raw_text,
            "output": output,
            "task_type": "qa",
        }

    def make_reasoning_task(self, record: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        seed = f"{record.get('model','')}|reasoning"
        instruction = self._pick_instruction(self.REASONING_INSTRUCTIONS, seed)

        thoughts: List[str] = []
        if record.get("storage"):
            thoughts.append(
                f"Storage is {record['storage']}, which supports heavier usage."
            )
        if record.get("processor"):
            thoughts.append(
                f"Processor is {record['processor']}, suggesting performance potential."
            )
        if record.get("rating"):
            thoughts.append(f"Rating is {record['rating']}, indicating user feedback quality.")
        if record.get("price"):
            thoughts.append(f"Price is {record['price']}, relevant for value evaluation.")
        if not thoughts:
            thoughts.append("Available evidence is limited in the provided context.")

        score = sum(
            1
            for key in ("storage", "processor", "price", "warranty")
            if record.get(key)
        ) + (1 if record.get("rating") else 0)
        if score >= 3:
            conclusion = "Recommendation: Based on available evidence, this product appears to offer good value."
        else:
            conclusion = "Recommendation: Value is uncertain due to missing evidence."

        output = "Analysis: " + " ".join(thoughts) + "\n" + conclusion
        return {
            "instruction": instruction,
            "input": raw_text,
            "output": output,
            "task_type": "reasoning",
        }

    def make_summarization_task(self, record: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
        seed = f"{record.get('model','')}|summary"
        instruction = self._pick_instruction(self.SUMMARY_INSTRUCTIONS, seed)

        model = record.get("model", "This product")
        brand = record.get("brand", "Unknown brand")
        ram = record.get("ram")
        storage = record.get("storage")
        line1 = f"{model} by {brand}"
        if ram:
            line1 += f" • {ram}"
        if storage:
            line1 += f" • {storage}"

        line2_parts = []
        if record.get("processor"):
            line2_parts.append(str(record["processor"]))
        if record.get("display"):
            line2_parts.append(str(record["display"]))
        if record.get("os"):
            line2_parts.append(str(record["os"]))
        if record.get("price"):
            line2_parts.append(f"Price: {record['price']}")
        if record.get("rating"):
            line2_parts.append(f"Rating: {record['rating']}")
        line2 = " | ".join(line2_parts) if line2_parts else "Specs available in product description."

        return {
            "instruction": instruction,
            "input": raw_text,
            "output": f"{line1}\n{line2}",
            "task_type": "summarization",
        }

    # ==========================================
    # STAGE 4: Dataset Verifier
    # ==========================================
    def verify_sample(self, sample: Dict[str, Any], record: Dict[str, Any]) -> bool:
        """
        Local strict verifier.
        Prevents obvious bad samples before global verification/correction.
        """
        instruction = (sample.get("instruction") or "").lower()
        input_text = str(sample.get("input") or "")
        task_type = sample.get("task_type", "qa")
        output_obj = sample.get("output")

        if output_obj in (None, ""):
            return False

        if task_type == "extraction" or "extract" in instruction:
            if isinstance(output_obj, dict):
                parsed = output_obj
            else:
                try:
                    parsed = json.loads(str(output_obj))
                except ValueError:
                    return False
                if not isinstance(parsed, dict):
                    return False

            allowed_fields = {
                "brand",
                "model",
                "category",
                "price",
                "price_inr",
                "ram",
                "storage",
                "display",
                "os",
                "rating",
                "review_count",
                "rear_camera",
                "front_camera",
                "processor",
                "warranty",
            }
            if any(k not in allowed_fields for k in parsed.keys()):
                return False

            for key, val in parsed.items():
                if key in record and record[key] not in (None, ""):
                    if str(val) != str(record[key]):
                        return False
            return True

        if task_type == "qa":
            answer = str(output_obj).strip()
            if not answer:
                return False

            lower_inst = instruction.lower()
            pairs = {}
            for line in answer.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    pairs[key.strip().lower()] = val.strip()

            if "price" in lower_inst:
                expected_price = str(record.get("price", "")).strip()
                answer_price = pairs.get("price")
                if not answer_price:
                    answer_price = re.sub(r"^\s*price\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
                if answer_price not in (expected_price, "Not provided"):
                    return False
            if "rating" in lower_inst and record.get("rating"):
                answer_rating = pairs.get("rating", "").strip()
                if answer_rating and answer_rating != str(record.get("rating")):
                    return False
            if "ram" in lower_inst and record.get("ram"):
                answer_ram = pairs.get("ram", "").strip()
                if answer_ram and answer_ram != str(record.get("ram")):
                    return False
            if "storage" in lower_inst and record.get("storage"):
                answer_storage = pairs.get("storage", "").strip()
                if answer_storage and answer_storage != str(record.get("storage")):
                    return False

            if answer and answer not in ("Not provided", ""):
                # If key/value format is used, each value part must be grounded.
                lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
                if any(":" in ln for ln in lines):
                    for ln in lines:
                        if ":" not in ln:
                            continue
                        _, rhs = ln.split(":", 1)
                        val = rhs.strip()
                        if not val:
                            continue
                        norm_val = re.sub(r"[,\s]", "", val)
                        norm_input = re.sub(r"[,\s]", "", input_text)
                        if norm_val and norm_val not in norm_input:
                            return False
                else:
                    norm_answer = re.sub(r"[,\s]", "", answer)
                    norm_input = re.sub(r"[,\s]", "", input_text)
                    if norm_answer and norm_answer not in norm_input:
                        return False
            return True

        if task_type == "reasoning":
            out_str = str(output_obj)
            if "analysis:" not in out_str.lower():
                return False
            if "recommendation:" not in out_str.lower():
                return False
            # Require at least one grounded spec token overlap.
            norm_out = re.sub(r"[,\s]+", " ", out_str.lower())
            norm_in = re.sub(r"[,\s]+", " ", input_text.lower())
            for token in [
                str(record.get("storage") or "").lower(),
                str(record.get("processor") or "").lower(),
                str(record.get("price") or "").lower(),
                str(record.get("rating") or "").lower(),
            ]:
                tok = token.strip()
                if tok and tok in norm_out and tok in norm_in:
                    return True
            return False

        if task_type == "summarization":
            out_str = str(output_obj).strip()
            if len(out_str) < 20:
                return False
            common = set(out_str.split()) & set(input_text.split())
            if len(common) < 3:
                return False
            return True

        return False

    # ==========================================
    # STAGE 5: Deduplication + Balancing
    # ==========================================
    @staticmethod
    def deduplicate_and_balance(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique_samples = []
        for sample in samples:
            key = md5((sample["instruction"] + sample["input"]).encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                unique_samples.append(sample)

        by_type = {"extraction": [], "qa": [], "summarization": [], "reasoning": []}
        for sample in unique_samples:
            ttype = sample.get("task_type", "qa")
            if ttype in by_type:
                by_type[ttype].append(sample)

        base_unit = min(
            len(by_type["extraction"]) / 4.0,
            len(by_type["qa"]) / 3.0,
            len(by_type["summarization"]) / 2.0,
            len(by_type["reasoning"]) / 1.0,
        )

        base_unit_int = int(base_unit)
        if base_unit_int == 0:
            for item in unique_samples:
                item.pop("task_type", None)
            return unique_samples

        e_count = int(base_unit_int * 4)
        q_count = int(base_unit_int * 3)
        s_count = int(base_unit_int * 2)
        r_count = int(base_unit_int * 1)

        balanced = (
            by_type["extraction"][:e_count]
            + by_type["qa"][:q_count]
            + by_type["summarization"][:s_count]
            + by_type["reasoning"][:r_count]
        )
        for item in balanced:
            item.pop("task_type", None)
        shuffle(balanced)
        return balanced

    # ==========================================
    # STAGE 6: Dataset Export
    # ==========================================
    @staticmethod
    def export_jsonl(samples: List[Dict[str, Any]], path: str):
        with open(path, "w") as f:
            for sample in samples:
                row = {
                    k: v for k, v in sample.items() if k in ("instruction", "input", "output")
                }
                if isinstance(row.get("output"), (dict, list)):
                    row["output"] = json.dumps(row["output"], ensure_ascii=False)
                f.write(json.dumps(row) + "\n")

    def get_stats(self) -> Dict[str, int]:
        return self._stats
