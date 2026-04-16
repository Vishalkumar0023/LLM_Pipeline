"""
Quality Scorer Module
=====================
Evaluates and filters LLM training samples for quality, deduplication,
and toxicity/bias before fine-tuning.
"""

import json
import os
import re
import math
from typing import List, Dict, Any, Optional, Set

# scikit-learn is already a project dependency
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from .embedding_engine import EmbeddingEngine

    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

import numpy as np

try:
    from .evidence_builder import EvidenceBuilder

    HAS_EVIDENCE_BUILDER = True
except ImportError:
    HAS_EVIDENCE_BUILDER = False


class QualityScorer:
    """
    Scores and filters training data samples for LLM fine-tuning quality.

    Features:
    - Semantic deduplication (TF-IDF cosine similarity)
    - Toxicity/bias keyword filtering
    - Per-sample quality scoring (length, diversity, coherence)
    - Batch scoring and threshold filtering

    Example:
    --------
    >>> scorer = QualityScorer(min_quality_score=0.5)
    >>> scored = scorer.score(training_pairs)
    >>> filtered = scorer.filter(scored)
    >>> print(f"Kept {len(filtered)}/{len(scored)} samples")
    """

    # Default toxicity/bias keywords (English, conservative)
    DEFAULT_TOXIC_KEYWORDS = {
        "hate speech",
        "terrorist attack",
        "slur",
        "racist",
        "sexist",
        "nazi",
        "supremacist",
        "genocide",
        "torture chamber",
        "rape",
        "child abuse",
    }

    # Low-quality indicators
    LOW_QUALITY_PATTERNS = [
        re.compile(r"lorem ipsum", re.IGNORECASE),
        re.compile(r"click here", re.IGNORECASE),
        re.compile(r"subscribe now", re.IGNORECASE),
        re.compile(r"cookie policy", re.IGNORECASE),
        re.compile(r"terms of service", re.IGNORECASE),
        re.compile(r"all rights reserved", re.IGNORECASE),
        re.compile(r"©\s*\d{4}", re.IGNORECASE),
        re.compile(r"http[s]?://\S+"),  # Excessive URLs
    ]

    def __init__(
        self,
        min_quality_score: float = 0.4,
        similarity_threshold: float = 0.85,
        toxic_keywords: Optional[Set[str]] = None,
        custom_blocklist: Optional[Set[str]] = None,
        min_length: int = 50,
        max_length: int = 10000,
    ):
        """
        Initialize the quality scorer.

        Parameters
        ----------
        min_quality_score : float
            Minimum quality score (0–1) to keep a sample.
        similarity_threshold : float
            Cosine similarity threshold for deduplication (samples above
            this similarity to an earlier sample are flagged as duplicates).
        toxic_keywords : set of str, optional
            Custom toxicity keywords. If None, uses defaults.
        custom_blocklist : set of str, optional
            Additional blocked words/phrases to filter.
        min_length : int
            Minimum text length in characters.
        max_length : int
            Maximum text length in characters.
        """
        self.min_quality_score = min_quality_score
        self.similarity_threshold = similarity_threshold
        self.toxic_keywords = toxic_keywords or self.DEFAULT_TOXIC_KEYWORDS
        if custom_blocklist:
            self.toxic_keywords = self.toxic_keywords | custom_blocklist

        self.min_length = min_length
        self.max_length = max_length

        if HAS_EMBEDDINGS:
            # We initialize it lazily in _detect_duplicates to save memory
            self._embedding_engine: Optional[EmbeddingEngine] = None
        else:
            self._embedding_engine = None

        self._stats = {
            "total_samples": 0,
            "passed": 0,
            "filtered_quality": 0,
            "filtered_toxic": 0,
            "filtered_duplicate": 0,
            "filtered_length": 0,
            "avg_quality_score": 0.0,
            "filtered_count": 0,
        }

    def score(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score all samples for quality.

        Parameters
        ----------
        samples : list of dict
            Training pairs (from InstructFormatter).

        Returns
        -------
        list of dict
            Same samples with added 'quality' dict containing scores
            and flags.
        """
        self._stats["total_samples"] = len(samples)

        # Extract text for each sample
        texts = [self._get_sample_text(s) for s in samples]

        # Run semantic deduplication
        dup_flags = self._detect_duplicates(texts)

        scored_samples = []
        for i, sample in enumerate(samples):
            text = texts[i]

            originality = self._score_output_originality(sample)

            quality = {
                "length_score": self._score_length(text),
                "diversity_score": self._score_diversity(text),
                "coherence_score": self._score_coherence(text),
                "ratio_score": self._score_instruction_ratio(sample),
                "originality_score": originality,
                "is_toxic": self._check_toxicity(text),
                "low_quality_flags": self._check_low_quality(text),
                "is_duplicate": dup_flags[i],
                "overall_score": 0.0,
            }

            # Compute overall score (weighted average)
            quality["overall_score"] = (
                quality["length_score"] * 0.2
                + quality["diversity_score"] * 0.3
                + quality["coherence_score"] * 0.2
                + quality["ratio_score"] * 0.3
            )

            # Penalize for flags
            if quality["is_toxic"]:
                quality["overall_score"] *= 0.0  # Zero out toxic content
            if quality["is_duplicate"]:
                quality["overall_score"] *= 0.1  # Heavy penalty for duplicates
            if quality["low_quality_flags"]:
                penalty = len(quality["low_quality_flags"]) * 0.15
                quality["overall_score"] *= max(0.0, 1.0 - penalty)
            # Penalize output that is just a copy of the input
            if originality < 1.0:
                quality["overall_score"] *= originality

            sample_copy = dict(sample)
            sample_copy["quality"] = quality
            scored_samples.append(sample_copy)

        # Update stats
        scores = [s["quality"]["overall_score"] for s in scored_samples]
        if scores:
            self._stats["avg_quality_score"] = sum(scores) / len(scores)

        return scored_samples

    def filter(
        self, scored_samples: List[Dict[str, Any]], min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter scored samples by quality threshold.

        Parameters
        ----------
        scored_samples : list of dict
            Samples with 'quality' dict from score().
        min_score : float, optional
            Override minimum quality score.

        Returns
        -------
        list of dict
            Filtered samples that pass quality checks.
        """
        threshold = min_score if min_score is not None else self.min_quality_score
        filtered = []

        for sample in scored_samples:
            q = sample.get("quality", {})
            text = self._get_sample_text(sample)

            # Length check
            if len(text) < self.min_length or len(text) > self.max_length:
                self._stats["filtered_length"] += 1
                continue

            # Toxicity check
            if q.get("is_toxic", False):
                self._stats["filtered_toxic"] += 1
                continue

            # Duplicate check
            if q.get("is_duplicate", False):
                self._stats["filtered_duplicate"] += 1
                continue

            # Quality score check
            if q.get("overall_score", 0) < threshold:
                self._stats["filtered_quality"] += 1
                continue

            filtered.append(sample)

        self._stats["passed"] = len(filtered)
        return filtered

    def score_and_filter(
        self, samples: List[Dict[str, Any]], min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Convenience method: score then filter in one call."""
        scored = self.score(samples)
        return self.filter(scored, min_score)

    # ─── Scoring Components ──────────────────────────────────────────────

    def _score_length(self, text: str) -> float:
        """Score based on text length (sweet spot: 100–2000 chars)."""
        n = len(text)
        if n < 20:
            return 0.0
        elif n < 100:
            return 0.3
        elif n < 200:
            return 0.6
        elif n <= 2000:
            return 1.0
        elif n <= 5000:
            return 0.7
        else:
            return 0.4

    def _score_diversity(self, text: str) -> float:
        """
        Score vocabulary diversity using type-token ratio (TTR).
        Higher TTR = more diverse vocabulary.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0

        unique = len(set(words))
        total = len(words)

        # Corrected TTR (accounts for text length)
        # Use root TTR to reduce length bias
        if total == 0:
            return 0.0

        root_ttr = unique / math.sqrt(total)

        # Normalize to 0–1 (typical root TTR range: 3–12)
        score = min(1.0, max(0.0, (root_ttr - 2) / 8))
        return round(score, 3)

    def _score_coherence(self, text: str) -> float:
        """
        Score text coherence using simple heuristics:
        - Sentence structure (has proper sentences)
        - Paragraph structure
        - Not just a list of keywords
        """
        score = 0.0

        # Has proper sentences (ending with punctuation)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(sentences) >= 2:
            score += 0.4
        elif len(sentences) == 1:
            score += 0.2

        # Has multi-word sentences (not just keywords)
        avg_words = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        if avg_words >= 8:
            score += 0.3
        elif avg_words >= 4:
            score += 0.15

        # Has connecting words (indicates flowing text)
        connectors = {
            "however",
            "therefore",
            "furthermore",
            "moreover",
            "additionally",
            "because",
            "since",
            "although",
            "while",
            "thus",
            "hence",
            "consequently",
            "also",
            "then",
            "next",
            "finally",
            "first",
            "second",
        }
        words_lower = set(text.lower().split())
        connector_count = len(words_lower & connectors)
        if connector_count >= 3:
            score += 0.3
        elif connector_count >= 1:
            score += 0.15

        return min(1.0, score)

    def _score_instruction_ratio(self, sample: Dict[str, Any]) -> float:
        """
        Score the ratio between instruction/input and output lengths.
        Good training data has balanced but meaningful responses.
        """
        instruction = self._get_instruction(sample)
        response = self._get_response(sample)

        if not response:
            return 0.0

        inst_len = len(instruction)
        resp_len = len(response)

        if resp_len < 20:
            return 0.1  # Too short response

        # Response should be meaningful relative to instruction
        if inst_len == 0:
            return 0.5

        ratio = resp_len / max(inst_len, 1)

        if 0.3 <= ratio <= 5.0:
            return 1.0
        elif ratio < 0.3:
            return 0.3  # Response too short
        else:
            return 0.6  # Response much longer than instruction (ok but flag)

    def _score_output_originality(self, sample: Dict[str, Any]) -> float:
        """
        Penalize when output is just a verbatim dump of the input.
        Checks word-level overlap between output and input.
        """
        inp = ""
        if "input" in sample:
            inp = (sample.get("input") or "").strip().lower()
        elif "messages" in sample:
            msgs = sample.get("messages", [])
            for m in msgs:
                if m.get("role") == "user":
                    inp = m.get("content", "").strip().lower()
                    break
        elif "conversations" in sample:
            for c in sample.get("conversations", []):
                if c.get("from") == "human":
                    inp = c.get("value", "").strip().lower()
                    break

        out = self._get_response(sample).strip().lower()

        if not inp or not out:
            return 1.0  # Can't check, assume OK

        out_words = set(out.split())
        inp_words = set(inp.split())
        if not out_words:
            return 1.0

        overlap = len(out_words & inp_words) / len(out_words)
        # If the instruction implies extraction, loosen the penalty
        instruction = self._get_instruction(sample).lower()
        if "extract" in instruction or "json" in instruction or "specification" in instruction or "find" in instruction:
            if overlap >= 0.95:
                return 0.3
            return 1.0
            
        if overlap >= 0.85:
            return 0.1  # Nearly identical — heavily penalize
        elif overlap >= 0.7:
            return 0.4  # High overlap — moderate penalty
        return 1.0

    # ─── Detection Methods ───────────────────────────────────────────────

    def _detect_duplicates(self, texts: List[str]) -> List[bool]:
        """
        Detect semantic duplicates using the new EmbeddingEngine (Sentence-Transformers/FAISS)
        or fallback to TF-IDF cosine similarity.
        """
        if len(texts) < 2:
            return [False] * len(texts)

        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        if len(valid_indices) < 2:
            return [False] * len(texts)

        valid_texts = [texts[i] for i in valid_indices]
        dup_flags = [False] * len(texts)

        # 1. Try to use EmbeddingEngine (better for semantic dedup)
        if HAS_EMBEDDINGS:
            try:
                if self._embedding_engine is None:
                    self._embedding_engine = EmbeddingEngine()

                result = self._embedding_engine.deduplicate(
                    valid_texts, threshold=self.similarity_threshold
                )

                for j in result["duplicate_indices"]:
                    real_j = valid_indices[j]
                    dup_flags[real_j] = True

                return dup_flags
            except Exception:
                # If it fails (e.g., OOM), fallback to TF-IDF
                pass

        # 2. Fallback to TF-IDF
        if not HAS_SKLEARN:
            return dup_flags

        try:
            vectorizer = TfidfVectorizer(
                max_features=5000, stop_words="english", ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(valid_texts)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except Exception:
            return [False] * len(texts)

        # Mark duplicates (keep the first occurrence)
        dup_flags = [False] * len(texts)
        seen: Set[int] = set()

        for i in range(len(valid_texts)):
            if i in seen:
                continue
            for j in range(i + 1, len(valid_texts)):
                if j in seen:
                    continue
                if sim_matrix[i, j] >= self.similarity_threshold:
                    # Mark j as duplicate of i
                    real_j = valid_indices[j]
                    dup_flags[real_j] = True
                    seen.add(j)

        return dup_flags

    def _check_toxicity(self, text: str) -> bool:
        """Check if text contains toxic/biased keywords."""
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))
        return bool(words & self.toxic_keywords)

    def _check_low_quality(self, text: str) -> List[str]:
        """Check for low-quality content indicators."""
        flags = []
        for pattern in self.LOW_QUALITY_PATTERNS:
            if pattern.search(text):
                flags.append(pattern.pattern)
        return flags

    # ─── Text Extraction Helpers ─────────────────────────────────────────

    def _get_sample_text(self, sample: Dict[str, Any]) -> str:
        """Extract full text from a sample regardless of template format."""
        parts = []

        # Alpaca format
        if "instruction" in sample:
            parts.append(sample.get("instruction", ""))
            parts.append(sample.get("input", ""))
            out = sample.get("output", "")
            if isinstance(out, (dict, list)):
                out = json.dumps(out, ensure_ascii=False)
            parts.append(out)

        # ChatML format
        elif "messages" in sample:
            for msg in sample["messages"]:
                parts.append(msg.get("content", ""))

        # ShareGPT format
        elif "conversations" in sample:
            for conv in sample["conversations"]:
                parts.append(conv.get("value", ""))

        # Raw text
        elif "text" in sample:
            parts.append(sample["text"])

        return " ".join(p for p in parts if p)

    def _get_instruction(self, sample: Dict[str, Any]) -> str:
        """Extract instruction part from a sample."""
        if "instruction" in sample:
            return sample["instruction"]
        elif "messages" in sample:
            msgs = sample["messages"]
            for m in msgs:
                if m.get("role") == "user":
                    return m.get("content", "")
        elif "conversations" in sample:
            for c in sample["conversations"]:
                if c.get("from") == "human":
                    return c.get("value", "")
        return ""

    def _get_response(self, sample: Dict[str, Any]) -> str:
        """Extract response part from a sample."""
        if "output" in sample:
            out = sample["output"]
            if isinstance(out, (dict, list)):
                return json.dumps(out, ensure_ascii=False)
            return out
        elif "messages" in sample:
            msgs = sample["messages"]
            for m in msgs:
                if m.get("role") == "assistant":
                    return m.get("content", "")
        elif "conversations" in sample:
            for c in sample["conversations"]:
                if c.get("from") == "gpt":
                    return c.get("value", "")
        return ""

    def get_stats(self) -> Dict[str, Any]:
        """Return quality scoring statistics."""
        return self._stats

    def print_summary(self) -> None:
        """Print a formatted quality scoring summary."""
        stats = self._stats
        print("=" * 60)
        print("DATA QUALITY SCORING SUMMARY")
        print("=" * 60)
        print(f"\n📊 Total samples: {stats['total_samples']}")
        print(f"✅ Passed: {stats['passed']}")
        print(f"📈 Avg quality score: {stats['avg_quality_score']:.3f}")

        filtered_total = (
            stats["filtered_quality"]
            + stats["filtered_toxic"]
            + stats["filtered_duplicate"]
            + stats["filtered_length"]
        )
        if filtered_total > 0:
            print(f"\n🚫 Filtered: {filtered_total}")
            print(f"   • Quality too low: {stats['filtered_quality']}")
            print(f"   • Toxic/biased: {stats['filtered_toxic']}")
            print(f"   • Duplicates: {stats['filtered_duplicate']}")
            print(f"   • Length violations: {stats['filtered_length']}")

        print("=" * 60)


class TwoLayerQualityScorer:
    """
    Quality scorer specialized for two-layer evidence->instruction datasets.

    Reject rules:
    - hallucinated fields
    - instruction mismatch
    - invalid JSON
    - empty outputs
    """

    ALLOWED_JSON_FIELDS = {
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
    }
    COMPAT_CORE_FIELDS = {
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

    def __init__(self):
        self._stats = {
            "total_samples": 0,
            "passed": 0,
            "rejected": 0,
            "rejected_empty_output": 0,
            "rejected_invalid_json": 0,
            "rejected_instruction_mismatch": 0,
            "rejected_hallucinated_fields": 0,
            "rejected_reasoning_format": 0,
            "rejected_not_grounded": 0,
            "rejected_near_duplicate_family": 0,
        }
        self._builder = EvidenceBuilder() if HAS_EVIDENCE_BUILDER else None
        # Stricter family-level caps to reduce color/storage variant overfitting.
        self._family_task_limits = {
            "extraction": 1,
            "qa": 1,
            "summarization": 1,
            "reasoning": 1,
        }
        # Additional hard ceiling per family across all tasks.
        # Can be overridden via env for tuning experiments.
        self._family_total_limit = max(
            1, int(os.getenv("LLM_FAMILY_TOTAL_LIMIT", "3"))
        )

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
                "reason",
                "evaluate",
                "step-by-step",
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
        Keep only intent-bearing question phrase and drop product variant noise.
        """
        inst = (instruction or "").lower()
        inst = re.sub(r"\([^)]*\)", " ", inst)
        for sep in (" for ", " of ", " about ", " regarding "):
            if sep in inst:
                inst = inst.split(sep, 1)[0]
                break
        inst = re.sub(r"\s+", " ", inst).strip()
        return inst

    def _requested_fields(self, instruction: str) -> Set[str]:
        inst = self._intent_text(instruction)
        mapping = {
            "model": ["product name", "name", "model"],
            "brand": ["brand"],
            "category": ["category", "type"],
            "price": ["price", "cost"],
            "price_inr": ["price inr", "numeric price"],
            "rating": ["rating", "review"],
            "review_count": ["review count", "number of reviews"],
            "ram": ["ram", "memory"],
            "storage": ["storage", "rom", "ssd", "hdd"],
            "display": ["display", "screen"],
            "rear_camera": ["rear camera", "camera"],
            "front_camera": ["front camera", "selfie camera"],
            "processor": ["chip", "processor", "soc"],
            "os": ["operating system", "os"],
            "warranty": ["warranty"],
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
            if any(self._has_hint(inst, h) for h in hints):
                fields.add(field)
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
            fields.update(self.ALLOWED_JSON_FIELDS)
        return fields

    def _parse_json_output(self, output: Any) -> Optional[Dict[str, Any]]:
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
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _line_value_pairs(self, output: str) -> Dict[str, str]:
        pairs = {}
        for line in (output or "").splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            pairs[self._normalize(key)] = val.strip()
        return pairs

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

    def _expand_missing_aliases(self, fields: List[str]) -> Set[str]:
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

    def _value_grounded(self, input_text: str, value: Any) -> bool:
        if value in (None, ""):
            return True
        return self._normalize(str(value)) in self._normalize(input_text)

    def _infer_category(self, product_name: str, input_text: str) -> str:
        text = f"{product_name or ''} {input_text or ''}".lower()
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

    def _derive_model(self, product_name: str, brand: str) -> Optional[str]:
        name = (product_name or "").strip()
        b = (brand or "").strip()
        if not name:
            return None
        if b and name.lower().startswith(b.lower() + " "):
            return name[len(b) :].strip()
        return name

    def _expected_payload_from_input(self, input_text: str) -> Dict[str, Any]:
        if not self._builder:
            return {}
        evidence = self._builder.parse_evidence_from_text(input_text)
        product_name = evidence.get("product_name")
        brand = evidence.get("brand")
        model = self._derive_model(product_name or "", brand or "")
        if isinstance(model, str):
            model = re.sub(r"[.]{3,}$", "", model).strip()
            model = re.sub(r"[…]+$", "", model).strip()
            model = model.rstrip("- ").strip() or None
        rating_val = None
        if evidence.get("rating"):
            m = re.search(r"([\d.]+)", str(evidence.get("rating")))
            if m:
                try:
                    rating_val = round(float(m.group(1)), 1)
                except ValueError:
                    rating_val = None
        price_text = str(evidence.get("price") or "")
        price_inr = None
        if price_text:
            cleaned = re.sub(r"[^0-9.]", "", price_text)
            if cleaned:
                try:
                    price_inr = int(round(float(cleaned)))
                except ValueError:
                    price_inr = None
        return {
            "brand": brand,
            "category": self._infer_category(product_name or "", input_text),
            "model": model,
            "price": evidence.get("price"),
            "price_inr": price_inr,
            "ram": evidence.get("unified_memory") or evidence.get("ram"),
            "storage": evidence.get("storage"),
            "processor": evidence.get("chip"),
            "display": evidence.get("display"),
            "os": evidence.get("os"),
            "rating": rating_val,
            "review_count": evidence.get("review_count"),
            "rear_camera": evidence.get("camera_rear"),
            "front_camera": evidence.get("camera_front"),
            "warranty": evidence.get("warranty"),
        }

    def _product_family_key(self, input_text: str) -> Optional[str]:
        """
        Family-level key for near-duplicate pruning.
        Example: "Apple iPhone 17 Pro Max (Silver, 256 GB)" -> "apple iphone 17 pro max"
        """
        if not self._builder:
            return None
        evidence = self._builder.parse_evidence_from_text(input_text or "")
        product_name = str(evidence.get("product_name") or "").strip()
        if not product_name:
            return None
        # Remove variant parenthetical suffix like color/storage.
        base = re.sub(r"\s*\([^)]*\)\s*$", "", product_name).strip()
        if not base:
            base = product_name
        return self._normalize(base)

    def _intent_signature(self, instruction: str, task: str) -> str:
        inst = (instruction or "").lower()
        req = sorted(self._requested_fields(instruction))
        if task == "extraction":
            req_json = [f for f in req if f in self.ALLOWED_JSON_FIELDS]
            return "extract:" + (",".join(req_json) if req_json else "all")
        if task == "qa":
            return "qa:" + (",".join(req) if req else "generic")
        if task == "summarization":
            return "summary"
        # reasoning
        if any(k in inst for k in ["pros and cons", "advantages", "disadvantages"]):
            return "reasoning:pros_cons"
        if any(k in inst for k in ["power user"]):
            return "reasoning:power_user"
        if any(k in inst for k in ["good value", "value"]):
            return "reasoning:value"
        if any(k in inst for k in ["recommend"]):
            return "reasoning:recommend"
        return "reasoning:generic"

    def score_quality(self, samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Validate task-specific quality constraints and return passing samples only.
        Adds `quality` metadata to passed samples.
        """
        self._stats = {k: 0 for k in self._stats}
        self._stats["total_samples"] = len(samples or [])

        passed_samples: List[Dict[str, str]] = []
        for sample in samples or []:
            instruction = (sample.get("instruction") or "").strip()
            input_text = (sample.get("input") or "").strip()
            output_obj = sample.get("output")
            output = (
                json.dumps(output_obj, ensure_ascii=False)
                if isinstance(output_obj, dict)
                else str(output_obj or "").strip()
            )
            task = self._task_type(instruction)
            issues: List[str] = []

            if not output:
                issues.append("empty_output")
                self._stats["rejected_empty_output"] += 1

            if task == "extraction":
                parsed = self._parse_json_output(output_obj)
                if parsed is None:
                    issues.append("invalid_json")
                    self._stats["rejected_invalid_json"] += 1
                else:
                    expected_payload = self._expected_payload_from_input(input_text)
                    extra = [k for k in parsed.keys() if k not in self.ALLOWED_JSON_FIELDS]
                    if extra:
                        issues.append("hallucinated_fields")
                        self._stats["rejected_hallucinated_fields"] += 1

                    requested = {
                        f for f in self._requested_fields(instruction)
                        if f in self.ALLOWED_JSON_FIELDS
                    }
                    if requested and not requested.issubset(set(parsed.keys())):
                        missing_requested = set(requested) - set(parsed.keys())
                        enrich_only_missing = missing_requested.issubset(
                            self.ALLOWED_JSON_FIELDS - self.COMPAT_CORE_FIELDS
                        )
                        if not enrich_only_missing:
                            issues.append("instruction_mismatch")
                            self._stats["rejected_instruction_mismatch"] += 1
                    model_val = str(parsed.get("model") or "")
                    if "..." in model_val or "…" in model_val:
                        issues.append("instruction_mismatch")
                        self._stats["rejected_instruction_mismatch"] += 1

                    for key, value in parsed.items():
                        expected = expected_payload.get(key)
                        if expected not in (None, ""):
                            val_norm = self._normalize(str(value))
                            exp_norm = self._normalize(str(expected))
                            val_digits = re.sub(r"\D", "", str(value))
                            exp_digits = re.sub(r"\D", "", str(expected))
                            if not (
                                val_norm == exp_norm
                                or (val_norm and exp_norm and (val_norm in exp_norm or exp_norm in val_norm))
                                or (
                                    val_digits
                                    and exp_digits
                                    and (
                                        val_digits == exp_digits
                                        or val_digits in exp_digits
                                        or exp_digits in val_digits
                                    )
                                )
                            ):
                                issues.append("not_grounded")
                                self._stats["rejected_not_grounded"] += 1
                                break
                        elif not self._value_grounded(input_text, value):
                            issues.append("not_grounded")
                            self._stats["rejected_not_grounded"] += 1
                            break

            elif task == "qa":
                requested = self._requested_fields(instruction)
                pairs = self._line_value_pairs(output)
                expected_payload = self._expected_payload_from_input(input_text)
                inst_lower = instruction.lower()
                is_missing_specs = (
                    self._has_hint(inst_lower, "missing")
                    and (
                        self._has_hint(inst_lower, "specifications")
                        or self._has_hint(inst_lower, "specification")
                        or self._has_hint(inst_lower, "specs")
                    )
                )

                # QA tasks should not emit JSON object outputs.
                if self._parse_json_output(output_obj) is not None:
                    issues.append("instruction_mismatch")
                    self._stats["rejected_instruction_mismatch"] += 1

                # QA must only answer requested fields (except missing-info sentence).
                if not requested:
                    issues.append("instruction_mismatch")
                    self._stats["rejected_instruction_mismatch"] += 1
                if requested and pairs and not is_missing_specs:
                    norm_req = {self._normalize(f.replace("_", " ")) for f in requested}
                    for key in pairs.keys():
                        if key not in norm_req:
                            issues.append("instruction_mismatch")
                            self._stats["rejected_instruction_mismatch"] += 1
                            break
                listed_missing = self._extract_missing_fields_from_output(output)
                if requested and listed_missing and not is_missing_specs:
                    requested_norm = {self._normalized_field_name(f) for f in requested}
                    listed_set = self._expand_missing_aliases(listed_missing)
                    if any(field not in requested_norm for field in listed_set):
                        issues.append("instruction_mismatch")
                        self._stats["rejected_instruction_mismatch"] += 1

                if is_missing_specs:
                    low_out = output.lower()
                    has_missing_phrase = "the context does not provide information about" in low_out
                    has_all_present_phrase = "provides the main specifications" in low_out
                    # Reject vague targets that don't enumerate concrete fields.
                    is_vague = "several specifications" in low_out or "missing specifications" in low_out
                    if is_vague or (not has_missing_phrase and not has_all_present_phrase):
                        issues.append("instruction_mismatch")
                        self._stats["rejected_instruction_mismatch"] += 1
                    if has_missing_phrase:
                        evidence_row = (
                            self._builder.parse_evidence_from_text(input_text)
                            if self._builder
                            else {}
                        )
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
                        core_missing = [
                            f.replace("_", " ")
                            for f in core_order
                            if not expected_payload.get(f)
                        ]
                        optional_missing: List[str] = []
                        if not evidence_row.get("ram") and not evidence_row.get("unified_memory"):
                            optional_missing.append("ram")
                        if not evidence_row.get("battery"):
                            optional_missing.append("battery")
                        if not evidence_row.get("touch_id"):
                            optional_missing.append("touch id")
                        if not evidence_row.get("seller"):
                            optional_missing.append("seller")
                        if not evidence_row.get("availability"):
                            optional_missing.append("availability")
                        if not evidence_row.get("discount"):
                            optional_missing.append("discount")
                        if not evidence_row.get("weight"):
                            optional_missing.append("weight")
                        expected_missing = core_missing or optional_missing
                        expected_set = self._expand_missing_aliases(expected_missing)
                        listed_set = self._expand_missing_aliases(
                            self._extract_missing_fields_from_output(output)
                        )
                        if expected_set and not expected_set.issubset(listed_set):
                            issues.append("instruction_mismatch")
                            self._stats["rejected_instruction_mismatch"] += 1
                        if listed_set and expected_set and any(
                            field not in expected_set for field in listed_set
                        ):
                            issues.append("instruction_mismatch")
                            self._stats["rejected_instruction_mismatch"] += 1

                if not is_missing_specs:
                    for key, value in pairs.items():
                        expected = None
                        key_norm = self._normalize(key)
                        for field in self.ALLOWED_JSON_FIELDS:
                            if key_norm == self._normalize(field.replace("_", " ")):
                                expected = expected_payload.get(field)
                                break
                        if expected not in (None, ""):
                            if self._normalize(str(value)) != self._normalize(str(expected)):
                                issues.append("not_grounded")
                                self._stats["rejected_not_grounded"] += 1
                                break
                        elif value and not self._value_grounded(input_text, value):
                            issues.append("not_grounded")
                            self._stats["rejected_not_grounded"] += 1
                            break

            elif task == "reasoning":
                if "analysis:" not in output.lower() or "recommendation:" not in output.lower():
                    issues.append("reasoning_format")
                    self._stats["rejected_reasoning_format"] += 1
                inst_lower = instruction.lower()
                out_lower = output.lower()
                requested = self._requested_fields(instruction)
                asks_camera = (
                    "rear_camera" in requested
                    or "front_camera" in requested
                    or self._has_hint(inst_lower, "camera")
                )
                asks_battery = "battery" in requested or self._has_hint(inst_lower, "battery")
                asks_pros_cons = (
                    self._has_hint(inst_lower, "pros and cons")
                    or self._has_hint(inst_lower, "trade-off")
                    or self._has_hint(inst_lower, "trade-offs")
                    or self._has_hint(inst_lower, "tradeoff")
                    or self._has_hint(inst_lower, "trade offs")
                    or self._has_hint(inst_lower, "advantages and disadvantages")
                )
                if asks_camera and not any(tok in out_lower for tok in ["camera", "rear", "front"]):
                    issues.append("instruction_mismatch")
                    self._stats["rejected_instruction_mismatch"] += 1
                if asks_battery and "battery" not in out_lower:
                    issues.append("instruction_mismatch")
                    self._stats["rejected_instruction_mismatch"] += 1
                if asks_pros_cons and not any(tok in out_lower for tok in ["pros", "cons", "advantage", "disadvantage", "trade-off", "tradeoff"]):
                    issues.append("instruction_mismatch")
                    self._stats["rejected_instruction_mismatch"] += 1
                if self._has_hint(inst_lower, "power user"):
                    if not any(tok in out_lower for tok in ["storage", "processor", "chip"]):
                        issues.append("instruction_mismatch")
                        self._stats["rejected_instruction_mismatch"] += 1
                # Reasoning must reference evidence from input.
                if output and not self._value_grounded(input_text, output):
                    # Soft check: require at least one numeric/spec token shared.
                    input_tokens = set(re.findall(r"\b[\w/+.%-]+\b", self._normalize(input_text)))
                    output_tokens = set(re.findall(r"\b[\w/+.%-]+\b", self._normalize(output)))
                    overlap = len(input_tokens & output_tokens)
                    if overlap < 3:
                        issues.append("not_grounded")
                        self._stats["rejected_not_grounded"] += 1

            else:  # summarization
                if output and not self._value_grounded(input_text, output):
                    input_tokens = set(re.findall(r"\b[\w/+.%-]+\b", self._normalize(input_text)))
                    output_tokens = set(re.findall(r"\b[\w/+.%-]+\b", self._normalize(output)))
                    overlap = len(input_tokens & output_tokens)
                    if overlap < 3:
                        issues.append("not_grounded")
                        self._stats["rejected_not_grounded"] += 1

            if issues:
                self._stats["rejected"] += 1
                continue

            passed = dict(sample)
            if isinstance(passed.get("output"), (dict, list)):
                passed["output"] = json.dumps(passed["output"], ensure_ascii=False)
            passed["quality"] = {
                "task_type": task,
                "pass": True,
                "issues": [],
            }
            passed_samples.append(passed)

        self._stats["passed"] = len(passed_samples)
        # Family-level near-duplicate pruning:
        # keep at most one sample per (product_family, task_type, intent_signature)
        # to reduce color/storage variant duplication.
        deduped_samples: List[Dict[str, str]] = []
        seen_family_keys: Set[str] = set()
        for sample in passed_samples:
            input_text = (sample.get("input") or "").strip()
            task = self._task_type(sample.get("instruction", ""))
            family = self._product_family_key(input_text)
            if not family:
                deduped_samples.append(sample)
                continue
            intent = self._intent_signature(sample.get("instruction", ""), task)
            key = f"{family}||{task}||{intent}"
            if key in seen_family_keys:
                self._stats["rejected"] += 1
                self._stats["rejected_near_duplicate_family"] += 1
                continue
            seen_family_keys.add(key)
            deduped_samples.append(sample)

        self._stats["passed"] = len(deduped_samples)
        # Additional family-level quota pass:
        # keep only a small diverse set of tasks per product family.
        final_samples: List[Dict[str, str]] = []
        family_task_counts: Dict[str, Dict[str, int]] = {}
        family_total_counts: Dict[str, int] = {}
        for sample in deduped_samples:
            input_text = (sample.get("input") or "").strip()
            task = self._task_type(sample.get("instruction", ""))
            family = self._product_family_key(input_text)
            if not family:
                final_samples.append(sample)
                continue

            total_current = family_total_counts.get(family, 0)
            if total_current >= self._family_total_limit:
                self._stats["rejected"] += 1
                self._stats["rejected_near_duplicate_family"] += 1
                continue

            bucket = family_task_counts.setdefault(family, {})
            current = bucket.get(task, 0)
            limit = self._family_task_limits.get(task, 1)
            if current >= limit:
                self._stats["rejected"] += 1
                self._stats["rejected_near_duplicate_family"] += 1
                continue
            bucket[task] = current + 1
            family_total_counts[family] = total_current + 1
            final_samples.append(sample)

        self._stats["passed"] = len(final_samples)
        return final_samples

    def get_stats(self) -> Dict[str, Any]:
        """Return scoring stats."""
        return self._stats

    def print_summary(self) -> None:
        """Print summary for strict two-layer scoring."""
        stats = self._stats
        print("=" * 60)
        print("TWO-LAYER DATASET QUALITY SUMMARY")
        print("=" * 60)
        print(f"Total: {stats['total_samples']}")
        print(f"Passed: {stats['passed']}")
        print(f"Rejected: {stats['rejected']}")
        print(f" - Empty output: {stats['rejected_empty_output']}")
        print(f" - Invalid JSON: {stats['rejected_invalid_json']}")
        print(f" - Instruction mismatch: {stats['rejected_instruction_mismatch']}")
        print(f" - Hallucinated fields: {stats['rejected_hallucinated_fields']}")
        print(f" - Reasoning format: {stats['rejected_reasoning_format']}")
        print(f" - Not grounded: {stats['rejected_not_grounded']}")
        print(f" - Near-duplicate family: {stats['rejected_near_duplicate_family']}")
        print("=" * 60)
