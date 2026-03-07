"""
Quality Scorer Module
=====================
Evaluates and filters LLM training samples for quality, deduplication,
and toxicity/bias before fine-tuning.
"""

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
        "hate",
        "kill",
        "murder",
        "terrorist",
        "bomb",
        "slur",
        "racist",
        "sexist",
        "nazi",
        "supremacist",
        "genocide",
        "torture",
        "rape",
        "abuse",
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

            quality = {
                "length_score": self._score_length(text),
                "diversity_score": self._score_diversity(text),
                "coherence_score": self._score_coherence(text),
                "ratio_score": self._score_instruction_ratio(sample),
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
            parts.append(sample.get("output", ""))

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
            return sample["output"]
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
