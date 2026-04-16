"""
Auto Pipeline Config Module
============================
Analyzes ingested documents and automatically recommends optimal
pipeline settings: chunking method, chunk size, quality threshold,
template, fine-tune method, and target model.

All heuristics are deterministic — no LLM calls required.
"""

import re
from typing import Any, Dict, List


class AutoPipelineConfig:
    """
    Smart configuration engine that analyzes ingested document metadata
    and returns optimal pipeline settings.

    Usage:
    ------
    >>> from data_pipeline.auto_config import AutoPipelineConfig
    >>> config = AutoPipelineConfig.detect(docs)
    >>> print(config["chunk_method"], config["reasons"])
    """

    # ── Source-type classification ────────────────────────────────────

    ECOMMERCE_KEYWORDS = [
        "price", "₹", "rs.", "inr", "add to cart", "buy now",
        "rating", "reviews", "seller", "flipkart", "amazon",
        "product", "specifications", "warranty",
    ]

    # ── Public API ───────────────────────────────────────────────────

    @classmethod
    def detect(cls, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze ingested documents and return recommended config.

        Parameters
        ----------
        docs : list of dict
            Documents from DocumentIngestor, each with 'text',
            'source_type', 'char_count', 'word_count' fields.

        Returns
        -------
        dict
            Recommended configuration with keys:
            chunk_method, chunk_size, min_quality, template,
            domain, model, method, reasons.
        """
        if not docs:
            return cls._default_config(["No documents — using defaults"])

        # ── Gather signals ──────────────────────────────────────────
        source_types = cls._count_source_types(docs)
        total_chars = sum(d.get("char_count", len(d.get("text", ""))) for d in docs)
        total_words = sum(d.get("word_count", 0) for d in docs)
        doc_count = len(docs)
        avg_chars = total_chars / doc_count if doc_count else 0
        avg_words = total_words / doc_count if doc_count else 0

        # Content analysis
        is_ecommerce = cls._detect_ecommerce(docs, source_types)
        is_structured = cls._detect_structured(source_types)
        is_pdf = source_types.get("pdf", 0) > 0
        is_long_prose = avg_chars > 2000
        is_short_text = avg_chars < 500
        has_paragraphs = cls._detect_paragraphs(docs)

        reasons: List[str] = []

        # ── Chunking method ─────────────────────────────────────────
        if is_ecommerce:
            chunk_method = "semantic"
            reasons.append("🛒 E-commerce data detected → semantic chunking")
        elif is_pdf and is_long_prose:
            chunk_method = "sliding_window"
            reasons.append("📄 PDF with long content → sliding window")
        elif is_structured:
            chunk_method = "paragraph"
            reasons.append("📋 Structured data (JSON/XML) → paragraph chunking")
        elif is_short_text:
            chunk_method = "paragraph"
            reasons.append("📝 Short text → paragraph chunking")
        elif has_paragraphs:
            chunk_method = "paragraph"
            reasons.append("¶ Paragraph-structured content detected")
        else:
            chunk_method = "sliding_window"
            reasons.append("📄 General content → sliding window")

        # ── Chunk size ──────────────────────────────────────────────
        if is_ecommerce:
            chunk_size = 1024
            reasons.append("🛒 E-commerce → chunk_size=1024 (full product context)")
        elif is_long_prose:
            chunk_size = 768
            reasons.append(f"📏 Long docs (avg {int(avg_chars)} chars) → chunk_size=768")
        elif is_short_text:
            chunk_size = 256
            reasons.append(f"📏 Short docs (avg {int(avg_chars)} chars) → chunk_size=256")
        elif is_pdf:
            chunk_size = 512
            reasons.append("📄 PDF → chunk_size=512")
        else:
            chunk_size = 512
            reasons.append("📏 Default → chunk_size=512")

        # ── Quality threshold ───────────────────────────────────────
        if is_ecommerce:
            min_quality = 0.6
            reasons.append("🛒 E-commerce → stricter quality (0.6)")
        elif is_structured:
            min_quality = 0.5
            reasons.append("📋 Structured data → moderate quality (0.5)")
        elif is_short_text:
            min_quality = 0.3
            reasons.append("📝 Short text → relaxed quality (0.3)")
        elif is_long_prose:
            min_quality = 0.4
            reasons.append("📄 Long prose → quality threshold 0.4")
        else:
            min_quality = 0.4
            reasons.append("📊 Default quality threshold (0.4)")

        # ── Template ────────────────────────────────────────────────
        # Alpaca is the safest default for instruction-output pairs
        template = "alpaca"
        reasons.append("📝 Template: alpaca (instruction/input/output)")

        # ── Domain ──────────────────────────────────────────────────
        if is_ecommerce:
            domain = "ecommerce"
            reasons.append("🏷️ Domain: ecommerce")
        else:
            domain = "general"

        # ── Fine-tune method + model (based on estimated dataset size) ──
        # We estimate pairs from the doc/word stats since actual pair
        # count isn't known until process runs. Use a rough heuristic.
        estimated_pairs = cls._estimate_pair_count(
            doc_count, total_words, is_ecommerce
        )

        model, method = cls._recommend_model_and_method(estimated_pairs)
        reasons.append(
            f"🔧 ~{estimated_pairs} estimated pairs → {method.upper()}"
        )
        reasons.append(f"🤖 Recommended model: {model.split('/')[-1]}")

        return {
            "chunk_method": chunk_method,
            "chunk_size": chunk_size,
            "min_quality": min_quality,
            "template": template,
            "domain": domain,
            "model": model,
            "method": method,
            "estimated_pairs": estimated_pairs,
            "reasons": reasons,
            # Raw signals for debugging / frontend display
            "signals": {
                "doc_count": doc_count,
                "total_chars": total_chars,
                "total_words": total_words,
                "avg_chars": int(avg_chars),
                "avg_words": int(avg_words),
                "source_types": source_types,
                "is_ecommerce": is_ecommerce,
                "is_pdf": is_pdf,
                "is_structured": is_structured,
                "is_long_prose": is_long_prose,
                "is_short_text": is_short_text,
            },
        }

    @classmethod
    def recommend_export_settings(
        cls, pair_count: int, avg_quality: float
    ) -> Dict[str, Any]:
        """
        Recommend export settings (model + method) based on actual
        dataset statistics available after processing.

        Parameters
        ----------
        pair_count : int
            Number of quality-filtered pairs.
        avg_quality : float
            Average quality score.

        Returns
        -------
        dict
            Recommended model, method, and reasons.
        """
        model, method = cls._recommend_model_and_method(pair_count)
        reasons: List[str] = []
        reasons.append(f"📊 {pair_count} pairs → {method.upper()}")
        reasons.append(f"🤖 Recommended: {model.split('/')[-1]}")

        if avg_quality >= 0.7:
            reasons.append("⭐ High quality dataset — full fine-tune viable")
        elif avg_quality >= 0.4:
            reasons.append("✅ Good quality — LoRA recommended")
        else:
            reasons.append("⚠️ Lower quality — QLoRA with more epochs recommended")

        return {
            "model": model,
            "method": method,
            "reasons": reasons,
        }

    # ── Private helpers ──────────────────────────────────────────────

    @classmethod
    def _default_config(cls, reasons: List[str]) -> Dict[str, Any]:
        return {
            "chunk_method": "sliding_window",
            "chunk_size": 512,
            "min_quality": 0.4,
            "template": "alpaca",
            "domain": "general",
            "model": "meta-llama/Meta-Llama-3-8B",
            "method": "lora",
            "estimated_pairs": 0,
            "reasons": reasons,
            "signals": {},
        }

    @staticmethod
    def _count_source_types(docs: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for doc in docs:
            st = doc.get("source_type", "unknown")
            counts[st] = counts.get(st, 0) + 1
        return counts

    @classmethod
    def _detect_ecommerce(
        cls, docs: List[Dict[str, Any]], source_types: Dict[str, int]
    ) -> bool:
        # Explicit ecommerce source type
        if source_types.get("ecommerce", 0) > 0:
            return True

        # Heuristic: check if enough docs contain ecommerce keywords
        ecom_docs = 0
        for doc in docs:
            text = (doc.get("text") or "").lower()
            hits = sum(1 for kw in cls.ECOMMERCE_KEYWORDS if kw in text)
            if hits >= 3:
                ecom_docs += 1
        return ecom_docs > len(docs) * 0.5

    @staticmethod
    def _detect_structured(source_types: Dict[str, int]) -> bool:
        return (
            source_types.get("json", 0) + source_types.get("xml", 0)
        ) > 0

    @staticmethod
    def _detect_paragraphs(docs: List[Dict[str, Any]]) -> bool:
        """Check if documents have clear paragraph structure (double newlines)."""
        for doc in docs[:5]:  # Sample first 5
            text = doc.get("text", "")
            if text.count("\n\n") >= 3:
                return True
        return False

    @staticmethod
    def _estimate_pair_count(
        doc_count: int, total_words: int, is_ecommerce: bool
    ) -> int:
        """Rough estimate of how many instruction pairs the data will yield."""
        if is_ecommerce:
            # E-commerce: ~3-4 pairs per product doc
            return doc_count * 3
        # General: ~1 pair per 200 words, minimum 1 per doc
        word_based = max(total_words // 200, 1)
        doc_based = doc_count
        return max(word_based, doc_based)

    @staticmethod
    def _recommend_model_and_method(pair_count: int) -> tuple:
        """
        Recommend model and fine-tune method based on dataset size.

        Returns (model_name, method)
        """
        if pair_count < 50:
            # Very small dataset — use a small model with QLoRA
            return ("microsoft/Phi-3-mini-4k-instruct", "qlora")
        elif pair_count < 500:
            # Small-medium — LoRA on smaller model
            return ("mistralai/Mistral-7B-v0.3", "lora")
        elif pair_count < 5000:
            # Medium — LoRA on standard model
            return ("meta-llama/Meta-Llama-3-8B", "lora")
        else:
            # Large dataset — LoRA with bigger model
            return ("meta-llama/Meta-Llama-3-8B", "lora")
