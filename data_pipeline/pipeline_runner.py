"""
Two-layer dataset pipeline runner
=================================

Pipeline flow:
1. ingest_urls()
2. scrape_html()
3. build_evidence_records()
4. generate_instruction_pairs()
5. validate_and_correct()
6. score_quality()
7. export_jsonl()
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dataset_generator import DatasetGenerator
from .evidence_builder import EvidenceBuilder
from .quality_scorer import TwoLayerQualityScorer
from .verification_agent import DatasetVerificationAgent


class TwoLayerPipelineRunner:
    """Strict two-layer pipeline orchestrator for LLM instruction datasets."""

    def __init__(
        self,
        seed: int = 42,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        if round(train_ratio + validation_ratio + test_ratio, 6) != 1.0:
            raise ValueError("Split ratios must sum to 1.0")

        self.seed = seed
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

        self.evidence_builder = EvidenceBuilder()
        self.dataset_generator = DatasetGenerator(seed=seed)
        self.verification_agent = DatasetVerificationAgent()
        self.quality_scorer = TwoLayerQualityScorer()

    # -------------------------
    # 1) ingest_urls
    # -------------------------
    def ingest_urls(self, urls: Sequence[str]) -> List[str]:
        return self.evidence_builder.ingest_urls(urls)

    # -------------------------
    # 2) scrape_html
    # -------------------------
    def scrape_html(self, url: str) -> str:
        return self.evidence_builder.scrape_html(url)

    # -------------------------
    # 3) build_evidence_records
    # -------------------------
    def build_evidence_records(
        self,
        urls: Optional[Sequence[str]] = None,
        html_pages: Optional[Sequence[str]] = None,
        raw_texts: Optional[Sequence[str]] = None,
        output_path: str = "evidence_records.jsonl",
    ) -> List[Dict[str, Any]]:
        return self.evidence_builder.build_evidence_records(
            urls=urls,
            html_pages=html_pages,
            raw_texts=raw_texts,
            output_path=output_path,
        )

    # -------------------------
    # 4) generate_instruction_pairs
    # -------------------------
    def generate_instruction_pairs(
        self,
        evidence_records: Optional[Sequence[Dict[str, Any]]] = None,
        evidence_path: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        if evidence_records is None:
            if not evidence_path:
                raise ValueError("Provide evidence_records or evidence_path.")
            evidence_records = self.evidence_builder.read_jsonl(evidence_path)
        return self.dataset_generator.generate_instruction_pairs(evidence_records)

    # -------------------------
    # 5) validate_and_correct
    # -------------------------
    def validate_and_correct(
        self, samples: Sequence[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        return self.verification_agent.validate_and_correct(samples)

    # -------------------------
    # 6) score_quality
    # -------------------------
    def score_quality(self, samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return self.quality_scorer.score_quality(samples)

    # -------------------------
    # 7) export_jsonl
    # -------------------------
    @staticmethod
    def _write_jsonl(records: Sequence[Dict[str, Any]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                row = {
                    "instruction": rec.get("instruction", ""),
                    "input": rec.get("input", ""),
                    "output": (
                        json.dumps(rec.get("output"), ensure_ascii=False)
                        if isinstance(rec.get("output"), (dict, list))
                        else rec.get("output", "")
                    ),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _split_dataset(self, samples: List[Dict[str, str]]) -> Tuple[List, List, List]:
        items = list(samples)
        rng = random.Random(self.seed)
        rng.shuffle(items)

        n = len(items)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.validation_ratio)
        train = items[:n_train]
        val = items[n_train : n_train + n_val]
        test = items[n_train + n_val :]
        return train, val, test

    def export_jsonl(self, samples: List[Dict[str, str]], output_dir: str = ".") -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        train, val, test = self._split_dataset(samples)

        train_path = os.path.join(output_dir, "train.jsonl")
        val_path = os.path.join(output_dir, "validation.jsonl")
        test_path = os.path.join(output_dir, "test.jsonl")

        self._write_jsonl(train, train_path)
        self._write_jsonl(val, val_path)
        self._write_jsonl(test, test_path)
        return {"train": train_path, "validation": val_path, "test": test_path}

    def run_pipeline(
        self,
        urls: Optional[Sequence[str]] = None,
        raw_texts: Optional[Sequence[str]] = None,
        html_pages: Optional[Sequence[str]] = None,
        output_dir: str = ".",
    ) -> Dict[str, Any]:
        evidence_path = os.path.join(output_dir, "evidence_records.jsonl")
        evidence_records = self.build_evidence_records(
            urls=urls,
            html_pages=html_pages,
            raw_texts=raw_texts,
            output_path=evidence_path,
        )

        generated = self.generate_instruction_pairs(evidence_records=evidence_records)
        corrected, reports = self.validate_and_correct(generated)
        quality_passed = self.score_quality(corrected)
        outputs = self.export_jsonl(quality_passed, output_dir=output_dir)

        return {
            "evidence_records": len(evidence_records),
            "generated_samples": len(generated),
            "validated_samples": len(corrected),
            "quality_passed": len(quality_passed),
            "quality_stats": self.quality_scorer.get_stats(),
            "report_status_counts": {
                "valid": sum(1 for r in reports if r["status"] == "valid"),
                "corrected": sum(1 for r in reports if r["status"] == "corrected"),
                "invalid": sum(1 for r in reports if r["status"] == "invalid"),
            },
            "files": {
                "evidence": evidence_path,
                **outputs,
            },
        }


def _read_lines(path: Optional[str]) -> List[str]:
    if not path:
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict two-layer dataset pipeline.")
    parser.add_argument("--urls_file", default="", help="Text file with one URL per line")
    parser.add_argument("--texts_file", default="", help="Text file with one raw text block per line")
    parser.add_argument("--output_dir", default=".", help="Directory for output files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    urls = _read_lines(args.urls_file)
    raw_texts = _read_lines(args.texts_file)

    runner = TwoLayerPipelineRunner(seed=args.seed)
    result = runner.run_pipeline(urls=urls, raw_texts=raw_texts, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
