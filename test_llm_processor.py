#!/usr/import env python3
import os
import sys
import json
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.llm_data_processor import LLMDataProcessor

PASS = "✅"
FAIL = "❌"
results = []

def report(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed))
    print(f"  {status} {name}")
    if detail:
        print(f"      {detail}")

def test_pipeline_stages():
    print("\n── Test: LLM Pipeline 6-Stage Flow ────────────────────────────────")
    processor = LLMDataProcessor()
    
    raw_text = """
    Product: Apple iPhone 17 Pro Max (Silver, 256 GB)
    Price: INR 149,900
    Rating: 4.8/5 (552 reviews)
    Features: 48MP + 48MP + 48MP Rear Camera
    A19 Chip Processor
    Display: 6.9 inch Super Retina XDR
    256 GB ROM
    1 year warranty
    """
    
    # Run pipeline stages 1-4
    valid_pairs = processor.process_raw_text(raw_text)
    
    # Assertions
    stats = processor.get_stats()
    report("Stats tracked processed count", stats["processed"] == 1)
    report("Stats tracked evidence_extracted count", stats["evidence_extracted"] == 1)
    report("Stats tracked tasks_generated count", stats["tasks_generated"] == 4)
    report("Stats tracked validated_valid count", stats["validated_valid"] == 4)
    report("Stats tracked validated_invalid count", stats["validated_invalid"] == 0)
    
    report("Returned exactly 4 valid pair", len(valid_pairs) == 4)
    if valid_pairs:
        # Check task types
        extraction = next((p for p in valid_pairs if p.get("task_type") == "extraction"), None)
        qa = next((p for p in valid_pairs if p.get("task_type") == "qa"), None)
        reasoning = next((p for p in valid_pairs if p.get("task_type") == "reasoning"), None)
        
        report("Extraction task generated", bool(extraction))
        if extraction:
            raw = extraction["output"]
            j = raw if isinstance(raw, dict) else json.loads(raw)
            report("Extraction contains brand", j.get("brand") == "Apple")
            report("Extraction extracted Apple", j.get("brand") == "Apple", detail=f"Found: {j.get('brand')}")
            report("Extraction contains price", j.get("price") == "149900", detail=f"Found: {j.get('price')}")
            
        report("QA task generated", bool(qa))
        if qa:
            report("QA output is correct price", qa["output"] == "149900")
            
        report("Reasoning task generated", bool(reasoning))
        if reasoning:
            report("Reasoning mentions camera specs", "48MP" in reasoning["output"])

def test_deduplication_and_balancing():
    print("\n── Test: Stage 5 Deduplication & Balancing ──────────────")
    processor = LLMDataProcessor()
    
    raw_tasks = [
        {"instruction": "q1", "input": "in", "output": "out", "task_type": "extraction"},
        {"instruction": "q1", "input": "in", "output": "out", "task_type": "extraction"}, # duplicate
        {"instruction": "q2", "input": "in", "output": "out", "task_type": "qa"},
        {"instruction": "q3", "input": "in", "output": "out", "task_type": "qa"},
        {"instruction": "q4", "input": "in", "output": "out", "task_type": "qa"},
        {"instruction": "s1", "input": "in", "output": "out", "task_type": "summarization"},
        {"instruction": "s2", "input": "in", "output": "out", "task_type": "summarization"},
        {"instruction": "r1", "input": "in", "output": "out", "task_type": "reasoning"}
    ]
    
    balanced = processor.deduplicate_and_balance(raw_tasks)
    report("Duplicates removed", len(balanced) < len(raw_tasks))
    # It should balance to 4:3:2:1 based on the limiting factor (reasoning = 1) -> 4 e, 3 q, 2 s, 1 r.
    # We only have 1 unique extraction, so base_unit is min(1/4, 3/3, 2/2, 1/1) = 0.25 -> 1 e, 0 q, 0 s, 0 r.
    # Wait, if base_unit == 0 in integer math? We return all unique as a fallback in our code!
    report("Successfully fell back if perfect balance impossible", len(balanced) == 7, detail=f"Length was {len(balanced)}")
    
    # Internal task_types should be removed
    has_ttype = any("task_type" in item for item in balanced)
    report("Internal task_type field stripped", not has_ttype)

def main():
    print("\n" + "=" * 60)
    print("  LLM DATA PROCESSOR TESTS")
    print("=" * 60)

    test_pipeline_stages()
    test_deduplication_and_balancing()

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
