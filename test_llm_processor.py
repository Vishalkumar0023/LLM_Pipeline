#!/usr/bin/env python3
"""
Test: LLM Data Processor
==========================
Validates the LLMDataProcessor pipeline stages using mock API responses.
"""

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


class MockLLMClient:
    """Mocks the LLMClient to return predefined responses for testing."""
    def __init__(self):
        self.calls = []
        
    def generate_text(self, system_prompt: str, user_prompt: str, model: str = "gpt-4o", temperature: float = 0.7) -> str:
        self.calls.append(("text", system_prompt, user_prompt))
        if "compression" in system_prompt.lower():
            return "Cleaned Mock Product: 16GB RAM, 512GB Storage, $999."
        return "Generic mock response"

    def generate_json(self, system_prompt: str, user_prompt: str, model: str = "gpt-4o", temperature: float = 0.1) -> Dict[str, Any]:
        self.calls.append(("json", system_prompt, user_prompt))
        
        # 1. Extraction Mock
        if "extraction" in system_prompt.lower():
            return {
                "product_name": "Mock Product",
                "brand": "MockBrand",
                "ram": "16GB",
                "storage": "512GB",
                "camera": None,
                "battery": "4000mAh",
                "price": "$999",
                "discount": "10%",
                "rating": "4.5",
                "delivery_date": "Tomorrow"
            }
            
        # 2. QA Generation Mock
        if "qa examples" in system_prompt.lower() or "instruction tuning data" in system_prompt.lower():
            return {
                "pairs": [
                    {
                        "instruction": "What is the RAM capacity?",
                        "input": "Cleaned Mock Product: 16GB RAM...",
                        "output": "The product has 16GB of RAM."
                    },
                    {
                        "instruction": "What is the price?",
                        "input": "Cleaned Mock Product...", # will be overwritten by processor
                        "output": "The price is $999."
                    }
                ]
            }
            
        # 3. Validation Mock
        if "validator" in system_prompt.lower():
            # Let's say the first QA pair is valid, second is invalid
            if "RAM capacity" in user_prompt:
                return {
                    "is_valid": True,
                    "problems": [],
                    "fix_suggestion": None
                }
            else:
                return {
                    "is_valid": False,
                    "problems": ["vague instruction"],
                    "fix_suggestion": "Specify the product name"
                }
                
        return {}


def test_pipeline_flow():
    print("\n── Test: LLM Pipeline Flow ────────────────────────────────")
    mock_client = MockLLMClient()
    processor = LLMDataProcessor(client=mock_client)
    
    raw_html = "<html><body>Buy now! Mock Product with 16GB RAM, 512GB Storage for $999. Free shipping!</body></html>"
    
    # Run pipeline
    valid_pairs = processor.process_raw_text(raw_html)
    
    # Assertions
    calls = mock_client.calls
    report("Called clean_text (text generation)", len([c for c in calls if c[0] == "text"]) == 1)
    
    json_calls = [c for c in calls if c[0] == "json"]
    report("Called extract, QA, and validation (json)", len(json_calls) >= 3)
    
    stats = processor.get_stats()
    report("Stats tracked processed count", stats["processed"] == 1)
    report("Stats tracked cleaned count", stats["cleaned"] == 1)
    report("Stats tracked QA generated count", stats["qa_generated"] == 2)
    report("Stats tracked validated_valid count", stats["validated_valid"] == 1)
    report("Stats tracked validated_invalid count", stats["validated_invalid"] == 1)
    
    report("Returned exactly 1 valid pair", len(valid_pairs) == 1)
    if valid_pairs:
        pair = valid_pairs[0]
        report("Valid pair kept", "RAM capacity" in pair["instruction"])
        report("Input was overwritten with clean text", pair["input"] == "Cleaned Mock Product: 16GB RAM, 512GB Storage, $999.")

def test_json_parsing_edge_cases():
    print("\n── Test: JSON Parsing Edge Cases ──────────────")
    from data_pipeline.llm_client import LLMClient
    
    class MockRequestsClient(LLMClient):
        def __init__(self, raw_resp):
            self.raw_resp = raw_resp
            super().__init__(api_key="mock")
            
        def _make_request(self, messages, model, temperature=0.7, max_tokens=1000, response_format=None):
            return self.raw_resp
            
    # Markdown wrapped JSON
    c1 = MockRequestsClient("```json\n{\"test\": 123}\n```")
    j1 = c1.generate_json("sys", "user")
    report("Parsed markdown-wrapped JSON", j1.get("test") == 123)
    
    # Raw JSON with leading/trailing spaces
    c2 = MockRequestsClient("   {\"test\": 456}   ")
    j2 = c2.generate_json("sys", "user")
    report("Parsed padded JSON", j2.get("test") == 456)
    
    # Invalid JSON
    c3 = MockRequestsClient("Not a json at all")
    try:
        c3.generate_json("sys", "user")
        report("Failed on invalid JSON", False)
    except ValueError:
        report("Caught Invalid JSON properly", True)


def main():
    print("\n" + "=" * 60)
    print("  LLM DATA PROCESSOR TESTS")
    print("=" * 60)

    test_pipeline_flow()
    test_json_parsing_edge_cases()

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
