import json
from typing import List, Dict, Any, Optional
from .llm_client import LLMClient

class LLMDataProcessor:
    """
    Implements a 5-stage LLM-based data cleaning and instruction generation pipeline
    for e-commerce product pages.
    """
    
    # 1. Cleaning Prompt
    CLEAN_SYSTEM = """You are a dataset compression system.
Remove:
- repeated product listings
- advertisement sentences
- delivery notices
- buying suggestions

Keep only factual product information.
Return a cleaned version of the text under 200 tokens."""

    # 2. Extraction Prompt
    EXTRACT_SYSTEM = """You are an information extraction system.
Extract product information from the given text and return structured JSON.

Fields to extract:
- product_name
- brand
- ram
- storage
- camera
- battery
- price
- discount
- rating
- delivery_date

Rules:
1. Only extract information that appears in the text.
2. If a field is missing, return null.
3. Do not invent data.

Output strictly in JSON format matching the fields above."""

    # 3. QA Generation Prompt
    QA_SYSTEM = """You are generating instruction tuning data for an LLM.

From the provided product text create 3 high quality QA examples.

Rules:
- Questions must be answerable ONLY from the text
- Avoid vague questions
- Answers must be short and factual
- Do not hallucinate

Return format MUST be a JSON array of objects:
{
 "pairs": [
  {
   "instruction": "",
   "input": "",
   "output": ""
  }
 ]
}"""

    # 4. Validation Prompt
    VALIDATE_SYSTEM = """You are a dataset validator for LLM training.

Analyze the following training example and check for issues:
- hallucinated output
- vague instruction
- missing answer in input
- excessive noise
- duplicated content

Return strictly in JSON format:
{
 "is_valid": true/false,
 "problems": ["problem1", "problem2"],
 "fix_suggestion": "suggestion if any, else null"
}"""

    def __init__(self, client: Optional[LLMClient] = None, model: str = "gpt-4o-mini"):
        self.client = client or LLMClient()
        self.model = model
        self._stats = {
            "processed": 0,
            "cleaned": 0,
            "extracted": 0,
            "qa_generated": 0,
            "validated_valid": 0,
            "validated_invalid": 0,
            "errors": 0
        }

    def process_raw_text(self, raw_text: str) -> List[Dict[str, str]]:
        """
        Run the full 5-stage pipeline on raw scraped text.
        Returns a list of valid instruction-input-output pairs.
        """
        self._stats["processed"] += 1
        valid_pairs = []
        
        try:
            # Stage 1: Clean & Compress
            clean_text = self._clean_text(raw_text)
            self._stats["cleaned"] += 1
            
            # Stage 2: Extract Structured Data (Optional for context, but requested by user rules)
            _ = self._extract_structured(clean_text)
            self._stats["extracted"] += 1
            
            # Stage 3: Generate QA Pairs
            qa_pairs = self._generate_qa(clean_text)
            self._stats["qa_generated"] += len(qa_pairs)
            
            # Stage 4 & 5: Validate each pair
            for pair in qa_pairs:
                if self._validate_pair(pair):
                    # Force the exact input to be the cleaned text to match the user's requested format
                    pair["input"] = clean_text 
                    valid_pairs.append(pair)
                    self._stats["validated_valid"] += 1
                else:
                    self._stats["validated_invalid"] += 1
                    
        except Exception as e:
            print(f"Error in LLM processing pipeline: {str(e)}")
            self._stats["errors"] += 1
            raise e
            
        return valid_pairs

    def _clean_text(self, text: str) -> str:
        prompt = f"Raw HTML/Text to clean:\n\n{text[:4000]}" # Limit input length safely
        return self.client.generate_text(self.CLEAN_SYSTEM, prompt, model=self.model, temperature=0.1)

    def _extract_structured(self, clean_text: str) -> Dict[str, Any]:
        prompt = f"Text to extract from:\n\n{clean_text}"
        return self.client.generate_json(self.EXTRACT_SYSTEM, prompt, model=self.model, temperature=0.0)

    def _generate_qa(self, clean_text: str) -> List[Dict[str, str]]:
        prompt = f"Product text:\n\n{clean_text}"
        result = self.client.generate_json(self.QA_SYSTEM, prompt, model=self.model, temperature=0.7)
        return result.get("pairs", [])

    def _validate_pair(self, pair: Dict[str, str]) -> bool:
        # Construct the evaluation prompt
        prompt = f"Instruction: {pair.get('instruction', '')}\nInput: {pair.get('input', '')}\nOutput: {pair.get('output', '')}"
        try:
            val_result = self.client.generate_json(self.VALIDATE_SYSTEM, prompt, model=self.model, temperature=0.0)
            return val_result.get("is_valid", False)
        except Exception:
            # If validation fails to parse or errors, be conservative and reject
            return False

    def get_stats(self) -> Dict[str, int]:
        return self._stats
