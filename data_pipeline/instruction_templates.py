"""
Instruction templates for Layer B dataset synthesis.
"""

from __future__ import annotations

from typing import List


EXTRACTION_INSTRUCTIONS: List[str] = [
    "Extract brand, category, price, storage, display and camera from the product.",
    "Extract product specifications as JSON.",
    "Parse the product details and return JSON.",
    "Identify all available product attributes from the context in JSON.",
    "Return the product specifications in structured JSON format.",
]

QA_INSTRUCTIONS: List[str] = [
    "What is the price and warranty?",
    "What is the storage and rating?",
    "What are the display and chip details?",
    "What is the product name and brand?",
    "What are the rear and front camera specifications?",
]

SUMMARIZATION_INSTRUCTIONS: List[str] = [
    "Summarize the product.",
    "Provide a concise factual summary of this product.",
    "Write a short product summary from the given context.",
]

REASONING_INSTRUCTIONS: List[str] = [
    "Evaluate whether this product is good value.",
    "Think step-by-step and evaluate if this product is suitable for heavy users.",
    "Reason over the specifications and give a value recommendation.",
]


ALL_OUTPUT_FIELDS: List[str] = [
    "product_name",
    "brand",
    "price",
    "rating",
    "storage",
    "display",
    "camera_rear",
    "camera_front",
    "chip",
    "warranty",
]


TASK_DISTRIBUTION = {
    "extraction": 0.40,
    "qa": 0.30,
    "summarization": 0.20,
    "reasoning": 0.10,
}
