"""
Instruct Formatter Module
=========================
Converts text chunks into instruction/output pairs formatted
for LLM fine-tuning (Alpaca, ChatML, ShareGPT).
"""

import json
import re
import os
from typing import List, Dict, Any, Optional


class InstructFormatter:
    """
    Converts text chunks into instruction-response pairs for LLM fine-tuning.

    Supports multiple output formats:
    - Alpaca: {"instruction": ..., "input": ..., "output": ...}
    - ChatML: {"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}
    - ShareGPT: {"conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}

    Example:
    --------
    >>> formatter = InstructFormatter(template='alpaca')
    >>> pairs = formatter.format_chunks(chunks, domain="medical research")
    >>> formatter.export_jsonl(pairs, "training_data.jsonl")
    """

    TEMPLATES = {
        "alpaca": {
            "keys": ["instruction", "input", "output"],
            "description": "Stanford Alpaca format",
        },
        "chatml": {"keys": ["messages"], "description": "OpenAI ChatML format"},
        "sharegpt": {
            "keys": ["conversations"],
            "description": "ShareGPT multi-turn format",
        },
    }


    # Instruction generation templates
    INSTRUCTION_TEMPLATES = [
        "Explain the key concepts of {domain} based on this text.",
        "Summarize the following information about {domain}.",
        "What are the main points discussed in this text about {domain}?",
        "Provide an overview of {domain} using the details below.",
        "Analyze the provided text and describe its relation to {domain}.",
    ] * 10

    ECOMMERCE_TEMPLATES_EXTRACTION = [
        "Extract product specifications for {product} in JSON.",
        "Provide a structured JSON output of the features for {product}.",
        "Parse the given text and return a JSON containing specs for {product}.",
        "Please extract all technical specifications for the item {product} as JSON.",
        "I need a JSON object with the product info for {product}.",
        "Can you pull the specs for {product} into a structured format?",
        "Format the details of {product} into a JSON dictionary.",
        "Identify the key specifications for {product} and return them in JSON.",
        "Extract brand, category, and price for {product} if available.",
        "Return the product data for {product} in a machine-readable JSON format."
    ] * 5

    ECOMMERCE_TEMPLATES_QA = [
        "What is the price and rating of {product}?",
        "Can you tell me the RAM and storage for {product}?",
        "List the display and camera details for {product}.",
        "Who is the seller for {product} and what is its availability?",
        "What are the main specifications missing for {product}?",
        "How much does {product} cost and what is its discount?",
        "Tell me about the battery and touch id features of {product}.",
        "What brand is {product} and what category does it belong to?",
        "Is there a discount available for {product}?",
        "What is the unified memory capacity of {product}?"
    ] * 5

    ECOMMERCE_TEMPLATES_SUMMARIZATION = [
        "Summarize {product} in two short factual lines.",
        "Give me a quick 2-line summary of {product}.",
        "Provide a brief overview of {product}.",
        "Summarize the key features and price of {product}.",
        "Write a short, concise description of {product} based on the text.",
        "Condense the information about {product} into a few sentences.",
        "What is {product} in a nutshell?",
        "Produce a quick summary of the product {product}.",
        "Briefly describe the product {product} and its main selling points.",
        "Give a concise summation of {product}."
    ] * 5

    ECOMMERCE_TEMPLATES_REASONING = [
        "Based on the specs, explain why someone might buy {product}.",
        "Compare the pros and cons of {product}.",
        "Analyze the features of {product} and provide a recommendation.",
        "Think step-by-step and tell me if {product} is a good value.",
        "Evaluate {product} based on its price and rating.",
        "What are the trade-offs of purchasing {product}?",
        "Provide a reasoned evaluation of {product}'s camera and battery.",
        "Would you recommend {product} for a power user? Explain your reasoning.",
        "Assess the overall quality of {product} using the provided context.",
        "Break down the advantages and disadvantages of {product}."
    ] * 5

    ECOMMERCE_NOISE_PATTERNS = [

        r"Up to \d+% back with.*?(?:card|pay)\b",
        r"(?:FREE|free)\s+delivery.*",
        r"\d+\s*-\s*\d+\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
        r"This item will be released on.*",
        r"Price,?\s*product\s*page",
        r"EMI\s+(?:available|starts?).*",
        r"(?:Save|Get)\s+extra.*?(?:coupon|offer|cashback).*",
        r"Fulfilled by.*",
        r"\b(?:Add to (?:Cart|Wishlist)|Buy Now)\b",
        r"Sponsored\b.*",
        r"See more product details",
        r"(?:₹|\$|USD|INR)\s*\n\s*(?:₹|\$|USD|INR)?\s*[\d,]+",  # duplicate price lines
        r"(?:Limited time deal|Lightning deal|Best Seller)\b.*",
        r"No Cost EMI\b.*",
        r"Offers?\b.*",
    ]

    EXTRACTION_SCHEMA_KEYS = [
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

    # Follow-up instruction templates for Q&A extraction
    QA_TEMPLATES = [
        ("What is {topic}?", "{answer}"),
        ("Extract the exact detail for {topic}.", "{answer}"),
        ("List facts about {topic} from the product data.", "{answer}"),
        ("Describe {topic} briefly.", "{answer}"),
        ("Provide a concise value for {topic}.", "{answer}"),
    ]

    QA_TEMPLATES_GENERAL = [
        ("What is {topic}?", "{answer}"),
        ("Explain {topic} briefly.", "{answer}"),
        ("Summarize {topic} in one or two lines.", "{answer}"),
        ("What does the text say about {topic}?", "{answer}"),
        ("Why is {topic} relevant here?", "{answer}"),
    ]

    GENERAL_STOPWORDS = {
        "the",
        "and",
        "from",
        "with",
        "that",
        "this",
        "based",
        "text",
        "about",
        "what",
        "are",
        "main",
        "points",
        "into",
        "using",
        "provide",
        "describe",
        "explain",
        "concept",
        "concepts",
        "general",
    }

    def __init__(self, template: str = "alpaca", system_prompt: Optional[str] = None):
        """
        Initialize the formatter.

        Parameters
        ----------
        template : str
            Output format: 'alpaca', 'chatml', or 'sharegpt'.
        system_prompt : str, optional
            System prompt for ChatML/ShareGPT formats.
        """
        if template not in self.TEMPLATES:
            raise ValueError(
                f"Unknown template '{template}'. "
                f"Choose from: {list(self.TEMPLATES.keys())}"
            )

        self.template = template
        self.system_prompt = system_prompt or (
            "You are a helpful, knowledgeable assistant. "
            "Provide accurate, detailed answers based on the given context."
        )
        self._stats = {
            "total_chunks": 0,
            "total_pairs": 0,
            "template": template,
            "avg_instruction_length": 0,
            "avg_response_length": 0,
        }

    def format_chunks(
        self,
        chunks: List[Dict[str, Any]],
        domain: str = "general",
        generate_qa: bool = True,
        pairs_per_chunk: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Convert chunks into instruction-response pairs.

        Parameters
        ----------
        chunks : list of dict
            Text chunks from TextChunker.
        domain : str
            Domain label for instruction generation (e.g., 'medical', 'legal').
        generate_qa : bool
            If True, also generate Q&A pairs from chunk content.
        pairs_per_chunk : int
            Number of instruction pairs to generate per chunk.

        Returns
        -------
        list of dict
            Formatted training pairs.
        """
        all_pairs = []
        self._stats["total_chunks"] = len(chunks)

        for chunk in chunks:
            text = chunk.get("text", "")
            if not text.strip() or len(text) < 50:
                continue

            is_ecommerce = chunk.get("source_type") == "ecommerce"

            if is_ecommerce:
                # Use e-commerce-specific generation for product data
                import random
                clean_text = self._clean_ecommerce_text(text, keep_noise=False)
                product_meta = (chunk.get("metadata") or {}).get("product_data", {})
                pairs = self._generate_ecommerce_pairs(
                    clean_text, product_meta, pairs_per_chunk
                )
            else:
                # Generic instruction-response generation
                pairs = self._generate_pairs(text, domain, pairs_per_chunk)
                if generate_qa:
                    qa_pairs = self._extract_qa_pairs(text, domain)
                    pairs.extend(qa_pairs)

            # Keep training data diverse and remove repeated prompts/answers.
            pairs = self._deduplicate_pairs(pairs, max_same_instruction=2)

            # Format to target template
            for pair in pairs:
                formatted = self._apply_template(pair)
                # Ensure no forbidden metadata fields are in the final JSON
                if "metadata" in formatted:
                    del formatted["metadata"]
                all_pairs.append(formatted)

        all_pairs = self._deduplicate_formatted_pairs(all_pairs)

        # Update stats
        self._stats["total_pairs"] = len(all_pairs)
        if all_pairs:
            inst_lens = []
            resp_lens = []
            for p in all_pairs:
                i, r = self._extract_instruction_response(p)
                if i:
                    inst_lens.append(len(i))
                if r:
                    resp_lens.append(len(r))
            if inst_lens:
                self._stats["avg_instruction_length"] = sum(inst_lens) / len(inst_lens)
            if resp_lens:
                self._stats["avg_response_length"] = sum(resp_lens) / len(resp_lens)

        return all_pairs

    def _generate_pairs(
        self, text: str, domain: str, n_pairs: int
    ) -> List[Dict[str, str]]:
        """Generate instruction-response pairs from text."""
        pairs = []

        for i in range(min(n_pairs, len(self.INSTRUCTION_TEMPLATES))):
            tmpl = self.INSTRUCTION_TEMPLATES[i]
            if domain.lower() in ("general", ""):
                instruction = tmpl.replace(" of {domain}", "").replace(" about {domain}", "").replace(" to {domain}", "").replace("{domain}", "the topic")
            else:
                instruction = tmpl.format(domain=domain)

            # Create a summarized response from the text
            response = self._create_response(text, instruction)

            pairs.append(
                {"instruction": instruction, "input": text, "output": response}
            )

        return pairs

    def _generate_ecommerce_pairs(
        self, text: str, product_meta: dict, n_pairs: int
    ) -> List[Dict[str, str]]:
        import random
        import json
        pairs = []
        title = product_meta.get("title", "this product")
        short_title = title[:80] if len(title) > 80 else title
        structured = self._build_structured_product_output(product_meta, text)
        extraction_payload = self._structured_to_extraction_payload(structured)
        features = structured.get("features", [])
        reviews = structured.get("review_snippets", [])

        # Input mutation: Randomly shuffle attributes in the input text block or inject HTML noise
        mutated_text = text
        
        # Determine number of tasks per distribution
        # Extraction: 40%, QA: 30%, Summarization: 20%, Reasoning: 10%
        tasks_dist = []
        for _ in range(n_pairs):
            r = random.random()
            if r < 0.4: tasks_dist.append("EXTRACTION")
            elif r < 0.7: tasks_dist.append("QA")
            elif r < 0.9: tasks_dist.append("SUMMARIZATION")
            else: tasks_dist.append("REASONING")
            
        for task_type in tasks_dist:
            instruction = ""
            output = ""
            
            if task_type == "EXTRACTION":
                tmpl = random.choice(self.ECOMMERCE_TEMPLATES_EXTRACTION)
                instruction = tmpl.format(product=short_title)

                # Occasionally append explicit warranty requirement.
                if random.random() < 0.2:
                    instruction += " Extract the warranty."

                requested_fields = self._requested_extraction_fields_from_instruction(
                    instruction
                )
                output = json.dumps(
                    self._apply_extraction_subset(extraction_payload, requested_fields),
                    ensure_ascii=False,
                )
                    
            elif task_type == "QA":
                tmpl = random.choice(self.ECOMMERCE_TEMPLATES_QA)
                instruction = tmpl.format(product=short_title)
                lower_tmpl = tmpl.lower()
                
                if "price" in lower_tmpl and "rating" in lower_tmpl:
                    payload = self._pick_fields(structured, ["product_name", "price", "rating"])
                    output = json.dumps(payload, ensure_ascii=False) if payload else "Price and rating not found."
                elif "ram and storage" in lower_tmpl:
                    payload = self._pick_fields(structured, ["product_name", "unified_memory", "ram", "storage"])
                    output = json.dumps(payload, ensure_ascii=False) if payload else "RAM and storage not found."
                elif "display and camera" in lower_tmpl:
                    payload = self._pick_fields(structured, ["display", "camera"])
                    output = json.dumps(payload, ensure_ascii=False) if payload else "Display and camera details not found."
                elif "seller" in lower_tmpl or "availability" in lower_tmpl:
                    payload = self._pick_fields(structured, ["seller", "availability"])
                    output = json.dumps(payload, ensure_ascii=False) if payload else "Seller and availability not found."
                elif "discount" in lower_tmpl:
                    payload = self._pick_fields(structured, ["price", "discount"])
                    output = json.dumps(payload, ensure_ascii=False) if payload else "Discount information not found."
                elif "battery" in lower_tmpl and "touch id" in lower_tmpl:
                    payload = self._pick_fields(structured, ["battery", "touch_id"])
                    output = json.dumps(payload, ensure_ascii=False) if payload else "Battery and Touch ID details not found."
                elif "brand" in lower_tmpl and "category" in lower_tmpl:
                    payload = self._pick_fields(structured, ["product_name", "brand", "category"])
                    output = json.dumps(payload, ensure_ascii=False) if payload else "Brand and category not found."
                elif "unified memory" in lower_tmpl:
                    payload = self._pick_fields(structured, ["unified_memory", "ram"])
                    output = json.dumps(payload, ensure_ascii=False) if payload else "Unified memory capacity not found."
                elif "missing" in lower_tmpl:
                    core_missing = []
                    core_fields = [
                        ("brand", "brand"),
                        ("category", "category"),
                        ("product_name", "model"),
                        ("price", "price"),
                        ("storage", "storage"),
                        ("display", "display"),
                        ("camera", "camera"),
                        ("processor", "processor"),
                        ("warranty", "warranty"),
                    ]
                    for key, label in core_fields:
                        if not structured.get(key):
                            core_missing.append(label)
                    if core_missing:
                        output = (
                            "The context does not provide information about "
                            + ", ".join(core_missing)
                            + "."
                        )
                    else:
                        optional_missing = []
                        optional_fields = [
                            ("ram", "ram"),
                            ("unified_memory", "unified memory"),
                            ("battery", "battery"),
                            ("touch_id", "touch id"),
                            ("seller", "seller"),
                            ("availability", "availability"),
                            ("discount", "discount"),
                            ("weight", "weight"),
                        ]
                        for key, label in optional_fields:
                            if not structured.get(key):
                                optional_missing.append(label)
                        if optional_missing:
                            output = (
                                "The context does not provide information about "
                                + ", ".join(optional_missing)
                                + "."
                            )
                        else:
                            output = "The context provides the main specifications available in the input."
                else:
                    # Generic fallback: basic identity fields.
                    payload = self._pick_fields(structured, ["product_name", "brand", "category", "price"])
                    output = json.dumps(payload, ensure_ascii=False) if payload else "Requested information not found."

            elif task_type == "SUMMARIZATION":
                tmpl = random.choice(self.ECOMMERCE_TEMPLATES_SUMMARIZATION)
                instruction = tmpl.format(product=short_title)
                
                line1_parts = [structured.get("product_name", "")]
                if structured.get("brand"):
                    line1_parts.append(f"by {structured.get('brand')}")
                tech_bits = []
                if structured.get("storage"):
                    tech_bits.append(str(structured["storage"]))
                if structured.get("display"):
                    tech_bits.append(str(structured["display"]))
                if structured.get("camera"):
                    tech_bits.append(str(structured["camera"]))
                if tech_bits:
                    line1_parts.append("• " + ", ".join(tech_bits[:2]))

                line2_parts = []
                if structured.get("processor"):
                    line2_parts.append(f"Processor: {structured['processor']}")
                if structured.get("price"):
                    line2_parts.append(f"Price: {structured['price']}")
                if structured.get("rating"):
                    line2_parts.append(f"Rating: {structured['rating']}/5")

                line1 = " ".join(p for p in line1_parts if p).strip()
                line2 = " | ".join(line2_parts).strip()
                output = "\\n".join([p for p in [line1, line2] if p]).strip()
                
            elif task_type == "REASONING":
                tmpl = random.choice(self.ECOMMERCE_TEMPLATES_REASONING)
                instruction = tmpl.format(product=short_title)
                
                # Keep reasoning concise and avoid exposing chain-of-thought tags.
                analysis = []
                if rating := structured.get("rating"):
                    analysis.append(
                        f"Rating is {rating}, which indicates generally positive buyer feedback."
                    )
                if price := structured.get("price"):
                    analysis.append(f"Listed price is {price}.")
                    
                pros_cons = self._build_pros_cons(product_meta, features, structured)
                if pros_cons:
                    analysis.append("Pros/cons are supported by the extracted specs.")
                analysis_text = " ".join(analysis).strip()
                if not analysis_text:
                    analysis_text = "Available specs are limited."
                recommendation = (
                    json.dumps(pros_cons, ensure_ascii=False)
                    if pros_cons
                    else "Insufficient data to recommend."
                )
                output = f"Analysis: {analysis_text}\\nRecommendation: {recommendation}"

            output = self._normalize_ecommerce_output(output)
            output = self._limit_output(output, max_chars=1000)

            if output and len(output.strip()) > 10:
                pairs.append({
                    "instruction": instruction,
                    "input": mutated_text,
                    "output": output
                })

        return pairs

    def _clean_ecommerce_text(self, text: str, keep_noise: bool = False) -> str:
        """
        Remove promotional noise, duplicate prices, delivery info, and ad copy
        from e-commerce text before using it for instruction generation.
        """
        if keep_noise:
            # Output messy formatting and raw elements intentionally
            return text
            
        cleaned = text
        for pattern in self.ECOMMERCE_NOISE_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Collapse multiple blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        # Remove lines that are just currency symbols or short numbers
        lines = cleaned.split("\n")
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are just price fragments
            if re.match(r"^[₹$€£]?\s*[\d,]+\.?\d*$", stripped):
                continue
            # Skip very short lines (likely UI artifacts)
            if 0 < len(stripped) < 3:
                continue
            filtered_lines.append(line)

        cleaned = "\n".join(filtered_lines).strip()
        # Collapse whitespace
        cleaned = re.sub(r"  +", " ", cleaned)
        cleaned = re.sub(r"\b(off)(?:\s+off)+\b", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[|]{2,}", "|", cleaned)
        cleaned = re.sub(r"[.]{3,}", "...", cleaned)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        return cleaned

    def _pick_fields(self, data: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        """Return a compact dict with non-empty values for selected keys."""
        picked = {}
        for key in keys:
            val = data.get(key)
            if val in (None, "", [], {}):
                continue
            picked[key] = val
        return picked

    def _structured_to_extraction_payload(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        """Map structured product fields to canonical extraction schema."""
        def _text(v: Any) -> Optional[str]:
            if v in (None, "", [], {}):
                return None
            return str(v).strip() or None

        brand = _text(structured.get("brand"))
        product_name = _text(structured.get("product_name"))
        model = product_name
        if brand and product_name and product_name.lower().startswith((brand + " ").lower()):
            model = product_name[len(brand) :].strip()
            model = model or product_name

        rear = _text(
            structured.get("rear_camera")
            or structured.get("camera_rear")
            or structured.get("camera")
        )
        front = _text(
            structured.get("front_camera")
            or structured.get("camera_front")
        )
        price = _text(structured.get("price"))
        price_inr = None
        if price:
            cleaned = re.sub(r"[^\d.]", "", price)
            if cleaned:
                try:
                    price_inr = int(round(float(cleaned)))
                except Exception:
                    price_inr = None

        rating_val = structured.get("rating")
        rating = None
        if rating_val not in (None, "", [], {}):
            try:
                rating = round(float(str(rating_val).strip()), 1)
            except Exception:
                rating = None

        review_count = structured.get("review_count")
        if review_count in (None, "", [], {}):
            review_count = structured.get("reviews_count")
        if review_count in (None, "", [], {}):
            review_count = structured.get("reviews")
        if review_count not in (None, "", [], {}):
            try:
                review_count = int(str(review_count).replace(",", "").strip())
            except Exception:
                review_count = None
        else:
            review_count = None

        return {
            "brand": brand,
            "category": _text(structured.get("category")),
            "model": model,
            "price": price,
            "price_inr": price_inr,
            "ram": _text(structured.get("ram") or structured.get("unified_memory")),
            "storage": _text(structured.get("storage")),
            "processor": _text(structured.get("processor") or structured.get("chip")),
            "display": _text(structured.get("display")),
            "os": _text(structured.get("os")),
            "rating": rating,
            "review_count": review_count,
            "rear_camera": rear,
            "front_camera": front,
            "warranty": _text(structured.get("warranty")),
        }

    def _requested_extraction_fields_from_instruction(self, instruction: str) -> set:
        """Infer requested extraction fields from instruction text."""
        inst = (instruction or "").lower()
        req = set()

        if any(
            k in inst
            for k in [
                "all technical specifications",
                "all available",
                "key specifications",
                "structured json output",
                "structured json",
                "machine-readable",
                "product info",
                "product data",
                "json dictionary",
                "json containing specs",
                "specifications for",
                "specifications in json",
                "features for",
            ]
        ):
            return set(self.EXTRACTION_SCHEMA_KEYS)

        field_hints = {
            "brand": ["brand"],
            "category": ["category"],
            "model": ["model", "product name", "name"],
            "price": ["price", "cost"],
            "price_inr": ["numeric price", "price inr", "price_inr"],
            "ram": ["ram", "memory", "unified memory"],
            "storage": ["storage", "rom"],
            "os": ["operating system", "os", "windows", "mac os", "macos"],
            "display": ["display", "screen"],
            "rating": ["rating"],
            "review_count": ["review count", "number of reviews", "reviews"],
            "rear_camera": ["rear camera"],
            "front_camera": ["front camera"],
            "processor": ["processor", "chip"],
            "warranty": ["warranty"],
        }
        for field, hints in field_hints.items():
            if any(h in inst for h in hints):
                req.add(field)
        if "camera" in inst and "rear camera" not in inst and "front camera" not in inst:
            req.update({"rear_camera", "front_camera"})
        if not req:
            return set(self.EXTRACTION_SCHEMA_KEYS)
        return req

    def _apply_extraction_subset(self, payload: Dict[str, Any], requested: set) -> Dict[str, Any]:
        """Keep full schema keys and use null for non-requested fields."""
        req = set(requested or set())
        full = set(self.EXTRACTION_SCHEMA_KEYS)
        if not req or req == full:
            return {k: payload.get(k, None) for k in self.EXTRACTION_SCHEMA_KEYS}
        return {
            k: (payload.get(k, None) if k in req else None)
            for k in self.EXTRACTION_SCHEMA_KEYS
        }

    def _extract_pattern(self, text: str, pattern: str) -> str:
        """Extract first regex group from text, else empty string."""
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return ""
        groups = [g for g in match.groups() if g]
        return groups[0].strip() if groups else match.group(0).strip()

    def _build_structured_product_output(
        self, product_meta: Dict[str, Any], text: str
    ) -> Dict[str, Any]:
        """Build concise structured product JSON fields for training output."""
        structured: Dict[str, Any] = {}

        title = self._clean_ecommerce_text(product_meta.get("title", "")).strip()
        if title:
            structured["product_name"] = title
        if product_meta.get("brand"):
            structured["brand"] = self._clean_ecommerce_text(str(product_meta.get("brand")))
        elif title:
            # Heuristic: first token of the title is the brand.
            first_token = title.split()[0]
            if first_token:
                structured["brand"] = first_token
        if product_meta.get("category"):
            structured["category"] = self._clean_ecommerce_text(str(product_meta.get("category")))
        else:
            # Simple heuristic category from title/text.
            lowered_all = f"{title} {text}".lower()
            if any(tok in lowered_all for tok in ["iphone", "phone", "smartphone"]):
                structured["category"] = "smartphone"
        if product_meta.get("availability"):
            structured["availability"] = self._clean_ecommerce_text(str(product_meta.get("availability")))
        if product_meta.get("seller"):
            structured["seller"] = self._clean_ecommerce_text(str(product_meta.get("seller")))

        currency = product_meta.get("currency", "INR")
        if product_meta.get("price") is not None:
            structured["price"] = f"{currency} {float(product_meta['price']):,.2f}"
        if product_meta.get("original_price") is not None:
            structured["original_price"] = (
                f"{currency} {float(product_meta['original_price']):,.2f}"
            )
        if product_meta.get("discount"):
            discount = self._clean_ecommerce_text(str(product_meta["discount"]))
            discount = re.sub(
                r"\b(off)(?:\s+off)+\b", r"\1", discount, flags=re.IGNORECASE
            )
            structured["discount"] = discount

        if product_meta.get("rating") is not None:
            structured["rating"] = product_meta.get("rating")
        if product_meta.get("reviews_count") is not None:
            structured["reviews_count"] = product_meta.get("reviews_count")

        features = []
        for feat in product_meta.get("features", [])[:8]:
            cleaned = self._clean_ecommerce_text(str(feat))
            if cleaned:
                features.append(cleaned)
        if features:
            structured["features"] = features

        reviews = []
        for rev in product_meta.get("reviews", [])[:3]:
            cleaned = self._clean_ecommerce_text(str(rev))
            if cleaned:
                reviews.append(cleaned[:180])
        if reviews:
            structured["review_snippets"] = reviews

        blob = " ".join(
            [
                title,
                self._clean_ecommerce_text(str(product_meta.get("description", ""))),
                " ".join(features),
                self._clean_ecommerce_text(text),
            ]
        )

        # --- RAM extraction (handles "6GB RAM", "16 GB RAM", first number in "12GB+512GB") ---
        unified_memory = self._extract_pattern(
            blob, r"(\d+\s?(?:GB|TB))\s+(?:Unified\s+Memory|RAM)"
        )
        if unified_memory:
            if "unified memory" in blob.lower():
                structured["unified_memory"] = unified_memory
            else:
                structured["ram"] = unified_memory
        if "ram" not in structured and "unified_memory" not in structured:
            # Fallback: try "12GB+512GB" format (first number is RAM)
            combo = self._extract_pattern(blob, r"(\d+\s?GB)\s*\+\s*\d+\s?(?:GB|TB)")
            if combo:
                structured["ram"] = combo

        # --- Storage extraction ---
        storage = self._extract_pattern(
            blob, r"(\d+\s?(?:GB|TB)\s*(?:SSD|HDD|Storage))"
        )
        if not storage:
            # Fallback: second number in "12GB+512GB" format
            storage = self._extract_pattern(blob, r"\d+\s?GB\s*\+\s*(\d+\s?(?:GB|TB))")
        if not storage:
            # Fallback: "256GB)" or "256 GB" near end of a parenthesized spec
            storage = self._extract_pattern(blob, r"(?:,\s*)(\d{3,4}\s?(?:GB|TB))(?:\s*\)|,|\s)")
        if storage:
            structured["storage"] = storage

        # --- OS extraction ---
        os_name = self._extract_pattern(
            blob,
            r"((?:Windows|Mac\s*OS|macOS|Ubuntu|Linux|Chrome\s*OS)[^,\n|]*(?:Operating System|OS)?)",
        )
        if os_name:
            structured["os"] = os_name

        # --- Display extraction (stop at | and newlines, not just ,.;) ---
        display = self._extract_pattern(
            blob,
            r"((?:\d{1,2}(?:\.\d+)?\s*(?:cm|inch|\"|″|in)?\s*(?:\([^)]*\)\s*)?)?(?:Liquid Retina|Super Retina|Retina|AMOLED|OLED|FHD|QHD|Full\s*HD\+?|Quad\s*HD\+?|HD\+?|LED)[^,.;|\n]*(?:Display|Screen)?)",
        )
        if display:
            # Trim trailing noise — stop at first pipe, newline, or price marker
            display = re.split(r"\s*[|\n]", display)[0].strip()
            structured["display"] = display

        # --- Camera extraction (stop at | and newlines) ---
        camera = self._extract_pattern(
            blob, r"((?:\d{3,4}p|\d+\s?MP)[^,.;|\n]*Camera|FaceTime\s+HD\s+Camera)"
        )
        if camera:
            camera = re.split(r"\s*[|\n]", camera)[0].strip()
            structured["camera"] = camera

        # --- Battery extraction ---
        battery = self._extract_pattern(blob, r"(\d{3,5}\s?mAh)")
        if battery:
            structured["battery"] = battery

        if "touch id" in blob.lower():
            structured["touch_id"] = True

        # --- Warranty extraction ---
        warranty = self._extract_pattern(
            blob, r"(\d+\s*year[^.,\n]*warranty)"
        )
        if warranty:
            structured["warranty"] = warranty

        # --- Weight extraction (e.g., 187 g, 0.2 kg) ---
        weight = self._extract_pattern(
            blob, r"(\d+(?:\.\d+)?\s*(?:g(?!b)|kg))\b"
        )
        if weight:
            structured["weight"] = weight

        # Fallback: if rating/reviews_count missing, parse from blob.
        if "rating" not in structured:
            m_rating = re.search(r"Rating:\s*([\d\.]+)", blob, flags=re.IGNORECASE)
            if m_rating:
                try:
                    structured["rating"] = float(m_rating.group(1))
                except ValueError:
                    pass

        # Fallback: if reviews_count missing but reviews like "(213,340 reviews)" appear in text
        if "reviews_count" not in structured:
            m = re.search(r"\(([\d,]+)\s+reviews?\)", blob, flags=re.IGNORECASE)
            if m:
                try:
                    structured["reviews_count"] = int(m.group(1).replace(",", ""))
                except ValueError:
                    pass

        return structured

    def _build_pros_cons(
        self, product_meta: Dict[str, Any], features: List[str], structured: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create concise fact-based pros/cons JSON object."""
        pros: List[str] = []
        cons: List[str] = []

        # Prefer normalized rating from structured, then fall back to raw meta
        rating = structured.get("rating", product_meta.get("rating"))
        try:
            rating_val = float(rating) if rating is not None else None
        except (TypeError, ValueError):
            rating_val = None
        if rating_val is not None:
            if rating_val >= 4.0:
                pros.append(f"High rating ({rating_val}/5)")
            elif rating_val < 3.5:
                cons.append(f"Lower rating ({rating_val}/5)")

        discount = product_meta.get("discount") or structured.get("discount")
        if discount:
            clean_discount = re.sub(
                r"\b(off)(?:\s+off)+\b", r"\1", str(discount), flags=re.IGNORECASE
            )
            pros.append(f"Discount available ({clean_discount})")

        for feat in features[:3]:
            pros.append(feat[:80])

        reviews_count = structured.get("reviews_count", product_meta.get("reviews_count"))
        if isinstance(reviews_count, (int, float)):
            if reviews_count < 50:
                cons.append("Limited review volume")
        else:
            # Unknown review volume – do not assert it's limited.
            pass

        payload: Dict[str, Any] = {}
        if pros:
            payload["pros"] = pros[:4]
        if cons:
            payload["cons"] = cons[:3]
        return payload

    def _normalize_ecommerce_output(self, output: str) -> str:
        """Normalize noisy artifacts while preserving structured JSON outputs."""
        text = (output or "").strip()
        if not text:
            return ""

        stripped = text.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            return text

        text = self._clean_ecommerce_text(text)
        text = re.sub(r"\b(off)(?:\s+off)+\b", r"\1", text, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def _limit_output(self, output: str, max_chars: int = 700) -> str:
        """Keep output concise to reduce token usage in fine-tuning data."""
        text = (output or "").strip()
        if len(text) <= max_chars:
            return text
        trimmed = text[:max_chars]
        cutoff = trimmed.rfind(" ")
        if cutoff > int(max_chars * 0.8):
            trimmed = trimmed[:cutoff]
        return trimmed.strip()

    def _deduplicate_pairs(
        self, pairs: List[Dict[str, str]], max_same_instruction: int = 2
    ) -> List[Dict[str, str]]:
        """Drop exact duplicates and cap repeated instruction forms."""
        deduped: List[Dict[str, str]] = []
        seen_exact = set()
        instruction_counts: Dict[str, int] = {}

        for pair in pairs:
            instruction = (pair.get("instruction") or "").strip()
            output = (pair.get("output") or "").strip()
            input_text = (pair.get("input") or "").strip()
            if not instruction or not output:
                continue

            inst_key = re.sub(r"\s+", " ", instruction.lower())
            exact_key = (
                inst_key,
                re.sub(r"\s+", " ", input_text.lower()),
                re.sub(r"\s+", " ", output.lower()),
            )
            if exact_key in seen_exact:
                continue
            if instruction_counts.get(inst_key, 0) >= max_same_instruction:
                continue

            seen_exact.add(exact_key)
            instruction_counts[inst_key] = instruction_counts.get(inst_key, 0) + 1
            deduped.append(pair)

        return deduped

    def _create_response(self, text: str, instruction: str) -> str:
        """
        Create a concise response from the text content.
        Prefers clean line/sentence summaries over raw verbatim chunks.
        """
        clean_text = self._clean_general_text(text)
        if not clean_text:
            return ""

        segments: List[str] = []
        for line in clean_text.splitlines():
            parts = re.split(r"(?<=[.!?])\s+|:\s+|;\s+", line)
            for part in parts:
                seg = re.sub(r"\s+", " ", part).strip(" -•\t")
                if not self._is_summary_candidate(seg):
                    continue
                segments.append(seg)

        if not segments:
            short_lines: List[str] = []
            for line in clean_text.splitlines():
                norm = re.sub(r"\s+", " ", line).strip(" -•\t")
                if not self._is_summary_candidate(norm):
                    continue
                short_lines.append(norm)
                if len(short_lines) >= 3:
                    break
            if short_lines:
                fallback = "Key concepts include: " + "; ".join(short_lines) + "."
                return self._limit_output(fallback, max_chars=420)
            flat = re.sub(r"\s+", " ", clean_text).strip()
            return self._limit_output(flat, max_chars=320)

        instruction_tokens = self._instruction_keywords(instruction)
        indicator_words = {
            "key",
            "main",
            "important",
            "architecture",
            "pipeline",
            "model",
            "security",
            "processing",
            "algorithm",
            "distributed",
            "infrastructure",
        }

        scored = []
        for idx, seg in enumerate(segments):
            lower = seg.lower()
            seg_tokens = set(re.findall(r"\b[a-z][a-z0-9_]{2,}\b", lower))
            keyword_hits = len(seg_tokens & instruction_tokens)
            indicator_hits = sum(1 for w in indicator_words if w in lower)
            length_bonus = 1.0 if 45 <= len(seg) <= 220 else 0.2
            position_bonus = 0.6 if idx < 2 else 0.0
            score = (keyword_hits * 2.0) + (indicator_hits * 1.2) + length_bonus + position_bonus
            scored.append((score, idx, seg))

        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

        selected: List[str] = []
        seen_norm = set()
        for _, _, seg in scored:
            norm = re.sub(r"\s+", " ", seg.lower())
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            selected.append(seg)
            if len(selected) >= 4:
                break

        if not selected:
            selected = segments[:3]

        inst_lower = (instruction or "").lower()
        summary_request = any(
            token in inst_lower
            for token in ["explain", "summarize", "main points", "overview", "key concepts"]
        )
        
        if summary_request:
            points: List[str] = []
            seen_points = set()
            for seg in selected[:3]:
                clean_point = re.sub(r'^\d+[\.\)]?\s*', '', seg.strip()).strip()
                if len(clean_point) < 20: continue
                # capitalize first letter
                clean_point = clean_point[0].upper() + clean_point[1:]
                # ensure ends with period
                if not clean_point.endswith('.'):
                    clean_point += '.'
                norm = clean_point.lower()
                if norm in seen_points:
                    continue
                seen_points.add(norm)
                points.append(clean_point)
            
            if not points:
                points = [selected[0].strip() + "."] if selected else ["The text provides detailed information on the subject."]
            
            response = "Key concepts discussed in the text include: " + " ".join(points)
        else:
            response = " ".join(selected[:3])

        response = re.sub(r"\s+", " ", response).strip()
        response = re.sub(r"\.{2,}", ".", response)
        return self._limit_output(response, max_chars=520)

    def _clean_general_text(self, text: str) -> str:
        """Remove common PDF/web noise artifacts while keeping semantic content."""
        raw = (text or "").replace("\r", "\n")
        raw = re.sub(r"https?://\S+", "", raw, flags=re.IGNORECASE)

        cleaned_lines: List[str] = []
        for line in raw.splitlines():
            line = re.sub(r"\s+", " ", line).strip(" \t-•")
            if not line:
                continue
            low_line = line.lower()
            if "datapipe (project:" in low_line:
                continue
            if re.fullmatch(r"\d+\s*/\s*\d+", line):
                continue
            if re.search(r"\b\d{2}/\d{2}/\d{4},\s*\d{1,2}:\d{2}\b", line):
                continue
            if re.search(r"\bpage\s*\d+\s*/\s*\d+\b", line, flags=re.IGNORECASE):
                continue
            # Drop isolated file-path lines from PDFs/code listings.
            if re.fullmatch(
                r"[\w./-]+\.(?:py|js|ts|tsx|html|css|md|json|yaml|yml|txt)",
                line,
                flags=re.IGNORECASE,
            ):
                continue
            # Repair tiny chopped leading fragments such as "re Guide", "g heavy".
            line = re.sub(r"^[a-z]{1,4}\s+(?=[A-Z])", "", line)
            cleaned_lines.append(line)

        # Merge wrapped PDF lines to reduce partial sentence fragments.
        merged: List[str] = []
        for line in cleaned_lines:
            if merged and self._should_merge_general_lines(merged[-1], line):
                merged[-1] = f"{merged[-1]} {line}"
            else:
                merged.append(line)

        deduped: List[str] = []
        seen = set()
        for line in merged:
            norm = line.lower()
            if norm in seen:
                continue
            seen.add(norm)
            deduped.append(line)

        return "\n".join(deduped).strip()

    def _should_merge_general_lines(self, prev: str, curr: str) -> bool:
        """Heuristic: merge PDF-wrapped lines that belong to one sentence/phrase."""
        prev = (prev or "").strip()
        curr = (curr or "").strip()
        if not prev or not curr:
            return False
        if len(prev) > 140:
            return False
        if prev.endswith((".", "!", "?", ":", ";")):
            return False
        if re.match(r"^\d+[\).\s-]", curr):
            return False
        if re.match(r"^(Part|Section|Chapter)\b", curr, flags=re.IGNORECASE):
            return False
        if re.match(r"^(Abstract|Conclusion|Introduction)\b", curr, flags=re.IGNORECASE):
            return False
        if curr.istitle() and curr.count(" ") <= 2:
            return False
        if re.match(r"^[#@]", curr):
            return False
        if not re.search(r"[A-Za-z]", prev) or not re.search(r"[A-Za-z]", curr):
            return False
        return True

    def _is_summary_candidate(self, seg: str) -> bool:
        """Filter out low-signal/chopped/code-like snippets for summaries."""
        seg = (seg or "").strip()
        if len(seg) < 25:
            return False
        low = seg.lower()
        if "http://" in low or "https://" in low:
            return False
        if "@app.route" in low:
            return False
        if re.search(r"\bdef\s+\w+\s*\(", low):
            return False
        if re.match(r"^(post|get|put|delete|patch)\s+/\S+", low):
            return False
        if "mapping every source file" in low:
            return False
        if low.endswith("source file in the project:"):
            return False
        if seg.endswith(":"):
            return False
        # Avoid likely chopped leading fragments and lowercase-start artifacts.
        if seg and not (seg[0].isupper() or seg[0].isdigit()):
            return False
        # Ignore code/file-only snippets.
        if re.search(r"\b(?:app\.py|model_trainer\.py|llm_pipeline\.py|dashboard\.html)\b", low):
            return False
        if re.fullmatch(r"[\w./-]+\.(?:py|js|ts|tsx|html|css|md|json|yaml|yml|txt)", seg, flags=re.IGNORECASE):
            return False
        alpha_ratio = sum(ch.isalpha() for ch in seg) / max(len(seg), 1)
        if alpha_ratio < 0.45:
            return False
        return True

    def _clean_summary_point(self, seg: str) -> str:
        """Normalize a summary point into a clean, sentence-like fragment."""
        text = re.sub(r"\s+", " ", (seg or "")).strip(" -•\t")
        text = re.sub(r"\.{2,}", ".", text)
        text = text.rstrip(" :;,.")
        # Remove leading section numbering.
        text = re.sub(r"^\d+(?:\.\d+)*\s+", "", text)
        return text.strip()

    def _instruction_keywords(self, instruction: str) -> set:
        """Extract lightweight topical keywords from instruction text."""
        tokens = set(re.findall(r"\b[a-z][a-z0-9_]{2,}\b", (instruction or "").lower()))
        return {t for t in tokens if t not in self.GENERAL_STOPWORDS}

    def _extract_qa_pairs(self, text: str, domain: str) -> List[Dict[str, str]]:
        """Extract Q&A pairs from text using simple heuristics."""
        pairs = []

        # Clean text first (removes PDF/web noise and artifacts)
        clean_text = self._clean_general_text(text)

        # Extract potential topics from the cleaned text
        topics = self._extract_topics(clean_text)

        if not topics:
            return pairs

        used_instructions = set()
        qa_templates = self.QA_TEMPLATES_GENERAL
        segments = self._split_general_segments(clean_text)
        # Generate Q&A for top topics
        for idx, topic in enumerate(topics[:5]):
            # Find the sentence(s) that discuss this topic
            relevant = [
                s.strip()
                for s in segments
                if topic.lower() in s.lower() and len(s.strip()) > 20
            ]

            if relevant:
                answer = " ".join(relevant[:2])
                # Skip if the answer is too noisy (mostly numbers/symbols)
                alpha_ratio = sum(c.isalpha() for c in answer) / max(len(answer), 1)
                if alpha_ratio < 0.4:
                    continue
                # Avoid near-verbatim chunk echo for QA outputs.
                if len(answer) > int(len(clean_text) * 0.65):
                    answer = self._create_response(answer, f"Explain {topic} briefly.")
                qa_template = qa_templates[idx % len(qa_templates)]
                instruction = qa_template[0].format(topic=topic).strip()
                normalized_inst = re.sub(r"\s+", " ", instruction.lower())
                if normalized_inst in used_instructions:
                    continue
                used_instructions.add(normalized_inst)
                answer = self._limit_output(re.sub(r"\s+", " ", answer).strip(), max_chars=260)
                if len(answer) < 30 or len(answer.split()) < 5:
                    continue
                pairs.append(
                    {
                        "instruction": instruction,
                        "input": clean_text,
                        "output": qa_template[1].format(answer=answer).strip(),
                    }
                )

        return pairs

    def _split_general_segments(self, text: str) -> List[str]:
        """Split generic prose/PDF text into concise semantic segments."""
        parts: List[str] = []
        for line in (text or "").splitlines():
            line = re.sub(r"\s+", " ", line).strip(" -•\t")
            if not line:
                continue
            # Drop the split on : and ; to preserve natural sentence structure
            subparts = re.split(r"(?<=[.!?])\s+", line)
            for part in subparts:
                seg = re.sub(r"\s+", " ", part).strip(" -•\t")
                if len(seg) < 12:
                    continue
                parts.append(seg)
        return parts

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract potential topic phrases from text using simple heuristics.
        Looks for capitalized phrases and noun-like patterns.
        """
        candidates = []

        # Find capitalized multi-word phrases
        candidates.extend(
            re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
        )

        # Find phrases after "is", "are", "refers to", "means"
        candidates.extend(
            re.findall(
                r"(?:^|\.\s+)([A-Z][^.]{10,60}?)(?:\s+is\s+|\s+are\s+|\s+refers?\s+to)",
                text,
            )
        )

        # Pull common e-commerce spec tokens for better extraction diversity.
        spec_patterns = [
            r"\b(\d+\s?(?:GB|TB)\s+Unified\s+Memory)\b",
            r"\b(\d+\s?(?:GB|TB)\s+(?:SSD|HDD|Storage))\b",
            r"\b(Liquid Retina Display|Retina Display|AMOLED Display|OLED Display)\b",
            r"\b(\d{3,4}p\s+[A-Za-z ]*Camera|FaceTime HD Camera)\b",
            r"\b(Touch ID)\b",
            r"\b(Battery)\b",
            r"\b(Price|Discount|Availability|Rating)\b",
        ]
        for pattern in spec_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                candidates.append(match if isinstance(match, str) else match[0])

        topics = []
        seen = set()
        for candidate in candidates:
            topic = re.sub(r"\s+", " ", str(candidate)).strip()
            norm = topic.lower()
            if len(topic) < 3 or norm in seen:
                continue
            seen.add(norm)
            topics.append(topic)

        return topics[:8]

    def _apply_template(self, pair: Dict[str, str]) -> Dict[str, Any]:
        """Apply the selected template format."""
        if self.template == "alpaca":
            return self._format_alpaca(pair)
        elif self.template == "chatml":
            return self._format_chatml(pair)
        elif self.template == "sharegpt":
            return self._format_sharegpt(pair)
        else:
            return pair

    def _format_alpaca(self, pair: Dict[str, str]) -> Dict[str, Any]:
        """Format as Alpaca-style dict."""
        return {
            "instruction": pair["instruction"],
            "input": pair.get("input", ""),
            "output": pair["output"],
        }

    def _format_chatml(self, pair: Dict[str, str]) -> Dict[str, Any]:
        """Format as ChatML-style dict."""
        messages = [{"role": "system", "content": self.system_prompt}]

        user_content = pair["instruction"]
        if pair.get("input"):
            user_content += f"\n\n{pair['input']}"

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": pair["output"]})

        return {"messages": messages}

    def _format_sharegpt(self, pair: Dict[str, str]) -> Dict[str, Any]:
        """Format as ShareGPT-style dict."""
        conversations = []

        human_content = pair["instruction"]
        if pair.get("input"):
            human_content += f"\n\n{pair['input']}"

        conversations.append({"from": "human", "value": human_content})
        conversations.append({"from": "gpt", "value": pair["output"]})

        return {"conversations": conversations}

    def _extract_instruction_response(self, formatted: Dict[str, Any]) -> tuple:
        """Extract instruction and response text from any template format."""
        if self.template == "alpaca":
            return formatted.get("instruction", ""), formatted.get("output", "")
        elif self.template == "chatml":
            msgs = formatted.get("messages", [])
            inst = msgs[1]["content"] if len(msgs) > 1 else ""
            resp = msgs[2]["content"] if len(msgs) > 2 else ""
            return inst, resp
        elif self.template == "sharegpt":
            convs = formatted.get("conversations", [])
            inst = convs[0]["value"] if len(convs) > 0 else ""
            resp = convs[1]["value"] if len(convs) > 1 else ""
            return inst, resp
        return "", ""

    def _deduplicate_formatted_pairs(
        self, pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate pairs after template formatting."""
        deduped = []
        seen = set()

        for pair in pairs:
            inst, resp = self._extract_instruction_response(pair)
            key = (
                re.sub(r"\s+", " ", (inst or "").strip().lower()),
                re.sub(r"\s+", " ", (resp or "").strip().lower()),
            )
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append(pair)
        return deduped

    # ─── Export ───────────────────────────────────────────────────────────

    def export_jsonl(
        self,
        pairs: List[Dict[str, Any]],
        output_path: str,
        include_metadata: bool = False,
    ) -> str:
        """
        Export formatted pairs to JSONL file.

        Parameters
        ----------
        pairs : list of dict
            Formatted training pairs.
        output_path : str
            Path to write the JSONL file.
        include_metadata : bool
            If True, include metadata in each line.

        Returns
        -------
        str
            Path to the written file.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                if not include_metadata and "metadata" in pair:
                    row = {k: v for k, v in pair.items() if k != "metadata"}
                else:
                    row = pair
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return output_path

    def export_json(
        self,
        pairs: List[Dict[str, Any]],
        output_path: str,
        include_metadata: bool = False,
    ) -> str:
        """Export formatted pairs to JSON file (array format)."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if not include_metadata:
            clean_pairs = [
                {k: v for k, v in p.items() if k != "metadata"} for p in pairs
            ]
        else:
            clean_pairs = pairs

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean_pairs, f, indent=2, ensure_ascii=False)

        return output_path

    def get_stats(self) -> Dict[str, Any]:
        """Return formatting statistics."""
        return self._stats

    def print_summary(self) -> None:
        """Print a formatted summary."""
        stats = self._stats
        print("=" * 60)
        print("INSTRUCTION FORMATTING SUMMARY")
        print("=" * 60)
        print(f"\n📋 Template: {stats['template']}")
        print(f"📥 Chunks processed: {stats['total_chunks']}")
        print(f"📝 Pairs generated: {stats['total_pairs']}")

        if stats["total_pairs"] > 0:
            print("\n📊 Averages:")
            print(
                f"   • Instruction length: {stats['avg_instruction_length']:.0f} chars"
            )
            print(f"   • Response length: {stats['avg_response_length']:.0f} chars")

        print("=" * 60)
