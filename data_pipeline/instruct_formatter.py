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
    ]

    # E-commerce-specific instruction templates
    ECOMMERCE_INSTRUCTION_TEMPLATES = [
        "Extract product specifications for {product} in JSON.",
        "Summarize {product} in two short factual lines.",
        "Extract price, discount, and rating fields for {product}.",
        "List key features of {product} as structured data.",
        "Summarize verified customer feedback for {product}.",
        "Extract brand and category for {product}.",
        "Provide pros and cons for {product} from available facts only.",
        "Extract availability and seller details for {product}.",
        "Extract RAM and storage for {product} if present.",
    ]

    # Noise patterns to strip from e-commerce text before instruction generation
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

    # Follow-up instruction templates for Q&A extraction
    QA_TEMPLATES = [
        ("What is {topic}?", "{answer}"),
        ("Extract the exact detail for {topic}.", "{answer}"),
        ("List facts about {topic} from the product data.", "{answer}"),
        ("Describe {topic} briefly.", "{answer}"),
        ("Provide a concise value for {topic}.", "{answer}"),
    ]

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
                clean_text = self._clean_ecommerce_text(text)
                product_meta = (chunk.get("metadata") or {}).get("product_data", {})
                pairs = self._generate_ecommerce_pairs(
                    clean_text, product_meta, pairs_per_chunk
                )
                if generate_qa:
                    qa_pairs = self._extract_qa_pairs(clean_text, domain)
                    pairs.extend(qa_pairs)
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
                formatted["metadata"] = {
                    "source": chunk.get("source", ""),
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "domain": domain,
                    "is_ecommerce": is_ecommerce,
                }
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
            instruction = self.INSTRUCTION_TEMPLATES[i].format(domain=domain)

            # Create a summarized response from the text
            response = self._create_response(text, instruction)

            pairs.append(
                {"instruction": instruction, "input": text, "output": response}
            )

        return pairs

    def _generate_ecommerce_pairs(
        self, text: str, product_meta: dict, n_pairs: int
    ) -> List[Dict[str, str]]:
        """
        Generate instruction-response pairs optimized for e-commerce product data.
        Uses structured product metadata instead of raw text extraction.
        """
        pairs = []
        title = product_meta.get("title", "this product")
        short_title = title[:80] if len(title) > 80 else title
        structured = self._build_structured_product_output(product_meta, text)
        compact_structured = json.dumps(structured, ensure_ascii=False)
        features = structured.get("features", [])
        reviews = structured.get("review_snippets", [])

        # Generate pairs using e-commerce templates
        templates_used = 0
        for tmpl in self.ECOMMERCE_INSTRUCTION_TEMPLATES:
            if templates_used >= n_pairs:
                break

            instruction = tmpl.format(product=short_title)
            lower_tmpl = tmpl.lower()
            output = ""

            # Build answer based on instruction type
            if "json" in lower_tmpl or "specification" in lower_tmpl:
                output = compact_structured
            elif "price" in lower_tmpl:
                payload = self._pick_fields(
                    structured,
                    ["product_name", "price", "original_price", "discount", "rating", "reviews_count"],
                )
                output = json.dumps(payload, ensure_ascii=False) if payload else ""
            elif "feature" in lower_tmpl and "structured data" in lower_tmpl:
                payload = self._pick_fields(
                    structured,
                    ["product_name", "unified_memory", "ram", "storage", "display", "camera", "battery", "touch_id", "features"],
                )
                output = json.dumps(payload, ensure_ascii=False) if payload else ""
            elif "feedback" in lower_tmpl:
                payload = self._pick_fields(
                    structured,
                    ["product_name", "rating", "reviews_count", "review_snippets"],
                )
                if payload:
                    output = json.dumps(payload, ensure_ascii=False)
                elif reviews:
                    output = " | ".join(reviews[:2])
            elif "brand and category" in lower_tmpl:
                payload = self._pick_fields(
                    structured,
                    ["product_name", "brand", "category"],
                )
                output = json.dumps(payload, ensure_ascii=False) if payload else ""
            elif "pros and cons" in lower_tmpl:
                payload = self._build_pros_cons(product_meta, features)
                output = json.dumps(payload, ensure_ascii=False) if payload else ""
            elif "availability and seller" in lower_tmpl:
                payload = self._pick_fields(
                    structured,
                    ["product_name", "availability", "seller"],
                )
                output = json.dumps(payload, ensure_ascii=False) if payload else ""
            elif "ram and storage" in lower_tmpl:
                payload = self._pick_fields(
                    structured,
                    ["product_name", "unified_memory", "ram", "storage"],
                )
                output = json.dumps(payload, ensure_ascii=False) if payload else ""
            elif "two short factual lines" in lower_tmpl:
                line1_parts = [
                    structured.get("product_name"),
                    f"by {structured.get('brand')}" if structured.get("brand") else "",
                ]
                line2_parts = []
                if structured.get("price"):
                    line2_parts.append(f"Price: {structured['price']}")
                if structured.get("rating"):
                    line2_parts.append(f"Rating: {structured['rating']}/5")
                if structured.get("availability"):
                    line2_parts.append(f"Availability: {structured['availability']}")
                line1 = " ".join(p for p in line1_parts if p).strip()
                line2 = " | ".join(line2_parts).strip()
                output = "\n".join([p for p in [line1, line2] if p]).strip()
            else:
                output = compact_structured

            output = self._normalize_ecommerce_output(output)
            output = self._limit_output(output, max_chars=700)

            if output and len(output.strip()) > 20:
                pairs.append(
                    {"instruction": instruction, "input": "", "output": output}
                )
                templates_used += 1

        return pairs

    def _clean_ecommerce_text(self, text: str) -> str:
        """
        Remove promotional noise, duplicate prices, delivery info, and ad copy
        from e-commerce text before using it for instruction generation.
        """
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
        if product_meta.get("category"):
            structured["category"] = self._clean_ecommerce_text(str(product_meta.get("category")))
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

        unified_memory = self._extract_pattern(
            blob, r"(\d+\s?(?:GB|TB))\s+(?:Unified\s+Memory|RAM)"
        )
        if unified_memory:
            if "unified memory" in blob.lower():
                structured["unified_memory"] = unified_memory
            else:
                structured["ram"] = unified_memory

        storage = self._extract_pattern(
            blob, r"(\d+\s?(?:GB|TB)\s*(?:SSD|HDD|Storage))"
        )
        if storage:
            structured["storage"] = storage

        display = self._extract_pattern(
            blob,
            r"((?:\d{1,2}(?:\.\d+)?\s*(?:cm|inch|\"|in)?)?\s*(?:Liquid Retina|Retina|AMOLED|OLED|FHD|QHD|LED)[^,.;]*)",
        )
        if display:
            structured["display"] = display

        camera = self._extract_pattern(
            blob, r"((?:\d{3,4}p|\d+\s?MP)[^,.;]*Camera|FaceTime\s+HD\s+Camera)"
        )
        if camera:
            structured["camera"] = camera

        battery = self._extract_pattern(blob, r"(\d{3,5}\s?mAh)")
        if battery:
            structured["battery"] = battery

        if "touch id" in blob.lower():
            structured["touch_id"] = True

        return structured

    def _build_pros_cons(
        self, product_meta: Dict[str, Any], features: List[str]
    ) -> Dict[str, Any]:
        """Create concise fact-based pros/cons JSON object."""
        pros: List[str] = []
        cons: List[str] = []

        rating = product_meta.get("rating")
        if isinstance(rating, (int, float)):
            if rating >= 4.0:
                pros.append(f"High rating ({rating}/5)")
            elif rating < 3.5:
                cons.append(f"Lower rating ({rating}/5)")

        discount = product_meta.get("discount")
        if discount:
            clean_discount = re.sub(
                r"\b(off)(?:\s+off)+\b", r"\1", str(discount), flags=re.IGNORECASE
            )
            pros.append(f"Discount available ({clean_discount})")

        for feat in features[:3]:
            pros.append(feat[:80])

        if not product_meta.get("reviews_count"):
            cons.append("Limited review volume")

        payload = {}
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
        Create a response from the text content.
        Uses extractive summarization (key sentence selection).
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return text[:500]

        # Select key sentences — first, middle, and last
        n = len(sentences)
        key_indices = [0]
        if n > 2:
            key_indices.append(n // 2)
        if n > 1:
            key_indices.append(n - 1)

        # Also include sentences with key indicators
        indicator_words = [
            "important",
            "key",
            "main",
            "significant",
            "essential",
            "critical",
            "primary",
            "fundamental",
            "conclusion",
            "result",
            "finding",
            "demonstrate",
            "show",
            "indicate",
        ]

        for i, sent in enumerate(sentences):
            lower = sent.lower()
            if any(w in lower for w in indicator_words) and i not in key_indices:
                key_indices.append(i)
                if len(key_indices) >= 6:
                    break

        key_indices = sorted(set(key_indices))
        response = " ".join(sentences[i] for i in key_indices if i < n)

        return response

    def _extract_qa_pairs(self, text: str, domain: str) -> List[Dict[str, str]]:
        """Extract Q&A pairs from text using simple heuristics."""
        pairs = []

        # Clean text first (removes e-commerce noise if present)
        clean_text = self._clean_ecommerce_text(text)

        # Extract potential topics from the cleaned text
        topics = self._extract_topics(clean_text)

        if not topics:
            return pairs

        used_instructions = set()
        # Generate Q&A for top topics
        for idx, topic in enumerate(topics[:5]):
            # Find the sentence(s) that discuss this topic
            sentences = re.split(r"(?<=[.!?])\s+", clean_text)
            relevant = [s.strip() for s in sentences
                        if topic.lower() in s.lower() and len(s.strip()) > 20]

            if relevant:
                answer = " ".join(relevant[:2])
                # Skip if the answer is too noisy (mostly numbers/symbols)
                alpha_ratio = sum(c.isalpha() for c in answer) / max(len(answer), 1)
                if alpha_ratio < 0.4:
                    continue
                qa_template = self.QA_TEMPLATES[idx % len(self.QA_TEMPLATES)]
                instruction = qa_template[0].format(topic=topic).strip()
                normalized_inst = re.sub(r"\s+", " ", instruction.lower())
                if normalized_inst in used_instructions:
                    continue
                used_instructions.add(normalized_inst)
                answer = self._limit_output(self._normalize_ecommerce_output(answer), max_chars=320)
                pairs.append(
                    {
                        "instruction": instruction,
                        "input": "",
                        "output": qa_template[1].format(answer=answer).strip(),
                    }
                )

        return pairs

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
