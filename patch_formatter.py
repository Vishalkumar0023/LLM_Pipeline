import re
import random

with open("data_pipeline/instruct_formatter.py", "r") as f:
    content = f.read()

# 1. Update Templates
template_replacement = """
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
"""

content = re.sub(
    r"    # Instruction generation templates.*?ECOMMERCE_NOISE_PATTERNS = \[",
    template_replacement,
    content,
    flags=re.DOTALL
)

# 2. Remove Metadata appending in format_chunks
metadata_target = """                formatted["metadata"] = {
                    "source": chunk.get("source", ""),
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "domain": domain,
                    "is_ecommerce": is_ecommerce,
                }
                all_pairs.append(formatted)"""

metadata_replacement = """                # Ensure no forbidden metadata fields are in the final JSON
                if "metadata" in formatted:
                    del formatted["metadata"]
                all_pairs.append(formatted)"""

content = content.replace(metadata_target, metadata_replacement)

# 3. Modify `_clean_ecommerce_text` to accept `keep_noise` parameter
clean_ecommerce_target = """    def _clean_ecommerce_text(self, text: str) -> str:
        \"\"\"
        Remove promotional noise, duplicate prices, delivery info, and ad copy
        from e-commerce text before using it for instruction generation.
        \"\"\"
        cleaned = text"""

clean_ecommerce_replacement = """    def _clean_ecommerce_text(self, text: str, keep_noise: bool = False) -> str:
        \"\"\"
        Remove promotional noise, duplicate prices, delivery info, and ad copy
        from e-commerce text before using it for instruction generation.
        \"\"\"
        if keep_noise:
            # Output messy formatting and raw elements intentionally
            return text
            
        cleaned = text"""

content = content.replace(clean_ecommerce_target, clean_ecommerce_replacement)

# 4. Rewrite `_generate_ecommerce_pairs` to support task distribution, reasoning, missing data, and input mutation
new_generate_ecommerce = """
    def _generate_ecommerce_pairs(
        self, text: str, product_meta: dict, n_pairs: int
    ) -> List[Dict[str, str]]:
        import random
        import json
        pairs = []
        title = product_meta.get("title", "this product")
        short_title = title[:80] if len(title) > 80 else title
        structured = self._build_structured_product_output(product_meta, text)
        compact_structured = json.dumps(structured, ensure_ascii=False)
        features = structured.get("features", [])
        reviews = structured.get("review_snippets", [])

        # Input mutation: Randomly shuffle attributes in the input text block or inject HTML noise
        mutated_text = text
        if random.random() < 0.3:
            # Keep noisy HTML
            mutated_text = f"<div><html>{text}</html></div>"
        
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
                
                # Missing data handling constraint
                if random.random() < 0.2:
                    instruction += " Extract the warranty and weight."
                    output = "The context does not provide information about the warranty and weight."
                else:
                    output = compact_structured
                    
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
                elif "missing" in lower_tmpl:
                    output = "The context does not provide information about several specifications."
                else:
                    payload = self._pick_fields(structured, ["product_name", "brand", "category"])
                    output = json.dumps(payload, ensure_ascii=False) if payload else "Brand and category not found."

            elif task_type == "SUMMARIZATION":
                tmpl = random.choice(self.ECOMMERCE_TEMPLATES_SUMMARIZATION)
                instruction = tmpl.format(product=short_title)
                
                line1_parts = [
                    structured.get("product_name", ""),
                    f"by {structured.get('brand')}" if structured.get("brand") else "",
                ]
                line2_parts = []
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
                
                # Reasoning / CoT Trace
                thought = "Let's think step-by-step. "
                if rating := structured.get("rating"):
                    thought += f"The rating is {rating}, meaning it's generally well-reviewed. "
                if price := structured.get("price"):
                    thought += f"The price is {price}. "
                    
                pros_cons = self._build_pros_cons(product_meta, features)
                thought += f"Based on the pros and cons, the recommendation is clear. "
                
                output = f"<thought>{thought}</thought>\\nRecommendation: " + (json.dumps(pros_cons, ensure_ascii=False) if pros_cons else "Insufficient data to recommend.")

            output = self._normalize_ecommerce_output(output)
            output = self._limit_output(output, max_chars=1000)

            if output and len(output.strip()) > 10:
                pairs.append({
                    "instruction": instruction,
                    "input": mutated_text,
                    "output": output
                })

        return pairs
"""

content = re.sub(
    r"    def _generate_ecommerce_pairs\(.*?return pairs\n",
    new_generate_ecommerce,
    content,
    flags=re.DOTALL
)

# 5. Connect keep_noise usage in format_chunks
update_call = """            if is_ecommerce:
                # Use e-commerce-specific generation for product data
                import random
                keep_noise = random.random() < 0.3
                clean_text = self._clean_ecommerce_text(text, keep_noise=keep_noise)"""
                
content = re.sub(
    r"            if is_ecommerce:\n\s+# Use e-commerce-specific generation for product data\n\s+clean_text = self\._clean_ecommerce_text\(text\)",
    update_call,
    content
)


with open("data_pipeline/instruct_formatter.py", "w") as f:
    f.write(content)
