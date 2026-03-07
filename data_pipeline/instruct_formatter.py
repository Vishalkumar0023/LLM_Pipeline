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
from datetime import datetime


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
        'alpaca': {
            'keys': ['instruction', 'input', 'output'],
            'description': 'Stanford Alpaca format'
        },
        'chatml': {
            'keys': ['messages'],
            'description': 'OpenAI ChatML format'
        },
        'sharegpt': {
            'keys': ['conversations'],
            'description': 'ShareGPT multi-turn format'
        }
    }

    # Instruction generation templates
    INSTRUCTION_TEMPLATES = [
        "Summarize the following {domain} text:",
        "Explain the key concepts in this {domain} content:",
        "What are the main points discussed in this {domain} passage?",
        "Provide a detailed analysis of this {domain} text:",
        "Based on this {domain} content, explain the topic in simple terms:",
        "Extract the most important information from this {domain} text:",
        "What can be learned from this {domain} passage?",
        "Describe the main ideas presented in the following {domain} text:",
    ]

    # Follow-up instruction templates for Q&A extraction
    QA_TEMPLATES = [
        ("What is {topic}?", "Based on the text: {answer}"),
        ("Explain {topic}.", "According to the source: {answer}"),
        ("Describe {topic} as discussed in the text.", "{answer}"),
    ]

    def __init__(
        self,
        template: str = 'alpaca',
        system_prompt: Optional[str] = None
    ):
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
            'total_chunks': 0,
            'total_pairs': 0,
            'template': template,
            'avg_instruction_length': 0,
            'avg_response_length': 0
        }

    def format_chunks(
        self,
        chunks: List[Dict[str, Any]],
        domain: str = 'general',
        generate_qa: bool = True,
        pairs_per_chunk: int = 2
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
        self._stats['total_chunks'] = len(chunks)

        for chunk in chunks:
            text = chunk.get('text', '')
            if not text.strip() or len(text) < 50:
                continue

            # Generate primary instruction-response pairs
            pairs = self._generate_pairs(
                text, domain, pairs_per_chunk
            )

            # Generate Q&A pairs if requested
            if generate_qa:
                qa_pairs = self._extract_qa_pairs(text, domain)
                pairs.extend(qa_pairs)

            # Format to target template
            for pair in pairs:
                formatted = self._apply_template(pair)
                formatted['metadata'] = {
                    'source': chunk.get('source', ''),
                    'doc_id': chunk.get('doc_id', ''),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'domain': domain
                }
                all_pairs.append(formatted)

        # Update stats
        self._stats['total_pairs'] = len(all_pairs)
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
                self._stats['avg_instruction_length'] = sum(inst_lens) / len(inst_lens)
            if resp_lens:
                self._stats['avg_response_length'] = sum(resp_lens) / len(resp_lens)

        return all_pairs

    def _generate_pairs(
        self,
        text: str,
        domain: str,
        n_pairs: int
    ) -> List[Dict[str, str]]:
        """Generate instruction-response pairs from text."""
        pairs = []

        for i in range(min(n_pairs, len(self.INSTRUCTION_TEMPLATES))):
            instruction = self.INSTRUCTION_TEMPLATES[i].format(
                domain=domain
            )

            # Create a summarized response from the text
            response = self._create_response(text, instruction)

            pairs.append({
                'instruction': instruction,
                'input': text,
                'output': response
            })

        return pairs

    def _create_response(self, text: str, instruction: str) -> str:
        """
        Create a response from the text content.
        Uses extractive summarization (key sentence selection).
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
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
            'important', 'key', 'main', 'significant', 'essential',
            'critical', 'primary', 'fundamental', 'conclusion',
            'result', 'finding', 'demonstrate', 'show', 'indicate'
        ]

        for i, sent in enumerate(sentences):
            lower = sent.lower()
            if any(w in lower for w in indicator_words) and i not in key_indices:
                key_indices.append(i)
                if len(key_indices) >= 6:
                    break

        key_indices = sorted(set(key_indices))
        response = ' '.join(sentences[i] for i in key_indices if i < n)

        return response

    def _extract_qa_pairs(
        self,
        text: str,
        domain: str
    ) -> List[Dict[str, str]]:
        """Extract Q&A pairs from text using simple heuristics."""
        pairs = []

        # Extract potential topics from the text
        topics = self._extract_topics(text)

        if not topics:
            return pairs

        # Generate Q&A for top topics
        for topic in topics[:3]:
            # Find the sentence(s) that discuss this topic
            sentences = re.split(r'(?<=[.!?])\s+', text)
            relevant = [s for s in sentences if topic.lower() in s.lower()]

            if relevant:
                answer = ' '.join(relevant[:2])
                qa_template = self.QA_TEMPLATES[
                    len(pairs) % len(self.QA_TEMPLATES)
                ]
                pairs.append({
                    'instruction': qa_template[0].format(topic=topic),
                    'input': '',
                    'output': qa_template[1].format(answer=answer)
                })

        return pairs

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract potential topic phrases from text using simple heuristics.
        Looks for capitalized phrases and noun-like patterns.
        """
        # Find capitalized multi-word phrases
        cap_phrases = re.findall(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text
        )

        # Find phrases after "is", "are", "refers to", "means"
        definition_patterns = re.findall(
            r'(?:^|\.\s+)([A-Z][^.]{10,60}?)(?:\s+is\s+|\s+are\s+|\s+refers?\s+to)',
            text
        )

        topics = list(set(cap_phrases[:5] + definition_patterns[:3]))
        return topics

    def _apply_template(self, pair: Dict[str, str]) -> Dict[str, Any]:
        """Apply the selected template format."""
        if self.template == 'alpaca':
            return self._format_alpaca(pair)
        elif self.template == 'chatml':
            return self._format_chatml(pair)
        elif self.template == 'sharegpt':
            return self._format_sharegpt(pair)
        else:
            return pair

    def _format_alpaca(self, pair: Dict[str, str]) -> Dict[str, Any]:
        """Format as Alpaca-style dict."""
        return {
            'instruction': pair['instruction'],
            'input': pair.get('input', ''),
            'output': pair['output']
        }

    def _format_chatml(self, pair: Dict[str, str]) -> Dict[str, Any]:
        """Format as ChatML-style dict."""
        messages = [
            {'role': 'system', 'content': self.system_prompt}
        ]

        user_content = pair['instruction']
        if pair.get('input'):
            user_content += f"\n\n{pair['input']}"

        messages.append({'role': 'user', 'content': user_content})
        messages.append({'role': 'assistant', 'content': pair['output']})

        return {'messages': messages}

    def _format_sharegpt(self, pair: Dict[str, str]) -> Dict[str, Any]:
        """Format as ShareGPT-style dict."""
        conversations = []

        human_content = pair['instruction']
        if pair.get('input'):
            human_content += f"\n\n{pair['input']}"

        conversations.append({'from': 'human', 'value': human_content})
        conversations.append({'from': 'gpt', 'value': pair['output']})

        return {'conversations': conversations}

    def _extract_instruction_response(
        self, formatted: Dict[str, Any]
    ) -> tuple:
        """Extract instruction and response text from any template format."""
        if self.template == 'alpaca':
            return formatted.get('instruction', ''), formatted.get('output', '')
        elif self.template == 'chatml':
            msgs = formatted.get('messages', [])
            inst = msgs[1]['content'] if len(msgs) > 1 else ''
            resp = msgs[2]['content'] if len(msgs) > 2 else ''
            return inst, resp
        elif self.template == 'sharegpt':
            convs = formatted.get('conversations', [])
            inst = convs[0]['value'] if len(convs) > 0 else ''
            resp = convs[1]['value'] if len(convs) > 1 else ''
            return inst, resp
        return '', ''

    # ─── Export ───────────────────────────────────────────────────────────

    def export_jsonl(
        self,
        pairs: List[Dict[str, Any]],
        output_path: str,
        include_metadata: bool = False
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
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                if not include_metadata and 'metadata' in pair:
                    row = {k: v for k, v in pair.items() if k != 'metadata'}
                else:
                    row = pair
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

        return output_path

    def export_json(
        self,
        pairs: List[Dict[str, Any]],
        output_path: str,
        include_metadata: bool = False
    ) -> str:
        """Export formatted pairs to JSON file (array format)."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        if not include_metadata:
            clean_pairs = [
                {k: v for k, v in p.items() if k != 'metadata'}
                for p in pairs
            ]
        else:
            clean_pairs = pairs

        with open(output_path, 'w', encoding='utf-8') as f:
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

        if stats['total_pairs'] > 0:
            print(f"\n📊 Averages:")
            print(f"   • Instruction length: {stats['avg_instruction_length']:.0f} chars")
            print(f"   • Response length: {stats['avg_response_length']:.0f} chars")

        print("=" * 60)
