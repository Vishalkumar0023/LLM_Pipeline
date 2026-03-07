"""
Text Chunker Module
===================
Splits ingested documents into training-ready text segments
for LLM fine-tuning data preparation.
"""

import re
from typing import List, Dict, Any, Optional


class TextChunker:
    """
    Chunks text documents into smaller segments using various strategies.

    Supports:
    - Sentence-based chunking
    - Paragraph-based chunking
    - Sliding window with overlap
    - Token-aware chunking (estimated token count)

    Example:
    --------
    >>> chunker = TextChunker(method='sliding_window', chunk_size=512, overlap=64)
    >>> chunks = chunker.chunk_documents(documents)
    """

    # Regex for sentence boundary detection
    SENTENCE_BOUNDARY = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n'
    )

    def __init__(
        self,
        method: str = 'paragraph',
        chunk_size: int = 512,
        overlap: int = 64,
        min_chunk_size: int = 50,
        max_chunk_size: int = 2048
    ):
        """
        Initialize the chunker.

        Parameters
        ----------
        method : str
            Chunking method: 'sentence', 'paragraph', 'sliding_window', 'token_aware'
        chunk_size : int
            Target chunk size in characters (or estimated tokens for 'token_aware').
        overlap : int
            Number of characters/tokens to overlap between consecutive chunks.
        min_chunk_size : int
            Minimum chunk size — chunks below this are merged with neighbors.
        max_chunk_size : int
            Maximum chunk size — chunks above this are split further.
        """
        valid_methods = {'sentence', 'paragraph', 'sliding_window', 'token_aware'}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Choose from: {valid_methods}"
            )

        self.method = method
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self._stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'min_chunk_size_actual': 0,
            'max_chunk_size_actual': 0
        }

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk a list of document dicts.

        Parameters
        ----------
        documents : list of dict
            Documents from DocumentIngestor, each with 'text' field.

        Returns
        -------
        list of dict
            List of chunk dicts with original metadata preserved.
        """
        all_chunks = []
        self._stats['total_documents'] = len(documents)

        for doc in documents:
            text = doc.get('text', '')
            if not text.strip():
                continue

            raw_chunks = self._chunk_text(text)

            # Post-process: merge small chunks, split oversized ones
            processed = self._post_process(raw_chunks)

            for i, chunk_text in enumerate(processed):
                chunk = {
                    'text': chunk_text,
                    'chunk_index': i,
                    'total_chunks': len(processed),
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'estimated_tokens': self._estimate_tokens(chunk_text),
                    'source': doc.get('source', ''),
                    'source_type': doc.get('source_type', ''),
                    'doc_id': doc.get('doc_id', ''),
                    'page': doc.get('page'),
                    'method': self.method
                }
                all_chunks.append(chunk)

        # Update stats
        if all_chunks:
            sizes = [c['char_count'] for c in all_chunks]
            self._stats['total_chunks'] = len(all_chunks)
            self._stats['avg_chunk_size'] = sum(sizes) / len(sizes)
            self._stats['min_chunk_size_actual'] = min(sizes)
            self._stats['max_chunk_size_actual'] = max(sizes)

        return all_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """Route to the selected chunking method."""
        if self.method == 'sentence':
            return self._chunk_by_sentence(text)
        elif self.method == 'paragraph':
            return self._chunk_by_paragraph(text)
        elif self.method == 'sliding_window':
            return self._chunk_sliding_window(text)
        elif self.method == 'token_aware':
            return self._chunk_token_aware(text)
        else:
            return [text]

    def _chunk_by_sentence(self, text: str) -> List[str]:
        """Split text into chunks of grouped sentences."""
        sentences = self.SENTENCE_BOUNDARY.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            slen = len(sentence)
            if current_len + slen > self.chunk_size and current:
                chunks.append(' '.join(current))
                # Keep overlap by retaining last sentence(s)
                overlap_chars = 0
                overlap_sents = []
                for s in reversed(current):
                    if overlap_chars + len(s) <= self.overlap:
                        overlap_sents.insert(0, s)
                        overlap_chars += len(s)
                    else:
                        break
                current = overlap_sents
                current_len = overlap_chars

            current.append(sentence)
            current_len += slen

        if current:
            chunks.append(' '.join(current))

        return chunks

    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """Split text by paragraph boundaries (double newlines)."""
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text] if text.strip() else []

        chunks = []
        current = []
        current_len = 0

        for para in paragraphs:
            plen = len(para)
            if current_len + plen > self.chunk_size and current:
                chunks.append('\n\n'.join(current))
                current = []
                current_len = 0

            current.append(para)
            current_len += plen

        if current:
            chunks.append('\n\n'.join(current))

        return chunks

    def _chunk_sliding_window(self, text: str) -> List[str]:
        """Fixed-size sliding window with overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        step = max(1, self.chunk_size - self.overlap)

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at word boundary
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > self.chunk_size * 0.5:
                    chunk = chunk[:last_space]

            if chunk.strip():
                chunks.append(chunk.strip())

            start += step

        return chunks

    def _chunk_token_aware(self, text: str) -> List[str]:
        """
        Chunk by estimated token count.
        Uses whitespace-split ÷ 0.75 as token estimate.
        """
        words = text.split()
        if not words:
            return []

        # Convert target token count to approximate word count
        target_words = int(self.chunk_size * 0.75)
        overlap_words = int(self.overlap * 0.75)

        if len(words) <= target_words:
            return [text]

        chunks = []
        start = 0
        step = max(1, target_words - overlap_words)

        while start < len(words):
            end = min(start + target_words, len(words))
            chunk_words = words[start:end]
            chunk = ' '.join(chunk_words)
            if chunk.strip():
                chunks.append(chunk.strip())
            start += step

        return chunks

    def _post_process(self, chunks: List[str]) -> List[str]:
        """Merge small chunks and split oversized chunks."""
        if not chunks:
            return chunks

        # Merge chunks smaller than min_chunk_size with neighbor
        merged = []
        buffer = ''

        for chunk in chunks:
            if len(chunk) < self.min_chunk_size:
                buffer = (buffer + ' ' + chunk).strip() if buffer else chunk
            else:
                if buffer:
                    # Attach buffer to this chunk
                    chunk = (buffer + ' ' + chunk).strip()
                    buffer = ''
                merged.append(chunk)

        if buffer:
            if merged:
                merged[-1] = (merged[-1] + ' ' + buffer).strip()
            else:
                merged.append(buffer)

        # Split chunks larger than max_chunk_size
        final = []
        for chunk in merged:
            if len(chunk) > self.max_chunk_size:
                sub_chunks = self._chunk_sliding_window(chunk)
                final.extend(sub_chunks)
            else:
                final.append(chunk)

        return final

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: word_count / 0.75."""
        return int(len(text.split()) / 0.75)

    def get_stats(self) -> Dict[str, Any]:
        """Return chunking statistics."""
        return self._stats

    def print_summary(self) -> None:
        """Print a formatted chunking summary."""
        stats = self._stats
        print("=" * 60)
        print("TEXT CHUNKING SUMMARY")
        print("=" * 60)
        print(f"\n📄 Documents processed: {stats['total_documents']}")
        print(f"🧩 Chunks created: {stats['total_chunks']}")
        print(f"📏 Method: {self.method}")
        print(f"   Target size: {self.chunk_size} chars")
        print(f"   Overlap: {self.overlap} chars")

        if stats['total_chunks'] > 0:
            print(f"\n📊 Chunk sizes:")
            print(f"   • Average: {stats['avg_chunk_size']:.0f} chars")
            print(f"   • Min: {stats['min_chunk_size_actual']} chars")
            print(f"   • Max: {stats['max_chunk_size_actual']} chars")

        print("=" * 60)
