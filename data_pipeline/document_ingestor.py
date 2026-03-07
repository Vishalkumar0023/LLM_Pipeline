"""
Document Ingestor Module
========================
Multi-source ingestion for LLM fine-tuning pipelines.
Parses PDF, URL/HTML, XML, plain text, and Markdown into a unified format.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Optional imports with graceful fallback
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB = True
except ImportError:
    HAS_WEB = False

try:
    import xml.etree.ElementTree as ET
    HAS_XML = True
except ImportError:
    HAS_XML = False


class DocumentIngestor:
    """
    Multi-source document ingestor that parses diverse file types into
    a unified list of document dictionaries.

    Each document dict has the format:
        {
            "source": str,          # file path or URL
            "source_type": str,     # "pdf", "url", "xml", "text", "markdown"
            "page": int or None,    # page number (PDF only)
            "text": str,            # extracted text content
            "metadata": dict,       # additional metadata
            "doc_id": str,          # unique hash-based document ID
            "ingested_at": str      # ISO timestamp
        }

    Example:
    --------
    >>> ingestor = DocumentIngestor()
    >>> docs = ingestor.ingest(["paper.pdf", "https://example.com", "notes.txt"])
    >>> print(len(docs), "documents ingested")
    """

    SUPPORTED_TEXT_EXTENSIONS = {'.txt', '.text', '.log', '.cfg', '.ini', '.rst'}
    SUPPORTED_MARKDOWN_EXTENSIONS = {'.md', '.markdown', '.mdown', '.mkd'}

    def __init__(self, encoding: str = 'utf-8', timeout: int = 30):
        """
        Initialize the ingestor.

        Parameters
        ----------
        encoding : str
            Default text encoding for file reading.
        timeout : int
            HTTP request timeout in seconds for URL ingestion.
        """
        self.encoding = encoding
        self.timeout = timeout
        self.documents: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, str]] = []
        self._stats = {
            'total_sources': 0,
            'successful': 0,
            'failed': 0,
            'total_documents': 0,
            'by_type': {}
        }

    def ingest(
        self,
        sources: Union[str, List[str]],
        recursive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Ingest one or more sources into unified document format.

        Parameters
        ----------
        sources : str or list of str
            File paths, URLs, or directory paths to ingest.
        recursive : bool
            If True and source is a directory, recursively ingest all
            supported files.

        Returns
        -------
        list of dict
            List of document dictionaries.
        """
        if isinstance(sources, str):
            sources = [sources]

        self.documents = []
        self.errors = []
        self._stats['total_sources'] = len(sources)

        for source in sources:
            try:
                docs = self._ingest_single(source, recursive)
                self.documents.extend(docs)
                self._stats['successful'] += 1
                source_type = docs[0]['source_type'] if docs else 'unknown'
                self._stats['by_type'][source_type] = (
                    self._stats['by_type'].get(source_type, 0) + len(docs)
                )
            except Exception as e:
                self.errors.append({
                    'source': source,
                    'error': str(e)
                })
                self._stats['failed'] += 1

        self._stats['total_documents'] = len(self.documents)
        return self.documents

    def _ingest_single(
        self, source: str, recursive: bool = False
    ) -> List[Dict[str, Any]]:
        """Route a single source to the appropriate parser."""
        # URL detection
        if source.startswith(('http://', 'https://')):
            return self._ingest_url(source)

        path = Path(source)

        # Directory handling
        if path.is_dir():
            return self._ingest_directory(path, recursive)

        if not path.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        ext = path.suffix.lower()

        if ext == '.pdf':
            return self._ingest_pdf(path)
        elif ext == '.xml':
            return self._ingest_xml(path)
        elif ext == '.json':
            return self._ingest_json(path)
        elif ext in self.SUPPORTED_MARKDOWN_EXTENSIONS:
            return self._ingest_text(path, source_type='markdown')
        elif ext in self.SUPPORTED_TEXT_EXTENSIONS or ext == '':
            return self._ingest_text(path, source_type='text')
        else:
            # Try as plain text
            return self._ingest_text(path, source_type='text')

    def _ingest_directory(
        self, dir_path: Path, recursive: bool
    ) -> List[Dict[str, Any]]:
        """Ingest all supported files from a directory."""
        all_docs = []
        pattern = '**/*' if recursive else '*'
        supported = (
            self.SUPPORTED_TEXT_EXTENSIONS |
            self.SUPPORTED_MARKDOWN_EXTENSIONS |
            {'.pdf', '.xml', '.json'}
        )

        for file_path in sorted(dir_path.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in supported:
                try:
                    docs = self._ingest_single(str(file_path), recursive=False)
                    all_docs.extend(docs)
                except Exception as e:
                    self.errors.append({
                        'source': str(file_path),
                        'error': str(e)
                    })

        if not all_docs:
            raise ValueError(f"No supported files found in: {dir_path}")

        return all_docs

    # ─── PDF Ingestion ───────────────────────────────────────────────────

    def _ingest_pdf(self, path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF, one document per page."""
        if not HAS_PDFPLUMBER:
            raise ImportError(
                "pdfplumber is required for PDF ingestion. "
                "Install it with: pip install pdfplumber"
            )

        docs = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ''
                text = self._clean_text(text)

                if text.strip():
                    docs.append(self._make_doc(
                        source=str(path),
                        source_type='pdf',
                        text=text,
                        page=i + 1,
                        metadata={
                            'total_pages': len(pdf.pages),
                            'page_width': page.width,
                            'page_height': page.height
                        }
                    ))

        if not docs:
            raise ValueError(f"No extractable text found in PDF: {path}")

        return docs

    # ─── URL/HTML Ingestion ──────────────────────────────────────────────

    def _ingest_url(self, url: str) -> List[Dict[str, Any]]:
        """Scrape and extract main content from a URL."""
        if not HAS_WEB:
            raise ImportError(
                "requests and beautifulsoup4 are required for URL ingestion. "
                "Install with: pip install requests beautifulsoup4"
            )

        response = requests.get(url, timeout=self.timeout, headers={
            'User-Agent': 'DataPipeline-Ingestor/1.0'
        })
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script, style, nav, footer, header elements
        for tag in soup.find_all([
            'script', 'style', 'nav', 'footer', 'header',
            'aside', 'form', 'iframe', 'noscript'
        ]):
            tag.decompose()

        # Try to find main content area
        main = (
            soup.find('main') or
            soup.find('article') or
            soup.find('div', {'role': 'main'}) or
            soup.find('div', class_=re.compile(r'content|article|post|entry')) or
            soup.body or
            soup
        )

        # Extract title
        title = ''
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)

        text = main.get_text(separator='\n', strip=True)
        text = self._clean_text(text)

        if not text.strip():
            raise ValueError(f"No extractable content from URL: {url}")

        return [self._make_doc(
            source=url,
            source_type='url',
            text=text,
            metadata={
                'title': title,
                'url': url,
                'content_length': len(response.text),
                'status_code': response.status_code
            }
        )]

    # ─── XML Ingestion ───────────────────────────────────────────────────

    def _ingest_xml(self, path: Path) -> List[Dict[str, Any]]:
        """Extract text content from XML documents."""
        tree = ET.parse(str(path))
        root = tree.getroot()

        texts = []
        self._extract_xml_text(root, texts)

        full_text = '\n'.join(texts)
        full_text = self._clean_text(full_text)

        if not full_text.strip():
            raise ValueError(f"No extractable text in XML: {path}")

        return [self._make_doc(
            source=str(path),
            source_type='xml',
            text=full_text,
            metadata={
                'root_tag': root.tag,
                'total_elements': len(list(root.iter()))
            }
        )]

    def _extract_xml_text(
        self, element: ET.Element, texts: List[str]
    ) -> None:
        """Recursively extract text from XML elements."""
        if element.text and element.text.strip():
            texts.append(element.text.strip())
        for child in element:
            self._extract_xml_text(child, texts)
            if child.tail and child.tail.strip():
                texts.append(child.tail.strip())

    # ─── JSON Ingestion ──────────────────────────────────────────────────

    def _ingest_json(self, path: Path) -> List[Dict[str, Any]]:
        """Ingest JSON files — handles both single objects and arrays."""
        with open(path, 'r', encoding=self.encoding) as f:
            data = json.load(f)

        docs = []

        if isinstance(data, list):
            for i, item in enumerate(data):
                text = self._json_to_text(item)
                if text.strip():
                    docs.append(self._make_doc(
                        source=str(path),
                        source_type='json',
                        text=text,
                        metadata={'index': i}
                    ))
        elif isinstance(data, dict):
            text = self._json_to_text(data)
            if text.strip():
                docs.append(self._make_doc(
                    source=str(path),
                    source_type='json',
                    text=text,
                    metadata={}
                ))

        if not docs:
            raise ValueError(f"No extractable text in JSON: {path}")

        return docs

    def _json_to_text(self, obj: Any) -> str:
        """Convert a JSON object to text, extracting string values."""
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            parts = []
            for key, value in obj.items():
                text = self._json_to_text(value)
                if text.strip():
                    parts.append(f"{key}: {text}")
            return '\n'.join(parts)
        elif isinstance(obj, list):
            parts = [self._json_to_text(item) for item in obj]
            return '\n'.join(p for p in parts if p.strip())
        else:
            return str(obj)

    # ─── Text/Markdown Ingestion ─────────────────────────────────────────

    def _ingest_text(
        self, path: Path, source_type: str = 'text'
    ) -> List[Dict[str, Any]]:
        """Read plain text or markdown files."""
        # Try multiple encodings
        text = None
        for enc in [self.encoding, 'utf-8', 'latin-1', 'cp1252']:
            try:
                with open(path, 'r', encoding=enc) as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if text is None:
            raise ValueError(f"Could not decode file with any encoding: {path}")

        text = self._clean_text(text)

        if not text.strip():
            raise ValueError(f"File is empty: {path}")

        return [self._make_doc(
            source=str(path),
            source_type=source_type,
            text=text,
            metadata={
                'file_size_bytes': path.stat().st_size,
                'encoding': self.encoding
            }
        )]

    # ─── Utilities ───────────────────────────────────────────────────────

    def _make_doc(
        self,
        source: str,
        source_type: str,
        text: str,
        page: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a standardized document dictionary."""
        doc_id = hashlib.sha256(
            f"{source}:{page}:{text[:200]}".encode()
        ).hexdigest()[:16]

        return {
            'source': source,
            'source_type': source_type,
            'page': page,
            'text': text,
            'char_count': len(text),
            'word_count': len(text.split()),
            'metadata': metadata or {},
            'doc_id': doc_id,
            'ingested_at': datetime.now().isoformat()
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean raw extracted text."""
        # Normalize whitespace (but preserve paragraph breaks)
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove excessive blank lines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()

    def get_stats(self) -> Dict[str, Any]:
        """Return ingestion statistics."""
        return {
            **self._stats,
            'errors': self.errors
        }

    def print_summary(self) -> None:
        """Print a formatted ingestion summary."""
        stats = self._stats
        print("=" * 60)
        print("DOCUMENT INGESTION SUMMARY")
        print("=" * 60)
        print(f"\n📥 Sources processed: {stats['successful']}/{stats['total_sources']}")
        print(f"📄 Documents extracted: {stats['total_documents']}")

        if stats['by_type']:
            print(f"\n📋 By type:")
            for stype, count in stats['by_type'].items():
                print(f"   • {stype}: {count}")

        if stats['failed'] > 0:
            print(f"\n⚠️  Failed: {stats['failed']}")
            for err in self.errors:
                print(f"   • {err['source']}: {err['error']}")

        total_chars = sum(d['char_count'] for d in self.documents)
        total_words = sum(d['word_count'] for d in self.documents)
        print(f"\n📊 Total: {total_chars:,} chars, {total_words:,} words")
        print("=" * 60)
