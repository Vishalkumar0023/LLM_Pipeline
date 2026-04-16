"""
Evidence Builder (Layer A)
==========================
Deterministic parser that converts raw product pages/text into canonical
evidence records for downstream dataset synthesis.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests
from bs4 import BeautifulSoup


CANONICAL_FIELDS = [
    "product_name",
    "brand",
    "price",
    "review_count",
    "rating",
    "ram",
    "unified_memory",
    "storage",
    "display",
    "os",
    "camera_rear",
    "camera_front",
    "chip",
    "warranty",
    "battery",
    "touch_id",
    "seller",
    "availability",
    "discount",
    "weight",
    "raw_text",
]


class EvidenceBuilder:
    """
    Build canonical evidence records from URLs, raw HTML, or raw text.

    Contract:
    - Deterministic extraction only (selectors/regex).
    - Missing fields are None.
    - No instruction/output generation in this layer.
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    BRAND_VOCAB = {
        "apple",
        "samsung",
        "xiaomi",
        "oneplus",
        "realme",
        "oppo",
        "vivo",
        "google",
        "motorola",
        "nothing",
        "iqoo",
        "nokia",
        "sony",
        "lenovo",
        "asus",
        "acer",
        "dell",
        "hp",
        "msi",
        "microsoft",
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    # -----------------------------
    # Pipeline Step 1: ingest_urls
    # -----------------------------
    def ingest_urls(self, urls: Sequence[str]) -> List[str]:
        """Normalize and deduplicate URL list."""
        clean = []
        seen = set()
        for url in urls or []:
            value = (url or "").strip()
            if not value:
                continue
            if not value.startswith(("http://", "https://")):
                continue
            if value in seen:
                continue
            seen.add(value)
            clean.append(value)
        return clean

    # -----------------------------
    # Pipeline Step 2: scrape_html
    # -----------------------------
    def scrape_html(self, url: str) -> str:
        """Fetch HTML for a single URL."""
        resp = requests.get(url, headers=self.HEADERS, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    # -----------------------------
    # Pipeline Step 3: build evidence
    # -----------------------------
    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text or ""
        text = re.sub(r"\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def _extract_text_from_html(html: str) -> str:
        """Deterministically strip irrelevant HTML blocks and return visible text."""
        soup = BeautifulSoup(html or "", "html.parser")
        for tag in soup.find_all(
            ["script", "style", "noscript", "svg", "path", "footer", "nav", "form"]
        ):
            tag.decompose()
        body = soup.body or soup
        text = body.get_text(separator="\n", strip=True)
        return EvidenceBuilder._normalize_text(text)

    @staticmethod
    def _parse_brand(product_name: Optional[str], raw_text: str) -> Optional[str]:
        if product_name:
            first = product_name.split()[0].strip(" ,.-").lower()
            if first in EvidenceBuilder.BRAND_VOCAB:
                return first.title()

        low = raw_text.lower()
        for brand in sorted(EvidenceBuilder.BRAND_VOCAB):
            if re.search(rf"\b{re.escape(brand)}\b", low):
                return brand.title()
        return None

    @staticmethod
    def _match_group(pattern: str, text: str, flags: int = re.IGNORECASE) -> Optional[str]:
        m = re.search(pattern, text, flags=flags)
        if not m:
            return None
        return m.group(1).strip()

    @staticmethod
    def _match_product_name(text: str) -> Optional[str]:
        """
        Parse product name safely in both multiline and single-line contexts.
        Stops before common field markers.
        """
        m = re.search(
            r"(?:^|\n)\s*Product\s*:\s*(.+?)(?=\s*(?:\n|Price\s*:|Rating\s*:|Description\s*:|$))",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        name = m.group(1).strip()
        # Listing cards often truncate product names with ellipsis.
        name = re.sub(r"[.]{3,}$", "", name).strip()
        name = re.sub(r"[…]+$", "", name).strip()
        name = name.rstrip("- ").strip()
        return name or None

    @staticmethod
    def parse_evidence_from_text(raw_text: str) -> Dict[str, Any]:
        """Deterministically parse canonical evidence fields from text."""
        text = EvidenceBuilder._normalize_text(raw_text)

        product_name = EvidenceBuilder._match_product_name(text)
        price = EvidenceBuilder._match_group(
            r"(?:^|\n)\s*Price\s*:\s*((?:INR|Rs\.?|₹|\$)\s*[\d,]+(?:\.\d+)?)",
            text,
        )
        rating = EvidenceBuilder._match_group(
            r"(?:^|\n)\s*Rating\s*:\s*([\d.]+/5)", text
        )
        review_count_raw = EvidenceBuilder._match_group(
            r"(?:^|\n)\s*Rating\s*:\s*[\d.]+/5\s*\(([\d,]+)\s*reviews?\)",
            text,
        )
        review_count = None
        if review_count_raw:
            try:
                review_count = int(review_count_raw.replace(",", ""))
            except ValueError:
                review_count = None

        # Memory / RAM
        unified_memory = EvidenceBuilder._match_group(
            r"(\d+\s*GB\s*Unified\s*Memory(?:\s*RAM)?)", text
        )
        ram = (
            unified_memory
            or EvidenceBuilder._match_group(
                r"(\d+\s*(?:GB|TB)\s*(?:LPDDR\d*X?|DDR\d*|SDRAM|RAM))",
                text,
            )
        )

        # Storage
        storage = (
            EvidenceBuilder._match_group(r"(\d+\s*(?:GB|TB)\s*ROM)", text)
            or EvidenceBuilder._match_group(r"(\d+\s*(?:GB|TB)\s*(?:SSD|HDD))", text)
        )

        # Display (supports both cm/inch mentions)
        display = (
            EvidenceBuilder._match_group(
                r"(\d+(?:\.\d+)?\s*cm\s*\([^)]+\)\s*[^.\n]*Display)", text
            )
            or EvidenceBuilder._match_group(
                r"(\d+(?:\.\d+)?\s*(?:inch|inches|\"|in)\s*[^.\n]*Display)", text
            )
        )

        # Operating system
        os_name = (
            EvidenceBuilder._match_group(
                r"((?:64\s*bit\s+)?(?:Windows\s*\d+(?:\s*(?:Home|Pro))?|Mac\s*OS|macOS(?:\s+[A-Za-z0-9]+)?|Chrome\s*OS)[^.\n]*Operating System)",
                text,
            )
            or EvidenceBuilder._match_group(r"(macOS\s+[A-Za-z0-9]+)", text)
            or EvidenceBuilder._match_group(r"(Windows\s*11(?:\s*(?:Home|Pro))?)", text)
            or EvidenceBuilder._match_group(r"(Mac\s*OS)", text)
        )

        # Cameras
        rear_front_combo = re.search(
            r"(\d+\s*MP(?:\s*\+\s*\d+\s*MP)*)\s*(?:Rear\s*Camera)?\s*\|\s*(\d+\s*MP)\s*Front\s*Camera",
            text,
            flags=re.IGNORECASE,
        ) or re.search(
            r"(\d+\s*MP)\s*Rear\s*Camera\s*\|\s*(\d+\s*MP)\s*Front\s*Camera",
            text,
            flags=re.IGNORECASE,
        )
        camera_rear = None
        camera_front = None
        if rear_front_combo:
            camera_rear = rear_front_combo.group(1).strip()
            camera_front = rear_front_combo.group(2).strip()
        else:
            camera_rear = EvidenceBuilder._match_group(
                r"(Rear\s*Camera\s*[:\-]?\s*[^.\n]+)", text
            ) or EvidenceBuilder._match_group(
                r"(\d+\s*MP)\s*Rear\s*Camera", text
            ) or EvidenceBuilder._match_group(
                r"((?:\d+\s*MP\s*\+\s*)+\d+\s*MP)", text
            )
            camera_front = EvidenceBuilder._match_group(
                r"(\d+\s*MP)\s*Front\s*Camera", text
            )

        # Chip / processor
        chip = (
            EvidenceBuilder._match_group(r"((?:A|M)\d+\s*(?:Pro\s*)?(?:Bionic\s*)?Chip)", text)
            or EvidenceBuilder._match_group(
                r"((?:Apple\s+)?(?:A|M)\d+\s*(?:Pro|Max|Ultra|Bionic)?\s*Processor)",
                text,
            )
            or EvidenceBuilder._match_group(
                r"((?:Intel|AMD)\s+[^\n,.;]*?\s*Processor)",
                text,
            )
            or EvidenceBuilder._match_group(
                r"((?:Snapdragon|Dimensity|Exynos)\s*[^\n,.;]*?\s*Processor)",
                text,
            )
            or EvidenceBuilder._match_group(
                r"(Tensor\s*[^\n,.;]*?\s*Processor)",
                text,
            )
            or EvidenceBuilder._match_group(
                r"((?:Snapdragon|Dimensity|Exynos)\s*[^\n,.;]+)", text
            )
            or EvidenceBuilder._match_group(r"(Tensor\s*\w+)", text)
        )
        if chip:
            chip = re.sub(r"\s+", " ", chip).strip(" ,.-")

        # Warranty
        warranty = EvidenceBuilder._match_group(
            r"((?:\(?\d+\)?\s*year|one\s*year)[^.\n]*warranty)", text
        )

        # Optional specs occasionally present in raw context.
        battery = (
            EvidenceBuilder._match_group(r"(?:^|\n)\s*Battery\s*[:\-]?\s*([^\n.]+)", text)
            or EvidenceBuilder._match_group(r"(\d{3,5}\s*mAh[^.\n]*)", text)
        )
        touch_id = (
            EvidenceBuilder._match_group(r"(Touch\s*ID)", text)
            or EvidenceBuilder._match_group(r"(?:^|\n)\s*Touch\s*ID\s*[:\-]?\s*([^\n.]+)", text)
        )
        seller = EvidenceBuilder._match_group(
            r"(?:^|\n)\s*Seller\s*[:\-]?\s*([^\n]+)", text
        )
        availability = EvidenceBuilder._match_group(
            r"(?:^|\n)\s*Availability\s*[:\-]?\s*([^\n]+)", text
        )
        discount = (
            EvidenceBuilder._match_group(r"(\d+\s*%\s*off)", text)
            or EvidenceBuilder._match_group(r"(?:^|\n)\s*Discount\s*[:\-]?\s*([^\n]+)", text)
        )
        weight = EvidenceBuilder._match_group(
            r"(\d+(?:\.\d+)?\s*(?:kg|g)\b[^.\n]*)", text
        )

        brand = EvidenceBuilder._parse_brand(product_name, text)

        record = {
            "product_name": product_name,
            "brand": brand,
            "price": price,
            "review_count": review_count,
            "rating": rating,
            "ram": ram,
            "unified_memory": unified_memory,
            "storage": storage,
            "display": display,
            "os": os_name,
            "camera_rear": camera_rear,
            "camera_front": camera_front,
            "chip": chip,
            "warranty": warranty,
            "battery": battery,
            "touch_id": touch_id,
            "seller": seller,
            "availability": availability,
            "discount": discount,
            "weight": weight,
            "raw_text": text,
        }

        # Ensure strict canonical keys and null missing values.
        normalized = {}
        for field in CANONICAL_FIELDS:
            value = record.get(field)
            normalized[field] = value if value not in ("", [], {}) else None
        if not normalized["raw_text"]:
            normalized["raw_text"] = ""
        return normalized

    def build_evidence_record(self, raw_input: str, is_html: bool = False) -> Dict[str, Any]:
        """Build a canonical evidence record from raw text or HTML."""
        text = self._extract_text_from_html(raw_input) if is_html else raw_input
        return self.parse_evidence_from_text(text)

    def build_evidence_records(
        self,
        urls: Optional[Sequence[str]] = None,
        html_pages: Optional[Sequence[str]] = None,
        raw_texts: Optional[Sequence[str]] = None,
        output_path: str = "evidence_records.jsonl",
    ) -> List[Dict[str, Any]]:
        """
        Build evidence records from URL fetches, raw HTML, and/or raw text.
        Saves to `evidence_records.jsonl` by default.
        """
        records: List[Dict[str, Any]] = []

        for url in self.ingest_urls(urls or []):
            try:
                html = self.scrape_html(url)
                rec = self.build_evidence_record(html, is_html=True)
                records.append(rec)
            except Exception:
                # Keep deterministic layer robust: skip unreadable URLs.
                continue

        for html in html_pages or []:
            rec = self.build_evidence_record(html, is_html=True)
            records.append(rec)

        for text in raw_texts or []:
            rec = self.build_evidence_record(text, is_html=False)
            records.append(rec)

        self.write_jsonl(records, output_path)
        return records

    @staticmethod
    def write_jsonl(records: Sequence[Dict[str, Any]], output_path: str) -> None:
        """Write evidence records to JSONL."""
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                clean = {field: rec.get(field, None) for field in CANONICAL_FIELDS}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    @staticmethod
    def read_jsonl(path: str) -> List[Dict[str, Any]]:
        """Read JSONL records from disk."""
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Enforce canonical contract while reading.
                row = {field: obj.get(field, None) for field in CANONICAL_FIELDS}
                row["raw_text"] = row.get("raw_text") or ""
                rows.append(row)
        return rows
