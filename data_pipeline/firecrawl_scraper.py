"""
Firecrawl-like Web Scraper
==========================
Lightweight Firecrawl-style scraping/crawling utilities for local deployment.
"""

from __future__ import annotations

import re
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from scrapy import Selector
    from scrapy.http import HtmlResponse
    from scrapy.linkextractors import LinkExtractor

    HAS_SCRAPY = True
except Exception:
    HAS_SCRAPY = False


FormatType = Union[str, Dict[str, Any]]


class FirecrawlLikeScraper:
    """Firecrawl-inspired scraper with scrape/map/crawl primitives."""

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self, timeout: int = 20, engine: str = "auto"):
        self.timeout = int(timeout)
        requested_engine = str(engine or "auto").strip().lower()
        if requested_engine not in {"auto", "requests", "scrapy"}:
            requested_engine = "auto"
        self.requested_engine = requested_engine
        self.engine = self._resolve_engine(requested_engine)

    def scrape(
        self,
        url: str,
        formats: Optional[List[FormatType]] = None,
        only_main_content: bool = True,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Scrape one page and return requested formats."""
        normalized_url = self._normalize_url(url, url)
        html, status_code = self._fetch_html(normalized_url, headers=headers)

        soup = BeautifulSoup(html, "html.parser")
        page_title = self._extract_title(soup)

        working = BeautifulSoup(html, "html.parser")
        self._remove_noise_tags(working, exclude_tags=exclude_tags)

        if include_tags:
            node = self._collect_included_content(working, include_tags)
        elif only_main_content:
            node = self._find_main_node(working)
        else:
            node = working.body or working

        node_html = str(node)
        links = self._extract_links_from_html(node_html, normalized_url)
        if self.engine == "scrapy":
            text = self._node_to_text_scrapy(node_html)
            markdown = self._node_to_markdown_scrapy(node_html)
        else:
            text = self._node_to_text(node)
            markdown = self._node_to_markdown(node)
        clean_html = str(node)

        format_items = self._normalize_formats(formats)
        out: Dict[str, Any] = {}
        for fmt in format_items:
            fmt_type = fmt.get("type")
            if fmt_type == "markdown":
                out["markdown"] = markdown
            elif fmt_type == "text":
                out["text"] = text
            elif fmt_type == "html":
                out["html"] = clean_html
            elif fmt_type == "rawHtml":
                out["rawHtml"] = html
            elif fmt_type == "links":
                out["links"] = links
            elif fmt_type == "metadata":
                out["metadata"] = self._metadata(
                    normalized_url, page_title, status_code, text
                )
            elif fmt_type == "json":
                out["json"] = self._json_extract_stub(text, fmt)

        # Keep metadata available for downstream indexing.
        if "metadata" not in out:
            out["metadata"] = self._metadata(
                normalized_url, page_title, status_code, text
            )
        if "links" not in out:
            out["links"] = links

        out["engine"] = self.engine
        return out

    def map(
        self,
        url: str,
        limit: int = 100,
        max_depth: int = 2,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        allow_external_links: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Discover links from a website using BFS traversal."""
        start = self._normalize_url(url, url)
        parsed_start = urlparse(start)
        base_domain = parsed_start.netloc

        discovered: List[str] = []
        visited = set()
        queue = deque([(start, 0)])

        while queue and len(discovered) < int(limit):
            current, depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if self._path_allowed(current, include_paths, exclude_paths):
                discovered.append(current)

            if depth >= int(max_depth):
                continue

            try:
                html, _ = self._fetch_html(current, headers=headers)
            except Exception:
                continue

            links = self._extract_links_from_html(html, current)

            for link in links:
                if not allow_external_links and urlparse(link).netloc != base_domain:
                    continue
                if link in visited:
                    continue
                if not self._path_allowed(link, include_paths, exclude_paths):
                    continue
                queue.append((link, depth + 1))

        return discovered[: int(limit)]

    def crawl(
        self,
        url: str,
        limit: int = 25,
        max_depth: int = 2,
        scrape_options: Optional[Dict[str, Any]] = None,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        allow_external_links: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Crawl a site and scrape each discovered page."""
        scrape_options = scrape_options or {}
        formats = scrape_options.get("formats", ["markdown", "metadata"])
        only_main_content = bool(scrape_options.get("onlyMainContent", True))
        include_tags = scrape_options.get("includeTags")
        exclude_tags = scrape_options.get("excludeTags")

        urls = self.map(
            url=url,
            limit=limit,
            max_depth=max_depth,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            allow_external_links=allow_external_links,
            headers=headers,
        )

        data = []
        errors = []
        for item_url in urls:
            try:
                page = self.scrape(
                    url=item_url,
                    formats=formats,
                    only_main_content=only_main_content,
                    include_tags=include_tags,
                    exclude_tags=exclude_tags,
                    headers=headers,
                )
                page["metadata"]["sourceURL"] = item_url
                data.append(page)
            except Exception as exc:
                errors.append({"url": item_url, "error": str(exc)})

        return {
            "status": "completed",
            "total": len(data),
            "engine": self.engine,
            "data": data,
            "errors": errors,
        }

    def _resolve_engine(self, requested_engine: str) -> str:
        if requested_engine == "requests":
            return "requests"
        if requested_engine == "scrapy":
            return "scrapy" if HAS_SCRAPY else "requests"
        # auto
        return "scrapy" if HAS_SCRAPY else "requests"

    def _fetch_html(
        self, url: str, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[str, int]:
        request_headers = dict(self.DEFAULT_HEADERS)
        if headers:
            request_headers.update(headers)
        response = requests.get(url, timeout=self.timeout, headers=request_headers)
        response.raise_for_status()
        return response.text, int(response.status_code)

    def _normalize_formats(self, formats: Optional[List[FormatType]]) -> List[Dict[str, Any]]:
        if not formats:
            return [{"type": "markdown"}]
        out: List[Dict[str, Any]] = []
        for item in formats:
            if isinstance(item, str):
                out.append({"type": item})
            elif isinstance(item, dict) and item.get("type"):
                out.append(item)
        return out or [{"type": "markdown"}]

    def _extract_title(self, soup: BeautifulSoup) -> str:
        if soup.title and soup.title.get_text(strip=True):
            return soup.title.get_text(strip=True)
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
        return ""

    def _remove_noise_tags(
        self, soup: BeautifulSoup, exclude_tags: Optional[Iterable[str]] = None
    ) -> None:
        defaults = [
            "script",
            "style",
            "noscript",
            "svg",
            "iframe",
            "form",
            "nav",
            "footer",
            "header",
            "aside",
        ]
        tags = set(defaults)
        for tag in exclude_tags or []:
            if isinstance(tag, str) and tag.strip():
                tags.add(tag.strip())
        for tag_name in tags:
            for node in soup.find_all(tag_name):
                node.decompose()

    def _find_main_node(self, soup: BeautifulSoup):
        return (
            soup.find("main")
            or soup.find("article")
            or soup.find("section")
            or soup.find("div", {"role": "main"})
            or soup.body
            or soup
        )

    def _collect_included_content(self, soup: BeautifulSoup, include_tags: List[str]):
        container = BeautifulSoup("<div></div>", "html.parser")
        root = container.div
        for tag in include_tags:
            if not isinstance(tag, str) or not tag.strip():
                continue
            for node in soup.find_all(tag.strip()):
                root.append(node)
        return root if root and root.contents else self._find_main_node(soup)

    def _node_to_text(self, node) -> str:
        text = node.get_text(separator="\n", strip=True)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _node_to_markdown(self, node) -> str:
        lines: List[str] = []

        def add_line(value: str = ""):
            value = re.sub(r"[ \t]+", " ", (value or "")).strip()
            lines.append(value)

        for element in node.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "blockquote"]
        ):
            txt = element.get_text(" ", strip=True)
            if not txt:
                continue
            name = element.name
            if name and name.startswith("h") and len(name) == 2 and name[1].isdigit():
                level = int(name[1])
                add_line("#" * max(1, min(6, level)) + " " + txt)
            elif name == "li":
                add_line("- " + txt)
            elif name == "blockquote":
                add_line("> " + txt)
            elif name in ("pre", "code"):
                add_line("```")
                add_line(txt)
                add_line("```")
            else:
                add_line(txt)

        markdown = "\n".join(line for line in lines if line is not None).strip()
        if not markdown:
            markdown = self._node_to_text(node)
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        return markdown.strip()

    def _extract_links(self, node, base_url: str) -> List[str]:
        links = []
        seen = set()
        for anchor in node.find_all("a", href=True):
            raw = (anchor.get("href") or "").strip()
            if not raw:
                continue
            joined = self._normalize_url(raw, base_url)
            if not joined:
                continue
            if joined in seen:
                continue
            seen.add(joined)
            links.append(joined)
        return links

    def _extract_links_scrapy(self, html: str, base_url: str) -> List[str]:
        if not HAS_SCRAPY:
            return []
        response = HtmlResponse(
            url=base_url,
            body=(html or "").encode("utf-8"),
            encoding="utf-8",
        )
        extractor = LinkExtractor()
        out = []
        seen = set()
        for link in extractor.extract_links(response):
            normalized = self._normalize_url(link.url, base_url)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
        return out

    def _extract_links_from_html(self, html: str, base_url: str) -> List[str]:
        if self.engine == "scrapy":
            links = self._extract_links_scrapy(html, base_url)
            if links:
                return links
        soup = BeautifulSoup(html, "html.parser")
        return self._extract_links(soup, base_url)

    def _normalize_url(self, value: str, base_url: str) -> str:
        if not value:
            return ""
        if value.startswith(("mailto:", "tel:", "javascript:")):
            return ""
        absolute = urljoin(base_url, value)
        absolute = urldefrag(absolute)[0].strip()
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return ""
        return absolute

    def _path_allowed(
        self,
        url: str,
        include_paths: Optional[List[str]],
        exclude_paths: Optional[List[str]],
    ) -> bool:
        path = urlparse(url).path or "/"

        if include_paths:
            include_ok = any(path.startswith(p) for p in include_paths if p)
            if not include_ok:
                return False

        if exclude_paths:
            exclude_hit = any(path.startswith(p) for p in exclude_paths if p)
            if exclude_hit:
                return False

        return True

    def _metadata(
        self, source_url: str, title: str, status_code: int, text: str
    ) -> Dict[str, Any]:
        return {
            "title": title,
            "sourceURL": source_url,
            "statusCode": status_code,
            "contentLength": len(text or ""),
            "engine": self.engine,
            "scrapedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    def _json_extract_stub(self, text: str, fmt: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(fmt.get("prompt", "") or "").strip()
        schema = fmt.get("schema")
        preview = re.sub(r"\s+", " ", (text or "")).strip()[:500]
        return {
            "prompt": prompt,
            "schemaProvided": bool(schema),
            "preview": preview,
        }

    def _node_to_text_scrapy(self, html: str) -> str:
        if not HAS_SCRAPY:
            return self._node_to_text(BeautifulSoup(html, "html.parser"))
        selector = Selector(text=html or "")
        lines = [
            re.sub(r"[ \t]+", " ", (t or "")).strip()
            for t in selector.css("::text").getall()
        ]
        lines = [line for line in lines if line]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _node_to_markdown_scrapy(self, html: str) -> str:
        if not HAS_SCRAPY:
            return self._node_to_markdown(BeautifulSoup(html, "html.parser"))

        selector = Selector(text=html or "")
        nodes = selector.xpath(
            "//h1|//h2|//h3|//h4|//h5|//h6|//p|//li|//pre|//code|//blockquote"
        )

        lines: List[str] = []
        for node in nodes:
            name = (node.root.tag or "").lower()
            txt = re.sub(r"\s+", " ", (node.xpath("string(.)").get() or "")).strip()
            if not txt:
                continue
            if name.startswith("h") and len(name) == 2 and name[1].isdigit():
                level = int(name[1])
                lines.append("#" * max(1, min(6, level)) + " " + txt)
            elif name == "li":
                lines.append("- " + txt)
            elif name == "blockquote":
                lines.append("> " + txt)
            elif name in ("pre", "code"):
                lines.extend(["```", txt, "```"])
            else:
                lines.append(txt)

        markdown = "\n".join(lines).strip()
        if not markdown:
            markdown = self._node_to_text_scrapy(html)
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        return markdown.strip()
