"""
E-Commerce Scraper Module
==========================
Site-specific product extractors for e-commerce platforms.
Supports Amazon, Flipkart, and extensible to more sites.
Converts structured product data into LLM pipeline documents.
"""

import re
import hashlib
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from urllib.parse import urlparse

# Core scraping (always available via project deps)
import requests
from bs4 import BeautifulSoup

# Optional: Playwright for JS-rendered pages
try:
    from playwright.sync_api import sync_playwright

    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


# ─── Product Data Model ─────────────────────────────────────────────────


class ProductData:
    """Structured product data container."""

    def __init__(
        self,
        title: str = "",
        price: Optional[float] = None,
        original_price: Optional[float] = None,
        discount: Optional[str] = None,
        currency: str = "INR",
        rating: Optional[float] = None,
        reviews_count: Optional[int] = None,
        description: str = "",
        features: Optional[List[str]] = None,
        image_urls: Optional[List[str]] = None,
        seller: str = "",
        availability: str = "",
        category: str = "",
        brand: str = "",
        url: str = "",
        platform: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.title = title
        self.price = price
        self.original_price = original_price
        self.discount = discount
        self.currency = currency
        self.rating = rating
        self.reviews_count = reviews_count
        self.description = description
        self.features = features or []
        self.image_urls = image_urls or []
        self.seller = seller
        self.availability = availability
        self.category = category
        self.brand = brand
        self.url = url
        self.platform = platform
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "price": self.price,
            "original_price": self.original_price,
            "discount": self.discount,
            "currency": self.currency,
            "rating": self.rating,
            "reviews_count": self.reviews_count,
            "description": self.description,
            "features": self.features,
            "image_urls": self.image_urls,
            "seller": self.seller,
            "availability": self.availability,
            "category": self.category,
            "brand": self.brand,
            "url": self.url,
            "platform": self.platform,
            "metadata": self.metadata,
        }

    def to_text(self) -> str:
        """Convert product data to structured text for LLM ingestion."""
        parts = []

        if self.title:
            parts.append(f"Product: {self.title}")
        if self.brand:
            parts.append(f"Brand: {self.brand}")
        if self.price is not None:
            price_str = f"Price: {self.currency} {self.price:,.2f}"
            if self.original_price and self.original_price > self.price:
                price_str += f" (Original: {self.currency} {self.original_price:,.2f})"
            if self.discount:
                price_str += f" — {self.discount} off"
            parts.append(price_str)
        if self.rating is not None:
            rating_str = f"Rating: {self.rating}/5"
            if self.reviews_count:
                rating_str += f" ({self.reviews_count:,} reviews)"
            parts.append(rating_str)
        if self.availability:
            parts.append(f"Availability: {self.availability}")
        if self.seller:
            parts.append(f"Seller: {self.seller}")
        if self.category:
            parts.append(f"Category: {self.category}")
        if self.description:
            parts.append(f"\nDescription:\n{self.description}")
        if self.features:
            parts.append("\nKey Features:")
            for feat in self.features:
                parts.append(f"  • {feat}")

        return "\n".join(parts)

    def is_valid(self) -> bool:
        """Check if product has minimum required data."""
        return bool(self.title and (self.price is not None or self.description))


# ─── Base Extractor ──────────────────────────────────────────────────────


class BaseExtractor(ABC):
    """Abstract base class for site-specific extractors."""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def __init__(
        self, use_playwright: bool = False, timeout: int = 30, headless: bool = True
    ):
        self.use_playwright = use_playwright and HAS_PLAYWRIGHT
        self.timeout = timeout
        self.headless = headless

    def fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a page, with JS rendering if configured."""
        if self.use_playwright:
            return self._fetch_with_playwright(url)
        return self._fetch_with_requests(url)

    def _fetch_with_requests(self, url: str) -> BeautifulSoup:
        """Fetch page using requests (static HTML only)."""
        response = requests.get(url, headers=self.HEADERS, timeout=self.timeout)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def _fetch_with_playwright(self, url: str) -> BeautifulSoup:
        """Fetch page using Playwright (renders JavaScript)."""
        if not HAS_PLAYWRIGHT:
            raise ImportError(
                "Playwright is required for JS rendering. "
                "Install: pip install playwright && playwright install chromium"
            )
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            page.set_extra_http_headers(self.HEADERS)
            page.goto(url, timeout=self.timeout * 1000)
            page.wait_for_load_state("networkidle")
            content = page.content()
            browser.close()
        return BeautifulSoup(content, "html.parser")

    @abstractmethod
    def extract_product(self, url: str) -> ProductData:
        """Extract product data from a URL."""
        pass

    @abstractmethod
    def is_product_url(self, url: str) -> bool:
        """Check if a URL is a valid product page for this site."""
        pass

    # ─── Selector helpers with fallback ──────────────────────────────

    @staticmethod
    def _select_text(
        soup: BeautifulSoup, selectors: List[str], default: str = ""
    ) -> str:
        """Try multiple CSS selectors, return first match's text."""
        for selector in selectors:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(strip=True)
                if text:
                    return text
        return default

    @staticmethod
    def _select_attr(
        soup: BeautifulSoup, selectors: List[str], attr: str, default: str = ""
    ) -> str:
        """Try multiple CSS selectors, return first match's attribute."""
        for selector in selectors:
            el = soup.select_one(selector)
            if el and el.get(attr):
                return el[attr]
        return default

    @staticmethod
    def _select_all_text(soup: BeautifulSoup, selectors: List[str]) -> List[str]:
        """Try multiple selectors, return all matching texts."""
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                texts = [el.get_text(strip=True) for el in elements]
                return [t for t in texts if t]
        return []

    @staticmethod
    def _parse_price(text: str) -> Optional[float]:
        """Parse price from text like '₹1,299.00' or '$29.99'."""
        if not text:
            return None
        # Remove currency symbols and commas
        cleaned = re.sub(r"[₹$€£,\s]", "", text)
        # Handle shorthand
        match = re.search(r"([\d.]+)", cleaned)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    @staticmethod
    def _parse_rating(text: str) -> Optional[float]:
        """Parse rating from text like '4.3 out of 5'."""
        if not text:
            return None
        match = re.search(r"([\d.]+)", text)
        if match:
            try:
                val = float(match.group(1))
                return val if val <= 5 else None
            except ValueError:
                return None
        return None

    @staticmethod
    def _parse_count(text: str) -> Optional[int]:
        """Parse count from text like '12,345 ratings'."""
        if not text:
            return None
        cleaned = re.sub(r"[,\s]", "", text)
        match = re.search(r"(\d+)", cleaned)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None


# ─── Amazon Extractor ────────────────────────────────────────────────────


class AmazonExtractor(BaseExtractor):
    """Product extractor for Amazon (.in, .com)."""

    DOMAINS = ["amazon.in", "amazon.com", "www.amazon.in", "www.amazon.com"]

    def is_product_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.hostname in self.DOMAINS and (
            "/dp/" in parsed.path or "/gp/product/" in parsed.path
        )

    def extract_product(self, url: str) -> ProductData:
        soup = self.fetch_page(url)

        # Title
        title = self._select_text(
            soup,
            [
                "#productTitle",
                "span#productTitle",
                "h1#title span",
                "h1.product-title-word-break",
            ],
        )

        # Price
        price_text = self._select_text(
            soup,
            [
                "span.a-price-whole",
                ".a-price .a-offscreen",
                "#priceblock_ourprice",
                "#priceblock_dealprice",
                "span.price3P",
                ".a-price-range span.a-offscreen",
            ],
        )
        price = self._parse_price(price_text)

        # Original price (MRP)
        orig_text = self._select_text(
            soup,
            [
                '.a-price[data-a-strike="true"] .a-offscreen',
                ".priceBlockStrikePriceString",
                "span.a-text-strike",
                ".basisPrice .a-offscreen",
            ],
        )
        original_price = self._parse_price(orig_text)

        # Discount
        discount = self._select_text(
            soup,
            [
                ".savingsPercentage",
                "#dealprice_savings .priceBlockSavingsString",
                "span.a-color-price",
            ],
        )

        # Rating
        rating_text = self._select_text(
            soup,
            [
                "span.a-icon-alt",
                "#acrPopover span.a-icon-alt",
                "#averageCustomerReviews span.a-icon-alt",
            ],
        )
        rating = self._parse_rating(rating_text)

        # Review count
        reviews_text = self._select_text(
            soup,
            [
                "#acrCustomerReviewText",
                "span#acrCustomerReviewText",
                "#reviewsMedley .a-size-base",
            ],
        )
        reviews_count = self._parse_count(reviews_text)

        # Description
        desc_parts = []
        # Feature bullets
        bullets = self._select_all_text(
            soup,
            [
                "#feature-bullets ul li span.a-list-item",
                "#feature-bullets li span",
            ],
        )
        if bullets:
            desc_parts.extend(bullets)

        # Product description section
        prod_desc = self._select_text(
            soup,
            [
                "#productDescription p",
                "#productDescription",
                "div.product-description-content",
            ],
        )
        if prod_desc:
            desc_parts.append(prod_desc)

        description = "\n".join(desc_parts)

        # Features / technical details
        features = []
        tech_rows = soup.select("#productDetails_techSpec_section_1 tr")
        if not tech_rows:
            tech_rows = soup.select(".prodDetTable tr")
        if not tech_rows:
            tech_rows = soup.select("#detailBullets_feature_div li")

        for row in tech_rows[:20]:
            cells = row.select("td, th")
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True)
                val = cells[1].get_text(strip=True)
                if key and val:
                    features.append(f"{key}: {val}")
            elif row.get_text(strip=True):
                features.append(row.get_text(strip=True))

        # Images
        image_urls = []
        for img in soup.select("#altImages img, #imgTagWrapperId img, #imageBlock img"):
            src = img.get("src", "") or img.get("data-old-hires", "")
            if src and "sprite" not in src and src.startswith("http"):
                image_urls.append(src)

        # Brand
        brand = self._select_text(
            soup,
            [
                "#bylineInfo",
                "a#bylineInfo",
                ".po-brand .a-span9 .a-size-base",
            ],
        )
        brand = re.sub(r"^(Visit the |Brand:\s*)", "", brand).strip()

        # Availability
        availability = self._select_text(
            soup,
            [
                "#availability span",
                "#availability",
                "div.instock",
            ],
        )

        # Seller
        seller = self._select_text(
            soup,
            [
                "#sellerProfileTriggerId",
                "#merchant-info a",
                "#tabular-buybox .tabular-buybox-text a",
            ],
        )

        # Category
        category_parts = self._select_all_text(
            soup,
            [
                "#wayfinding-breadcrumbs_container a",
                ".a-breadcrumb a",
            ],
        )
        category = " > ".join(category_parts) if category_parts else ""

        return ProductData(
            title=title,
            price=price,
            original_price=original_price,
            discount=discount,
            currency="INR" if "amazon.in" in url else "USD",
            rating=rating,
            reviews_count=reviews_count,
            description=description,
            features=features,
            image_urls=image_urls[:5],
            seller=seller,
            availability=availability,
            category=category,
            brand=brand,
            url=url,
            platform="amazon",
        )


# ─── Flipkart Extractor ─────────────────────────────────────────────────


class FlipkartExtractor(BaseExtractor):
    """Product extractor for Flipkart."""

    DOMAINS = ["flipkart.com", "www.flipkart.com"]

    def is_product_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.hostname in self.DOMAINS and (
            "/p/" in parsed.path or "pid=" in (parsed.query or "")
        )

    def extract_product(self, url: str) -> ProductData:
        soup = self.fetch_page(url)

        # Title
        title = self._select_text(
            soup,
            [
                "span.VU-ZEz",
                "span.B_NuCI",
                "h1._9E25nV",
                "h1.yhB1nd",
            ],
        )

        # Price
        price_text = self._select_text(
            soup,
            [
                "div.Nx9bqj.CxhGGd",
                "div._30jeq3._16Jk6d",
                "div._30jeq3",
                "div.CEmiEU div",
            ],
        )
        price = self._parse_price(price_text)

        # Original price
        orig_text = self._select_text(
            soup,
            [
                "div.yRaY8j.A6\\+E6v",
                "div._3I9_wc._2p6lqe",
                "div._3I9_wc",
            ],
        )
        original_price = self._parse_price(orig_text)

        # Discount
        discount = self._select_text(
            soup,
            [
                "div.UkUFwK span",
                "div._3Ay6Sb._31Dcoz span",
                "span._3Ay6Sb",
            ],
        )

        # Rating
        rating_text = self._select_text(
            soup,
            [
                "div.XQDdHH",
                "div._3LWZlK",
                "span._1lRcqv div.XQDdHH",
            ],
        )
        rating = self._parse_rating(rating_text)

        # Reviews count
        reviews_text = self._select_text(
            soup,
            [
                "span.Wphh3N span span",
                "span._2_R_DZ span",
                "span._13vcmD",
            ],
        )
        reviews_count = self._parse_count(reviews_text)

        # Description
        desc_parts = []
        highlights = self._select_all_text(
            soup,
            [
                "div._2418kt li",
                "div.xFVion li",
                "div._2o3t9m li",
            ],
        )
        if highlights:
            desc_parts.extend(highlights)

        prod_desc = self._select_text(
            soup,
            [
                "div._1mXcCf.RmoJUa",
                "div._1AN87F",
                "div.RmoJUa",
            ],
        )
        if prod_desc:
            desc_parts.append(prod_desc)

        description = "\n".join(desc_parts)

        # Specifications / features
        features = []
        spec_rows = soup.select("div._4BJ2V\\+ tr, table._14cfVK tr")
        for row in spec_rows[:20]:
            cells = row.select("td")
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True)
                val = cells[1].get_text(strip=True)
                if key and val:
                    features.append(f"{key}: {val}")

        # Images
        image_urls = []
        for img in soup.select("img._396cs4, img._2r_T1I, div._3kidJX img"):
            src = img.get("src", "")
            if src and src.startswith("http"):
                # Replace thumbnail with full-size
                src = re.sub(r"/\d+/\d+/", "/800/800/", src)
                image_urls.append(src)

        # Brand
        brand = self._select_text(
            soup,
            [
                "span.mEh187",
                "span._2WkVRV",
            ],
        )

        # Seller
        seller = self._select_text(
            soup,
            [
                "div._1RLviY span span",
                "div._3enH3G span span",
                "#sellerName span",
            ],
        )

        # Availability
        avail_el = soup.select_one("div._16FRp0, div._1dVbu9")
        availability = "Out of Stock" if avail_el else "In Stock"

        # Category from breadcrumbs
        category_parts = self._select_all_text(
            soup,
            [
                "div._3GIHBu a",
                "div._1MR4o5 a",
            ],
        )
        category = " > ".join(category_parts) if category_parts else ""

        return ProductData(
            title=title,
            price=price,
            original_price=original_price,
            discount=discount,
            currency="INR",
            rating=rating,
            reviews_count=reviews_count,
            description=description,
            features=features,
            image_urls=image_urls[:5],
            seller=seller,
            availability=availability,
            category=category,
            brand=brand,
            url=url,
            platform="flipkart",
        )


# ─── Scraper Orchestrator ───────────────────────────────────────────────


class EcommerceScraper:
    """
    Multi-site e-commerce product scraper.

    Automatically detects the platform from the URL and routes
    to the appropriate extractor. Converts product data into
    document dicts compatible with the LLM pipeline.

    Example:
    --------
    >>> scraper = EcommerceScraper()
    >>> products = scraper.scrape(["https://amazon.in/dp/B0EXAMPLE"])
    >>> docs = scraper.to_documents(products)
    >>> # Feed docs into LLMPipeline.ingest() or use directly
    """

    def __init__(
        self,
        use_playwright: bool = False,
        timeout: int = 30,
        delay: float = 1.0,
        max_retries: int = 2,
        headless: bool = True,
    ):
        """
        Initialize the scraper.

        Parameters
        ----------
        use_playwright : bool
            Use Playwright for JS-rendered pages (requires install).
        timeout : int
            HTTP request timeout in seconds.
        delay : float
            Delay between requests (rate limiting).
        max_retries : int
            Number of retries on failure.
        headless: bool
            Whether to run the browser in headless mode.
        """
        self.delay = delay
        self.max_retries = max_retries
        self.headless = headless

        # Initialize extractors
        self._extractors: List[BaseExtractor] = [
            AmazonExtractor(
                use_playwright=use_playwright, timeout=timeout, headless=self.headless
            ),
            FlipkartExtractor(
                use_playwright=use_playwright, timeout=timeout, headless=self.headless
            ),
        ]

        self._stats = {
            "total_urls": 0,
            "successful": 0,
            "failed": 0,
            "products": [],
            "errors": [],
        }

    def register_extractor(self, extractor: BaseExtractor) -> None:
        """Register a custom extractor for additional sites."""
        self._extractors.append(extractor)

    def _find_extractor(self, url: str) -> Optional[BaseExtractor]:
        """Find the appropriate extractor for a URL."""
        for extractor in self._extractors:
            if extractor.is_product_url(url):
                return extractor
        return None

    def scrape(self, urls: Union[str, List[str]]) -> List[ProductData]:
        """
        Scrape product data from one or more URLs.

        Parameters
        ----------
        urls : str or list of str
            Product page URLs.

        Returns
        -------
        list of ProductData
            Extracted product data objects.
        """
        if isinstance(urls, str):
            urls = [urls]

        self._stats["total_urls"] = len(urls)
        products = []

        for i, url in enumerate(urls):
            extractor = self._find_extractor(url)
            if not extractor:
                self._stats["errors"].append(
                    {"url": url, "error": "No extractor found for URL domain"}
                )
                self._stats["failed"] += 1
                continue

            # Retry logic
            for attempt in range(self.max_retries + 1):
                try:
                    product = extractor.extract_product(url)
                    if product.is_valid():
                        products.append(product)
                        self._stats["successful"] += 1
                        self._stats["products"].append(product.title[:80])
                    else:
                        self._stats["errors"].append(
                            {
                                "url": url,
                                "error": "Extracted product data is incomplete",
                            }
                        )
                        self._stats["failed"] += 1
                    break
                except Exception as e:
                    if attempt < self.max_retries:
                        time.sleep(self.delay * (attempt + 1))
                        continue
                    self._stats["errors"].append({"url": url, "error": str(e)})
                    self._stats["failed"] += 1

            # Rate limiting between URLs
            if i < len(urls) - 1:
                time.sleep(self.delay)

        return products

    def to_documents(self, products: List[ProductData]) -> List[Dict[str, Any]]:
        """
        Convert ProductData objects to document dicts compatible
        with the LLM pipeline (same format as DocumentIngestor output).

        Parameters
        ----------
        products : list of ProductData
            Products from scrape().

        Returns
        -------
        list of dict
            Document dicts ready for TextChunker / LLMPipeline.
        """
        documents = []

        for product in products:
            text = product.to_text()
            doc_id = hashlib.sha256(
                f"{product.url}:{product.title[:100]}".encode()
            ).hexdigest()[:16]

            doc = {
                "source": product.url,
                "source_type": "ecommerce",
                "page": None,
                "text": text,
                "char_count": len(text),
                "word_count": len(text.split()),
                "metadata": {
                    "platform": product.platform,
                    "product_data": product.to_dict(),
                    "title": product.title,
                },
                "doc_id": doc_id,
                "ingested_at": datetime.now().isoformat(),
            }
            documents.append(doc)

        return documents

    def scrape_to_documents(self, urls: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Convenience: scrape URLs and convert to documents in one call."""
        products = self.scrape(urls)
        return self.to_documents(products)

    def get_stats(self) -> Dict[str, Any]:
        """Return scraping statistics."""
        return {
            "total_urls": self._stats["total_urls"],
            "successful": self._stats["successful"],
            "failed": self._stats["failed"],
            "products_scraped": len(self._stats["products"]),
            "errors": self._stats["errors"],
        }

    def print_summary(self) -> None:
        """Print a formatted scraping summary."""
        stats = self._stats
        print("=" * 60)
        print("E-COMMERCE SCRAPING SUMMARY")
        print("=" * 60)
        print(f"\n🛒 URLs processed: {stats['total_urls']}")
        print(f"✅ Successful: {stats['successful']}")
        print(f"❌ Failed: {stats['failed']}")

        if stats["products"]:
            print("\n📦 Products scraped:")
            for title in stats["products"][:10]:
                print(f"   • {title}")
            if len(stats["products"]) > 10:
                print(f"   ... and {len(stats['products']) - 10} more")

        if stats["errors"]:
            print("\n⚠️  Errors:")
            for err in stats["errors"][:5]:
                print(f"   • {err['url'][:50]}: {err['error']}")

        print("=" * 60)


# ─── Supported Sites Registry ───────────────────────────────────────────


SUPPORTED_DOMAINS = {
    "amazon.in": "Amazon India",
    "amazon.com": "Amazon US",
    "flipkart.com": "Flipkart",
}


def is_ecommerce_url(url: str) -> bool:
    """Check if a URL belongs to a supported e-commerce platform."""
    try:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        domain = domain.replace("www.", "")
        return domain in SUPPORTED_DOMAINS
    except Exception:
        return False
