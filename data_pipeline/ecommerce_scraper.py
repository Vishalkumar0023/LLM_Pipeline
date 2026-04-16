"""
E-Commerce Scraper Module
==========================
Site-specific product extractors for e-commerce platforms.
Supports Amazon, Flipkart, and extensible to more sites.
Converts structured product data into LLM pipeline documents.
"""

import re
import random
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
        reviews: Optional[List[str]] = None,
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
        self.reviews = reviews or []

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
        if self.reviews:
            parts.append("\nCustomer Reviews:")
            for i, rev in enumerate(self.reviews):
                # Clean up multiple spaces and newlines
                clean_rev = re.sub(r'\s+', ' ', rev).strip()
                if clean_rev:
                    parts.append(f"  [{i+1}] {clean_rev}")

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
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "Connection": "keep-alive",
    }

    def __init__(
        self, use_playwright: bool = False, timeout: int = 30, headless: bool = True, max_pages: int = 1
    ):
        self.use_playwright = use_playwright and HAS_PLAYWRIGHT
        self.timeout = timeout
        self.headless = headless
        self.max_pages = max_pages

    def fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a page, with JS rendering if configured."""
        if self.use_playwright:
            return self._fetch_with_playwright(url)
        return self._fetch_with_requests(url)

    def _fetch_with_requests(self, url: str) -> BeautifulSoup:
        """Fetch page using requests (static HTML only)."""
        response = requests.get(url, headers=self.HEADERS, timeout=self.timeout)
        # Some e-commerce sites block first-hit requests; try one warmup retry.
        if response.status_code == 403:
            parsed = urlparse(url)
            home = f"{parsed.scheme}://{parsed.netloc}/"
            retry_headers = dict(self.HEADERS)
            retry_headers["Referer"] = home
            retry_headers["sec-fetch-site"] = "same-origin"
            try:
                requests.get(home, headers=retry_headers, timeout=self.timeout)
            except Exception:
                pass
            response = requests.get(url, headers=retry_headers, timeout=self.timeout)
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")

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
        return BeautifulSoup(content, "lxml")

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

        # Collect reviews with pagination
        reviews = self._select_all_text(soup, [".review-text-content span", "div[data-hook='review-collapsed'] span"])
        all_reviews_link = soup.select_one("a[data-hook='see-all-reviews-link-foot']")
        
        if all_reviews_link and self.max_pages > 1:
            next_url = all_reviews_link.get("href")
            if next_url:
                parsed = urlparse(url)
                if next_url.startswith("/"):
                    next_url = f"{parsed.scheme}://{parsed.netloc}{next_url}"
                pages_fetched = 1
                while next_url and pages_fetched < self.max_pages:
                    try:
                        time.sleep(1.0)
                        rev_soup = self.fetch_page(next_url)
                        page_reviews = self._select_all_text(rev_soup, [".review-text-content span", "div[data-hook='review'] span[data-hook='review-body']"])
                        if page_reviews:
                            reviews.extend(page_reviews)
                        
                        next_page_tag = rev_soup.select_one("li.a-last a")
                        if next_page_tag and next_page_tag.get("href"):
                            next_route = next_page_tag.get("href")
                            next_url = f"{parsed.scheme}://{parsed.netloc}{next_route}" if next_route.startswith("/") else next_route
                        else:
                            next_url = None
                        pages_fetched += 1
                    except Exception:
                        break

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
            reviews=reviews,
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

        # Collect reviews with pagination
        reviews = self._select_all_text(soup, ["div.ZmyDvd", "div.t-ZTKy div div"])
        all_reviews_parent = soup.select_one("div._3UAT2v")
        
        if all_reviews_parent and all_reviews_parent.parent and all_reviews_parent.parent.name == "a" and self.max_pages > 1:
            next_url = all_reviews_parent.parent.get("href")
            if next_url:
                parsed = urlparse(url)
                if next_url.startswith("/"):
                    next_url = f"{parsed.scheme}://{parsed.netloc}{next_url}"
                pages_fetched = 1
                while next_url and pages_fetched < self.max_pages:
                    try:
                        time.sleep(1.0)
                        rev_soup = self.fetch_page(next_url)
                        page_reviews = self._select_all_text(rev_soup, ["div.ZmyDvd", "div.t-ZTKy div div"])
                        if page_reviews:
                            reviews.extend(page_reviews)
                        
                        next_page_tag = None
                        for a in rev_soup.select("nav a"):
                            if "NEXT" in a.get_text(strip=True).upper():
                                next_page_tag = a
                                break
                        
                        if next_page_tag and next_page_tag.get("href"):
                            next_route = next_page_tag.get("href")
                            next_url = f"{parsed.scheme}://{parsed.netloc}{next_route}" if next_route.startswith("/") else next_route
                        else:
                            next_url = None
                        pages_fetched += 1
                    except Exception:
                        break

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
            reviews=reviews,
        )



# ─── Meesho Extractor ───────────────────────────────────────────────────

class MeeshoExtractor(BaseExtractor):
    """Product extractor for Meesho."""

    DOMAINS = ["meesho.com", "www.meesho.com"]

    def is_product_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.hostname in self.DOMAINS and "/p/" in parsed.path

    def extract_product(self, url: str) -> ProductData:
        soup = self.fetch_page(url)

        title = self._select_text(soup, ["span.Text__StyledText-sc-oo0kvp-0.fqolwF", "h1", ".ProductDescription__Brand-sc-"])
        price_text = self._select_text(soup, ["h4.Text__StyledText-sc-oo0kvp-0", ".ProductPrice__Price-sc-"])
        price = self._parse_price(price_text)
        
        rating_text = self._select_text(soup, [".Rating__StyledRating-sc-", "span.Rating__StyledRating-sc-"])
        rating = self._parse_rating(rating_text)
        
        description = self._select_text(soup, [".ProductDescription__Description-sc-", "div.FreeShippingAndReturn__Container-sc-"])
        
        image_urls = []
        for img in soup.select("img"):
            src = img.get("src", "")
            if src and src.startswith("http") and ("images" in src or "product" in src):
                image_urls.append(src)

        return ProductData(
            title=title, price=price, currency="INR", rating=rating,
            description=description, image_urls=image_urls[:5], url=url, platform="meesho"
        )

# ─── Myntra Extractor ───────────────────────────────────────────────────

class MyntraExtractor(BaseExtractor):
    """Product extractor for Myntra."""

    DOMAINS = ["myntra.com", "www.myntra.com"]

    def is_product_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.hostname in self.DOMAINS

    def extract_product(self, url: str) -> ProductData:
        soup = self.fetch_page(url)

        title = self._select_text(soup, ["h1.pdp-name", "h1.pdp-title"])
        brand = self._select_text(soup, ["h1.pdp-title"])  # Myntra puts brand in title
        price_text = self._select_text(soup, ["span.pdp-price", ".pdp-price strong"])
        price = self._parse_price(price_text)
        
        original_price_text = self._select_text(soup, ["span.pdp-mrp s"])
        original_price = self._parse_price(original_price_text)

        rating_text = self._select_text(soup, ["div.index-overallRating div"])
        rating = self._parse_rating(rating_text)
        
        description = self._select_text(soup, [".pdp-productDescriptorsContainer", ".pdp-product-description-content"])
        
        image_urls = []
        for div in soup.select(".image-grid-image"):
            style = div.get("style", "")
            match = re.search(r'url\("([^"]+)"\)', style)
            if match:
                image_urls.append(match.group(1))

        return ProductData(
            title=title, brand=brand, price=price, original_price=original_price, 
            currency="INR", rating=rating, description=description, 
            image_urls=image_urls[:5], url=url, platform="myntra"
        )

# ─── Ajio Extractor ─────────────────────────────────────────────────────

class AjioExtractor(BaseExtractor):
    """Product extractor for Ajio."""

    DOMAINS = ["ajio.com", "www.ajio.com"]

    def is_product_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.hostname in self.DOMAINS and "/p/" in parsed.path

    def extract_product(self, url: str) -> ProductData:
        soup = self.fetch_page(url)

        title = self._select_text(soup, ["h1.prod-name"])
        brand = self._select_text(soup, ["h2.brand-name"])
        price_text = self._select_text(soup, ["div.prod-price"])
        price = self._parse_price(price_text)
        
        discount = self._select_text(soup, [".discount-perc"])
        description = self._select_text(soup, [".prod-desc"])

        image_urls = []
        for img in soup.select(".prod-image img"):
            src = img.get("src", "")
            if src and src.startswith("http"):
                image_urls.append(src)

        return ProductData(
            title=title, brand=brand, price=price, discount=discount,
            currency="INR", description=description, 
            image_urls=image_urls[:5], url=url, platform="ajio"
        )

# ─── eBay Extractor ─────────────────────────────────────────────────────

class EbayExtractor(BaseExtractor):
    """Product extractor for eBay."""

    DOMAINS = ["ebay.com", "www.ebay.com", "ebay.co.uk", "ebay.in"]

    def is_product_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return any(d in parsed.hostname for d in self.DOMAINS) and "/itm/" in parsed.path

    def extract_product(self, url: str) -> ProductData:
        soup = self.fetch_page(url)

        title = self._select_text(soup, ["h1.x-item-title__mainTitle span", "h1.x-item-title__mainTitle"])
        price_text = self._select_text(soup, ["div.x-price-primary span", "div.x-price-primary"])
        price = self._parse_price(price_text)
        
        seller = self._select_text(soup, [".x-sellercard-atf__info__about-seller a span"])
        condition = self._select_text(soup, [".x-item-condition-text .ux-textspans"])

        features = self._select_all_text(soup, [".ux-labels-values__labels", ".ux-labels-values__values"])

        image_urls = []
        for img in soup.select(".ux-image-carousel-item img"):
            src = img.get("src", "") or img.get("data-src", "")
            if src and src.startswith("http"):
                src = re.sub(r's-l\d+\.', 's-l1600.', src)
                image_urls.append(src)

        # Build feature list nicely
        cleaned_features = []
        for i in range(0, len(features)-1, 2):
             cleaned_features.append(f"{features[i]}: {features[i+1]}")

        return ProductData(
            title=title, price=price, currency="USD", seller=seller,
            description=f"Condition: {condition}", features=cleaned_features,
            image_urls=image_urls[:5], url=url, platform="ebay"
        )


# ─── Listing Page Extractors (Pagination) ────────────────────────────────


class BaseListingExtractor(ABC):
    """
    Abstract base class for paginated search/listing page extractors.
    Scrapes product cards from search results across multiple pages.
    """

    # Rotate User-Agents to reduce blocking risk
    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    ]

    def __init__(
        self,
        use_playwright: bool = False,
        timeout: int = 30,
        headless: bool = True,
        max_pages: int = 10,
        min_delay: float = 1.0,
        max_delay: float = 3.0,
    ):
        self.use_playwright = use_playwright and HAS_PLAYWRIGHT
        self.timeout = timeout
        self.headless = headless
        self.max_pages = max_pages
        self.min_delay = min_delay
        self.max_delay = max_delay
        # Updated on each extract_listing() call.
        self.last_pages_scraped = 0

    def _get_headers(self) -> Dict[str, str]:
        """Return headers with a randomly selected User-Agent."""
        return {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.google.com/",
            "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "cross-site",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "Connection": "keep-alive",
        }

    def _human_delay(self):
        """Sleep a random duration to mimic human browsing."""
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)

    def fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a page with anti-blocking measures."""
        if self.use_playwright:
            return self._fetch_with_playwright(url)
        return self._fetch_with_requests(url)

    def _fetch_with_requests(self, url: str) -> BeautifulSoup:
        """Fetch page using requests with rotated headers."""
        headers = self._get_headers()
        response = requests.get(url, headers=headers, timeout=self.timeout)
        # Try a warmed retry once for 403 responses (common on listing pages).
        if response.status_code == 403:
            parsed = urlparse(url)
            home = f"{parsed.scheme}://{parsed.netloc}/"
            retry_headers = self._get_headers()
            retry_headers["Referer"] = home
            retry_headers["sec-fetch-site"] = "same-origin"
            try:
                requests.get(home, headers=retry_headers, timeout=self.timeout)
            except Exception:
                pass
            response = requests.get(url, headers=retry_headers, timeout=self.timeout)
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")

    def _fetch_with_playwright(self, url: str) -> BeautifulSoup:
        """Fetch page using Playwright (JS rendering)."""
        if not HAS_PLAYWRIGHT:
            raise ImportError(
                "Playwright is required for JS rendering. "
                "Install: pip install playwright && playwright install chromium"
            )
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent=random.choice(self.USER_AGENTS),
                viewport={"width": 1920, "height": 1080},
            )
            page = context.new_page()
            page.goto(url, timeout=self.timeout * 1000)
            page.wait_for_load_state("networkidle")
            # Scroll down to trigger lazy loading
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(1)
            content = page.content()
            browser.close()
        return BeautifulSoup(content, "lxml")

    @abstractmethod
    def is_listing_url(self, url: str) -> bool:
        """Check if a URL is a search/listing page for this site."""
        pass

    @abstractmethod
    def _extract_product_cards(self, soup: BeautifulSoup, page_url: str) -> List[ProductData]:
        """Extract product cards from a single listing page."""
        pass

    @abstractmethod
    def _detect_next_page(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        """Detect the URL of the next page, or None if no more pages."""
        pass

    def extract_listing(
        self, url: str, max_pages: Optional[int] = None
    ) -> List[ProductData]:
        """
        Paginate through a search/listing page and extract all product cards.

        Parameters
        ----------
        url : str
            Starting search/listing URL.
        max_pages : int, optional
            Override for max pages to scrape.

        Returns
        -------
        list of ProductData
            All products found across pages.
        """
        limit = max_pages or self.max_pages
        all_products: List[ProductData] = []
        current_url = url
        pages_scraped = 0
        visited_urls = set()

        print(f"  🔍 Starting listing scrape: {url[:80]}...")

        while current_url and pages_scraped < limit:
            if current_url in visited_urls:
                print("     ⏹ Pagination loop detected; stopping.")
                break
            visited_urls.add(current_url)
            pages_scraped += 1
            print(f"  📄 Page {pages_scraped}/{limit}: {current_url[:80]}...")

            try:
                soup = self.fetch_page(current_url)
                products = self._extract_product_cards(soup, current_url)

                # Tag each product with page number
                for product in products:
                    product.metadata = product.metadata or {}
                    product.metadata["page_number"] = pages_scraped
                    product.metadata["listing_url"] = url

                all_products.extend(products)
                print(f"     ✅ Extracted {len(products)} products (total: {len(all_products)})")

                # Check for next page
                next_url = self._detect_next_page(soup, current_url)
                if not next_url or next_url == current_url:
                    print(f"     ⏹ No more pages found.")
                    break
                if next_url in visited_urls:
                    print("     ⏹ Next page already visited; stopping.")
                    break

                current_url = next_url

                # Anti-blocking delay between pages
                if pages_scraped < limit:
                    self._human_delay()

            except Exception as e:
                print(f"     ❌ Error on page {pages_scraped}: {e}")
                break

        self.last_pages_scraped = pages_scraped
        print(f"  🏁 Listing scrape complete: {len(all_products)} products from {pages_scraped} pages")
        return all_products

    # ─── Shared Helpers ──────────────────────────────────────────────

    @staticmethod
    def _clean_html_text(text: str) -> str:
        """Strip HTML tags, extra spaces, and newlines from text."""
        if not text:
            return ""
        # Remove HTML tags
        cleaned = re.sub(r"<[^>]+>", "", text)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _normalize_price(text: str) -> Optional[float]:
        """
        Convert price text to float. Handles:
        - '₹1,299.00' → 1299.0
        - '$29.99' → 29.99
        - '₹1.5k' → 1500.0
        - '₹12,000' → 12000.0
        """
        if not text:
            return None
        text = text.strip()

        # Handle k/K suffix (e.g., ₹1.5k → 1500)
        k_match = re.search(r"([\d,]+\.?\d*)\s*[kK]", text)
        if k_match:
            try:
                return float(k_match.group(1).replace(",", "")) * 1000
            except ValueError:
                pass

        # Handle L/lakh suffix
        l_match = re.search(r"([\d,]+\.?\d*)\s*(?:L|lakh)", text, re.IGNORECASE)
        if l_match:
            try:
                return float(l_match.group(1).replace(",", "")) * 100000
            except ValueError:
                pass

        # Standard price parsing
        price_match = re.search(r"[\d,]+\.?\d*", text.replace(",", ""))
        if price_match:
            try:
                # Re-parse from original to handle commas properly
                nums = re.findall(r"[\d,]+\.?\d*", text)
                if nums:
                    return float(nums[0].replace(",", ""))
            except ValueError:
                pass
        return None

    @staticmethod
    def _select_text(
        soup: BeautifulSoup, selectors: List[str], default: str = ""
    ) -> str:
        """Try multiple CSS selectors, return first match's text."""
        for selector in selectors:
            try:
                el = soup.select_one(selector)
                if el:
                    text = el.get_text(strip=True)
                    if text:
                        return text
            except Exception:
                continue
        return default

    @staticmethod
    def _select_attr(
        soup: BeautifulSoup, selectors: List[str], attr: str, default: str = ""
    ) -> str:
        """Try multiple CSS selectors, return first match's attribute."""
        for selector in selectors:
            try:
                el = soup.select_one(selector)
                if el and el.get(attr):
                    return el.get(attr, default)
            except Exception:
                continue
        return default

    @staticmethod
    def _parse_rating(text: str) -> Optional[float]:
        """Parse rating from text like '4.3 out of 5' or '4.3'."""
        if not text:
            return None
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            val = float(match.group(1))
            if 0 <= val <= 5:
                return val
        return None

    @staticmethod
    def _parse_count(text: str) -> Optional[int]:
        """Parse count from text like '12,345 ratings'."""
        if not text:
            return None
        nums = re.findall(r"[\d,]+", text)
        if nums:
            try:
                return int(nums[0].replace(",", ""))
            except ValueError:
                return None
        return None


class AmazonListingExtractor(BaseListingExtractor):
    """
    Extract product cards from Amazon search result pages.
    Handles pagination via 'Next' button.
    """

    DOMAINS = ["amazon.in", "amazon.com", "www.amazon.in", "www.amazon.com"]

    def is_listing_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return (
            parsed.hostname in self.DOMAINS
            and ("/s?" in url or "/s/" in parsed.path or "/s?" in url)
        )

    def _extract_product_cards(self, soup: BeautifulSoup, page_url: str) -> List[ProductData]:
        """Extract product cards from Amazon search results."""
        products = []
        parsed = urlparse(page_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        currency = "INR" if "amazon.in" in page_url else "USD"

        # Main product card selectors
        cards = soup.select('div[data-component-type="s-search-result"]')
        if not cards:
            cards = soup.select("div.s-result-item[data-asin]")
        if not cards:
            cards = soup.select("div.sg-col-inner .s-result-item")

        for card in cards:
            try:
                asin = card.get("data-asin", "")
                if not asin:
                    continue

                # Title
                title = self._select_text(
                    card,
                    [
                        "h2 a span",
                        "h2 span.a-text-normal",
                        ".a-size-medium.a-text-normal",
                        ".a-size-base-plus.a-text-normal",
                    ],
                )
                if not title:
                    continue  # Skip ads / empty cards

                # Price
                price_text = self._select_text(
                    card,
                    [
                        "span.a-price span.a-offscreen",
                        "span.a-price-whole",
                        ".a-price .a-offscreen",
                    ],
                )
                price = self._normalize_price(price_text)

                # Original price (strikethrough)
                orig_text = self._select_text(
                    card,
                    [
                        'span.a-price[data-a-strike="true"] span.a-offscreen',
                        "span.a-text-price span.a-offscreen",
                    ],
                )
                original_price = self._normalize_price(orig_text)

                # Rating
                rating_text = self._select_text(
                    card,
                    [
                        "span.a-icon-alt",
                        "i.a-icon-star-small span.a-icon-alt",
                    ],
                )
                rating = self._parse_rating(rating_text)

                # Reviews count
                reviews_text = self._select_text(
                    card,
                    [
                        "span.a-size-base.s-underline-text",
                        "a.s-underline-text span",
                    ],
                )
                reviews_count = self._parse_count(reviews_text)

                # Product URL
                product_link = self._select_attr(
                    card, ["h2 a", "a.a-link-normal.s-no-outline"], "href"
                )
                product_url = ""
                if product_link:
                    if product_link.startswith("/"):
                        product_url = f"{domain}{product_link}"
                    elif product_link.startswith("http"):
                        product_url = product_link

                # Image
                image_url = self._select_attr(
                    card, ["img.s-image", ".s-image"], "src"
                )

                # Availability / delivery
                availability = self._select_text(
                    card,
                    [
                        "span.a-color-base.a-text-bold",
                        ".a-row.a-size-base .a-text-bold",
                    ],
                )
                if not availability:
                    availability = "Available"

                # Discount
                discount = ""
                if price and original_price and original_price > price:
                    pct = round((1 - price / original_price) * 100)
                    discount = f"{pct}% off"

                product = ProductData(
                    title=self._clean_html_text(title),
                    price=price,
                    original_price=original_price,
                    discount=discount,
                    currency=currency,
                    rating=rating,
                    reviews_count=reviews_count,
                    availability=self._clean_html_text(availability),
                    url=product_url,
                    platform="amazon",
                    image_urls=[image_url] if image_url else [],
                    metadata={"asin": asin, "source_page": page_url},
                )
                if product.is_valid():
                    products.append(product)

            except Exception:
                continue

        return products

    def _detect_next_page(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        """Find the Next button URL on Amazon search results."""
        parsed = urlparse(current_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        # Try multiple selectors for the Next button
        for selector in [
            "a.s-pagination-next",
            "li.a-last a",
            'ul.a-pagination li.a-last a',
        ]:
            el = soup.select_one(selector)
            if el and el.get("href"):
                href = el["href"]
                if href.startswith("/"):
                    return f"{domain}{href}"
                elif href.startswith("http"):
                    return href
        return None


class FlipkartListingExtractor(BaseListingExtractor):
    """
    Extract product cards from Flipkart search result pages.
    Handles pagination via 'Next' button.
    """

    DOMAINS = ["flipkart.com", "www.flipkart.com"]

    def is_listing_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.hostname in self.DOMAINS and (
            "/search?q=" in url or "/q/" in parsed.path
        )

    def _extract_product_cards(self, soup: BeautifulSoup, page_url: str) -> List[ProductData]:
        """Extract product cards from Flipkart search results."""
        products = []
        parsed = urlparse(page_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        # Flipkart product card selectors (they change class names frequently)
        cards = soup.select("div._1AtVbE")
        if not cards:
            cards = soup.select("div._2kHMtA")
        if not cards:
            cards = soup.select("div._4ddWXP")
        if not cards:
            # Fallback: look for product links with images
            cards = soup.select("div[data-id]")

        for card in cards:
            try:
                # Title
                title = self._select_text(
                    card,
                    [
                        "div._4rR01T",
                        "a.s1Q9rs",
                        "a.IRpwTa",
                        "div.KzDlHZ",
                        "a.wjcEIp",
                        "div.RG5Slk",
                    ],
                )
                if not title or len(title) < 5:
                    continue

                # Price
                price_text = self._select_text(
                    card,
                    [
                        "div._30jeq3._1_WHN1",
                        "div._30jeq3",
                        "div.Nx9bqj",
                        "div.hZ3P6w",
                    ],
                )
                price = self._normalize_price(price_text)

                # Original price
                orig_text = self._select_text(
                    card,
                    [
                        "div._3I9_wc._27UcVY",
                        "div._3I9_wc",
                        "div.yRaY8j",
                    ],
                )
                original_price = self._normalize_price(orig_text)

                # Discount
                discount = self._select_text(
                    card,
                    [
                        "div._3Ay6Sb._31Dcoz",
                        "div.UkUFwK",
                    ],
                )

                # Rating
                rating_text = self._select_text(
                    card,
                    [
                        "div._3LWZlK",
                        "div.XQDdHH",
                        "div.MKiFS6",
                    ],
                )
                rating = self._parse_rating(rating_text)

                # Reviews / ratings count
                reviews_text = self._select_text(
                    card,
                    [
                        "span._2_R_DZ",
                        "span.Wphh3N",
                        "span.PvbNMB",
                    ],
                )
                reviews_count = self._parse_count(reviews_text)

                # Product URL
                product_link = self._select_attr(
                    card, ["a._1fQZEK", "a.s1Q9rs", "a.IRpwTa", "a._2rpwqI", "a.CGtC98", "a.k7wcnx"], "href"
                )
                product_url = ""
                if product_link:
                    if product_link.startswith("/"):
                        product_url = f"{domain}{product_link}"
                    elif product_link.startswith("http"):
                        product_url = product_link

                # Image URL
                image_url = self._select_attr(
                    card, ["img._396cs4", "img._2r_T1I", "img"], "src"
                )
                # Convert thumbnail to full size
                if image_url:
                    image_url = re.sub(r"/\d+/\d+/", "/800/800/", image_url)

                # Description snippet
                desc_parts = []
                for sel in ["li.rgWa7D", "ul.G4BRas li", "div._1xgFaf", "li.DTBslk"]:
                    items = card.select(sel)
                    for item in items[:5]:
                        t = item.get_text(strip=True)
                        if t:
                            desc_parts.append(t)

                product = ProductData(
                    title=self._clean_html_text(title),
                    price=price,
                    original_price=original_price,
                    discount=self._clean_html_text(discount),
                    currency="INR",
                    rating=rating,
                    reviews_count=reviews_count,
                    description="\n".join(desc_parts),
                    url=product_url,
                    platform="flipkart",
                    image_urls=[image_url] if image_url else [],
                    metadata={"source_page": page_url},
                )
                if product.is_valid():
                    products.append(product)

            except Exception:
                continue

        return products

    def _detect_next_page(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        """Find the Next button URL on Flipkart search results."""
        parsed = urlparse(current_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        for selector in ["a._1LKTO3", "nav a[href]:last-child", "a._9QVEpD"]:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(strip=True).upper()
                href = el.get("href", "")
                if "NEXT" in text and href:
                    if href.startswith("/"):
                        return f"{domain}{href}"
                    elif href.startswith("http"):
                        return href

        # Fallback: look for page number links and find current + 1
        page_links = soup.select("nav a[href]")
        for link in page_links:
            text = link.get_text(strip=True)
            if text.isdigit():
                # Check if this is the next sequential page
                href = link.get("href", "")
                if href and "page=" in href:
                    if href.startswith("/"):
                        return f"{domain}{href}"
                    elif href.startswith("http"):
                        return href
        return None


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
        max_pages: int = 10,
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
        max_pages: int
            Maximum number of pages to scrape (e.g., for reviews).
        """
        self.delay = delay
        self.max_retries = max_retries
        self.headless = headless
        self.max_pages = max_pages

        self.use_playwright = use_playwright
        self.timeout = timeout

        # Initialize product page extractors
        self._extractors: List[BaseExtractor] = [
            AmazonExtractor(use_playwright=use_playwright, timeout=timeout, headless=self.headless, max_pages=self.max_pages),
            FlipkartExtractor(use_playwright=True, timeout=timeout, headless=self.headless, max_pages=self.max_pages),
            MeeshoExtractor(use_playwright=use_playwright, timeout=timeout, headless=self.headless, max_pages=self.max_pages),
            MyntraExtractor(use_playwright=use_playwright, timeout=timeout, headless=self.headless, max_pages=self.max_pages),
            AjioExtractor(use_playwright=use_playwright, timeout=timeout, headless=self.headless, max_pages=self.max_pages),
            EbayExtractor(use_playwright=use_playwright, timeout=timeout, headless=self.headless, max_pages=self.max_pages),
        ]

        # Initialize listing/search page extractors
        self._listing_extractors: List[BaseListingExtractor] = [
            AmazonListingExtractor(
                use_playwright=use_playwright, timeout=timeout,
                headless=self.headless, max_pages=self.max_pages,
            ),
            FlipkartListingExtractor(
                use_playwright=True, timeout=timeout,
                headless=self.headless, max_pages=self.max_pages,
            ),
        ]

        self._stats = {
            "total_urls": 0,
            "successful": 0,
            "failed": 0,
            "products": [],
            "errors": [],
            "pages_scraped": 0,
        }

    def register_extractor(self, extractor: BaseExtractor) -> None:
        """Register a custom product page extractor for additional sites."""
        self._extractors.append(extractor)

    def register_listing_extractor(self, extractor: BaseListingExtractor) -> None:
        """Register a custom listing page extractor for additional sites."""
        self._listing_extractors.append(extractor)

    def _find_extractor(self, url: str) -> Optional[BaseExtractor]:
        """Find the appropriate product page extractor for a URL."""
        for extractor in self._extractors:
            if extractor.is_product_url(url):
                return extractor
        return None

    def _find_listing_extractor(self, url: str) -> Optional[BaseListingExtractor]:
        """Find the appropriate listing page extractor for a URL."""
        for extractor in self._listing_extractors:
            if extractor.is_listing_url(url):
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

    def scrape_listings(
        self, urls: Union[str, List[str]], max_pages: int = 10
    ) -> List[ProductData]:
        """
        Scrape product data from search/listing pages with pagination.

        Parameters
        ----------
        urls : str or list of str
            Search or category listing URLs.
        max_pages : int
            Maximum pages to scrape per URL.

        Returns
        -------
        list of ProductData
            All products found across all listing pages.
        """
        if isinstance(urls, str):
            urls = [urls]

        self._stats["total_urls"] = self._stats.get("total_urls", 0) + len(urls)
        all_products: List[ProductData] = []

        for i, url in enumerate(urls):
            extractor = self._find_listing_extractor(url)
            if not extractor:
                self._stats["errors"].append(
                    {"url": url, "error": "No listing extractor found for URL"}
                )
                self._stats["failed"] = self._stats.get("failed", 0) + 1
                continue

            try:
                products = extractor.extract_listing(url, max_pages=max_pages)
                all_products.extend(products)
                self._stats["successful"] = self._stats.get("successful", 0) + 1
                self._stats["pages_scraped"] = (
                    self._stats.get("pages_scraped", 0)
                    + getattr(extractor, "last_pages_scraped", 0)
                )
                for p in products:
                    self._stats["products"].append(p.title[:80])
            except Exception as e:
                self._stats["errors"].append({"url": url, "error": str(e)})
                self._stats["failed"] = self._stats.get("failed", 0) + 1

            # Rate limiting between different listing URLs
            if i < len(urls) - 1:
                time.sleep(self.delay)

        return all_products

    def scrape_listings_to_documents(
        self, urls: Union[str, List[str]], max_pages: int = 10
    ) -> List[Dict[str, Any]]:
        """Convenience: scrape listing URLs and convert to documents."""
        products = self.scrape_listings(urls, max_pages=max_pages)
        return self.to_documents(products)

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
    "meesho.com": "Meesho",
    "myntra.com": "Myntra",
    "ajio.com": "Ajio",
    "ebay.com": "eBay",
    "ebay.co.uk": "eBay UK",
    "ebay.in": "eBay India",
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
