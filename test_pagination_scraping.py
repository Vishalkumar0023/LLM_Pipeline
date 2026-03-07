#!/usr/bin/env python3
"""
Test: Pagination Scraping & Listing Extractors
================================================
Validates the new multi-page listing extractors using mock HTML,
without making any real HTTP requests.
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "✅"
FAIL = "❌"
results = []


def report(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed))
    print(f"  {status} {name}")
    if detail:
        print(f"      {detail}")


# ─── Mock HTML Generators ──────────────────────────────────────────────

def make_amazon_listing_html(num_products=5, has_next=True, page=1):
    """Generate mock Amazon search results HTML."""
    cards = ""
    for i in range(num_products):
        cards += f"""
        <div data-component-type="s-search-result" data-asin="B0ASIN{page:02d}{i:02d}">
            <h2><a href="/dp/B0ASIN{page:02d}{i:02d}"><span>Test Product {page}-{i+1} Wireless Speaker 50W</span></a></h2>
            <span class="a-price"><span class="a-offscreen">₹2,{499+i*100}</span></span>
            <span class="a-text-price"><span class="a-offscreen">₹3,999</span></span>
            <span class="a-icon-alt">4.{i} out of 5 stars</span>
            <span class="a-size-base s-underline-text">{1000+i*200}</span>
            <img class="s-image" src="https://images.amazon.in/product{page}{i}.jpg" />
            <span class="a-color-base a-text-bold">In Stock</span>
        </div>
        """

    next_link = ""
    if has_next:
        next_link = f'<a class="s-pagination-next" href="/s?k=speaker&page={page+1}">Next</a>'

    return f"""
    <html><body>
    <div class="s-main-slot">{cards}</div>
    <div class="s-pagination-container">{next_link}</div>
    </body></html>
    """


def make_flipkart_listing_html(num_products=5, has_next=True, page=1):
    """Generate mock Flipkart search results HTML."""
    cards = ""
    for i in range(num_products):
        cards += f"""
        <div class="_1AtVbE">
            <div class="_4rR01T">Flipkart Product {page}-{i+1} Bluetooth Headphones</div>
            <div class="_30jeq3">₹{1299+i*100}</div>
            <div class="_3I9_wc">₹2,999</div>
            <div class="_3Ay6Sb _31Dcoz">{30+i}% off</div>
            <div class="_3LWZlK">4.{i}</div>
            <span class="_2_R_DZ">{500+i*100} Ratings</span>
            <a class="_1fQZEK" href="/product-{page}-{i+1}/p/id{page}{i}"></a>
            <img class="_396cs4" src="https://images.flipkart.com/product{page}{i}.jpg" />
        </div>
        """

    next_link = ""
    if has_next:
        next_link = f'<a class="_1LKTO3" href="/search?q=headphones&page={page+1}">Next</a>'

    return f"""
    <html><body>
    <div class="_1YokD2">{cards}</div>
    <nav>{next_link}</nav>
    </body></html>
    """


# ─── Tests ──────────────────────────────────────────────────────────────

def test_url_detection():
    """Test that listing URLs are correctly identified."""
    print("\n── Test: URL Detection ────────────────────────────────────")
    from data_pipeline.ecommerce_scraper import AmazonListingExtractor, FlipkartListingExtractor

    amazon = AmazonListingExtractor()
    flipkart = FlipkartListingExtractor()

    # Amazon listing URLs
    report("Amazon search URL detected", amazon.is_listing_url("https://www.amazon.in/s?k=laptop"))
    report("Amazon category URL detected", amazon.is_listing_url("https://www.amazon.in/s/ref=nb_sb_noss?k=phone"))
    report("Amazon product URL NOT listing", not amazon.is_listing_url("https://www.amazon.in/dp/B0EXAMPLE"))

    # Flipkart listing URLs
    report("Flipkart search URL detected", flipkart.is_listing_url("https://www.flipkart.com/search?q=laptop"))
    report("Flipkart product URL NOT listing", not flipkart.is_listing_url("https://www.flipkart.com/product/p/id123"))


def test_amazon_card_extraction():
    """Test extracting product cards from mock Amazon HTML."""
    print("\n── Test: Amazon Card Extraction ────────────────────────────")
    from data_pipeline.ecommerce_scraper import AmazonListingExtractor
    from bs4 import BeautifulSoup

    extractor = AmazonListingExtractor()
    html = make_amazon_listing_html(num_products=5, has_next=True, page=1)
    soup = BeautifulSoup(html, "lxml")

    products = extractor._extract_product_cards(soup, "https://www.amazon.in/s?k=speaker")

    report("Extracted 5 products", len(products) == 5, f"Got {len(products)}")

    if products:
        p = products[0]
        report("Has title", bool(p.title), f"'{p.title}'")
        report("Has price", p.price is not None, f"{p.price}")
        report("Has original price", p.original_price is not None, f"{p.original_price}")
        report("Has rating", p.rating is not None, f"{p.rating}")
        report("Has reviews_count", p.reviews_count is not None, f"{p.reviews_count}")
        report("Has image", len(p.image_urls) > 0)
        report("Has product URL", bool(p.url))
        report("Platform is amazon", p.platform == "amazon")
        report("Currency is INR", p.currency == "INR")
        report("Title is clean (no HTML)", "<" not in p.title)


def test_flipkart_card_extraction():
    """Test extracting product cards from mock Flipkart HTML."""
    print("\n── Test: Flipkart Card Extraction ──────────────────────────")
    from data_pipeline.ecommerce_scraper import FlipkartListingExtractor
    from bs4 import BeautifulSoup

    extractor = FlipkartListingExtractor()
    html = make_flipkart_listing_html(num_products=4, has_next=True, page=1)
    soup = BeautifulSoup(html, "lxml")

    products = extractor._extract_product_cards(soup, "https://www.flipkart.com/search?q=headphones")

    report("Extracted 4 products", len(products) == 4, f"Got {len(products)}")

    if products:
        p = products[0]
        report("Has title", bool(p.title), f"'{p.title}'")
        report("Has price", p.price is not None, f"{p.price}")
        report("Has rating", p.rating is not None, f"{p.rating}")
        report("Platform is flipkart", p.platform == "flipkart")


def test_next_page_detection():
    """Test pagination next-page detection for both sites."""
    print("\n── Test: Next Page Detection ───────────────────────────────")
    from data_pipeline.ecommerce_scraper import AmazonListingExtractor, FlipkartListingExtractor
    from bs4 import BeautifulSoup

    # Amazon — has next
    amazon = AmazonListingExtractor()
    html_with_next = make_amazon_listing_html(has_next=True, page=1)
    soup = BeautifulSoup(html_with_next, "lxml")
    next_url = amazon._detect_next_page(soup, "https://www.amazon.in/s?k=speaker")
    report("Amazon next page found", next_url is not None, f"{next_url}")
    if next_url:
        report("Next URL has page=2", "page=2" in next_url)

    # Amazon — no next
    html_no_next = make_amazon_listing_html(has_next=False, page=5)
    soup_no = BeautifulSoup(html_no_next, "lxml")
    no_next = amazon._detect_next_page(soup_no, "https://www.amazon.in/s?k=speaker&page=5")
    report("Amazon last page returns None", no_next is None)

    # Flipkart — has next
    flipkart = FlipkartListingExtractor()
    fk_html = make_flipkart_listing_html(has_next=True, page=1)
    fk_soup = BeautifulSoup(fk_html, "lxml")
    fk_next = flipkart._detect_next_page(fk_soup, "https://www.flipkart.com/search?q=headphones")
    report("Flipkart next page found", fk_next is not None, f"{fk_next}")


def test_price_normalization():
    """Test price normalization for various formats."""
    print("\n── Test: Price Normalization ───────────────────────────────")
    from data_pipeline.ecommerce_scraper import BaseListingExtractor

    # Standard prices
    report("₹1,299 → 1299.0", BaseListingExtractor._normalize_price("₹1,299") == 1299.0)
    report("$29.99 → 29.99", BaseListingExtractor._normalize_price("$29.99") == 29.99)
    report("₹12,000 → 12000.0", BaseListingExtractor._normalize_price("₹12,000") == 12000.0)

    # K/k suffix
    report("₹1.5k → 1500.0", BaseListingExtractor._normalize_price("₹1.5k") == 1500.0)
    report("₹15K → 15000.0", BaseListingExtractor._normalize_price("₹15K") == 15000.0)

    # Lakh suffix
    report("₹1.5L → 150000.0", BaseListingExtractor._normalize_price("₹1.5L") == 150000.0)
    report("₹2lakh → 200000.0", BaseListingExtractor._normalize_price("₹2lakh") == 200000.0)

    # Edge cases
    report("Empty string → None", BaseListingExtractor._normalize_price("") is None)
    report("None → None", BaseListingExtractor._normalize_price(None) is None)


def test_html_text_cleaning():
    """Test HTML tag stripping and whitespace normalization."""
    print("\n── Test: HTML Text Cleaning ────────────────────────────────")
    from data_pipeline.ecommerce_scraper import BaseListingExtractor

    dirty = "<b>Test  Product</b>  <span>with  <i>tags</i></span>  "
    cleaned = BaseListingExtractor._clean_html_text(dirty)
    report("HTML tags removed", "<" not in cleaned, f"'{cleaned}'")
    report("Whitespace normalized", "  " not in cleaned)
    report("Content preserved", "Test Product" in cleaned)

    report("Empty string handled", BaseListingExtractor._clean_html_text("") == "")


def test_page_limit_enforcement():
    """Test that pagination stops at max_pages limit."""
    print("\n── Test: Page Limit Enforcement ────────────────────────────")
    from data_pipeline.ecommerce_scraper import AmazonListingExtractor
    from bs4 import BeautifulSoup

    # Create extractor with max_pages=3
    extractor = AmazonListingExtractor(max_pages=3)

    # Track how many pages were "fetched"
    pages_fetched = []
    original_fetch = extractor.fetch_page

    def mock_fetch(url):
        pages_fetched.append(url)
        page_num = len(pages_fetched)
        # Always has next to test limit enforcement
        html = make_amazon_listing_html(num_products=2, has_next=True, page=page_num)
        return BeautifulSoup(html, "lxml")

    extractor.fetch_page = mock_fetch

    products = extractor.extract_listing("https://www.amazon.in/s?k=test", max_pages=3)

    report("Stopped at 3 pages", len(pages_fetched) == 3, f"Fetched {len(pages_fetched)} pages")
    report("Got products from all pages", len(products) == 6, f"Got {len(products)} products")

    # Check page_number metadata
    if products:
        page_nums = [p.metadata.get("page_number") for p in products if p.metadata]
        report("Page numbers tracked", set(page_nums) == {1, 2, 3}, f"Pages: {set(page_nums)}")


def test_empty_cards_rejected():
    """Test that cards without title or data are filtered out."""
    print("\n── Test: Empty Card Rejection ──────────────────────────────")
    from data_pipeline.ecommerce_scraper import AmazonListingExtractor
    from bs4 import BeautifulSoup

    # HTML with 1 valid card and 1 empty card (no asin)
    html = """
    <html><body>
    <div data-component-type="s-search-result" data-asin="B0VALID">
        <h2><a href="/dp/B0VALID"><span>Valid Product Test Speaker</span></a></h2>
        <span class="a-price"><span class="a-offscreen">₹999</span></span>
    </div>
    <div data-component-type="s-search-result" data-asin="">
        <h2><a href=""><span></span></a></h2>
    </div>
    <div data-component-type="s-search-result">
    </div>
    </body></html>
    """

    extractor = AmazonListingExtractor()
    soup = BeautifulSoup(html, "lxml")
    products = extractor._extract_product_cards(soup, "https://www.amazon.in/s?k=test")

    report("Only valid cards extracted", len(products) == 1, f"Got {len(products)}")
    if products:
        report("Valid title present", "Valid Product" in products[0].title)


def test_scraper_orchestrator_listing():
    """Test that EcommerceScraper routes to listing extractors."""
    print("\n── Test: Orchestrator Listing Routing ──────────────────────")
    from data_pipeline.ecommerce_scraper import EcommerceScraper

    scraper = EcommerceScraper()

    # Test that listing extractors are found
    ext = scraper._find_listing_extractor("https://www.amazon.in/s?k=laptop")
    report("Amazon listing extractor found", ext is not None)

    ext2 = scraper._find_listing_extractor("https://www.flipkart.com/search?q=phone")
    report("Flipkart listing extractor found", ext2 is not None)

    # Product URLs should NOT match listing extractors
    ext3 = scraper._find_listing_extractor("https://www.amazon.in/dp/B0EXAMPLE")
    report("Product URL → no listing extractor", ext3 is None)


def test_discount_calculation():
    """Test discount percentage calculation from prices."""
    print("\n── Test: Discount Calculation ──────────────────────────────")
    from data_pipeline.ecommerce_scraper import AmazonListingExtractor
    from bs4 import BeautifulSoup

    html = """
    <html><body>
    <div data-component-type="s-search-result" data-asin="B0DISC">
        <h2><a href="/dp/B0DISC"><span>Discounted Item Premium Quality</span></a></h2>
        <span class="a-price"><span class="a-offscreen">₹2,000</span></span>
        <span class="a-text-price"><span class="a-offscreen">₹4,000</span></span>
    </div>
    </body></html>
    """

    extractor = AmazonListingExtractor()
    soup = BeautifulSoup(html, "lxml")
    products = extractor._extract_product_cards(soup, "https://www.amazon.in/s?k=test")

    if products:
        p = products[0]
        report("Discount calculated", p.discount == "50% off", f"Got: '{p.discount}'")
        report("Price correct", p.price == 2000.0, f"{p.price}")
        report("Original price correct", p.original_price == 4000.0, f"{p.original_price}")


def main():
    print("\n" + "=" * 60)
    print("  PAGINATION SCRAPING TESTS")
    print("=" * 60)

    test_url_detection()
    test_amazon_card_extraction()
    test_flipkart_card_extraction()
    test_next_page_detection()
    test_price_normalization()
    test_html_text_cleaning()
    test_page_limit_enforcement()
    test_empty_cards_rejected()
    test_scraper_orchestrator_listing()
    test_discount_calculation()

    # Summary
    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
