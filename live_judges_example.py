#!/usr/bin/env python3
"""
Live Execution Example: Watch the scraper and the judges in action!
This script:
1. Opens a visible browser (headless=False) to scrape an e-commerce URL.
2. Processes the text into chunks and instruction formatting.
3. Runs the Quality Scorer ("the live judges") and prints the verdict for each chunk.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import (
    DocumentIngestor,
    TextChunker,
    InstructFormatter,
    QualityScorer,
)
from data_pipeline.ecommerce_scraper import EcommerceScraper


def main():
    if len(sys.argv) < 2:
        url = "https://www.amazon.in/dp/B0CHX1W1ZT"  # Default test URL (iPhone 15)
        print(f"No URL provided. Using default: {url}")
    else:
        url = sys.argv[1]

    print("\n" + "=" * 60)
    print(f"🚀 STARTING LIVE SCRAPE: {url}")
    print("=" * 60)
    print("👀 Opening visible browser (headless=False)... Watch the screen!")

    # ─── 1. Live Scraping ──────────────────────────────────────
    scraper = EcommerceScraper(use_playwright=True, headless=False)
    products = scraper.scrape([url])

    if not products:
        print(
            "❌ E-commerce extraction failed or unavailable. Falling back to generic text ingestion..."
        )
        ingestor = DocumentIngestor()
        try:
            docs = ingestor.ingest([url])
            print(
                f"\n✅ Generic Ingestion Successful: {docs[0].get('char_count')} characters."
            )
        except Exception as e:
            print(f"❌ Generic ingestion also failed: {e}")
            return
    else:
        product = products[0]
        print(f"\n✅ Successfully Scraped Product: {product.title[:60]}...")
        if product.price:
            print(f"💰 Price: {product.price} {product.currency}")
        docs = scraper.to_documents(products)

    # ─── 2. Chunking ──────────────────────────────────────────
    print("\n🧩 Chunking Document (Semantic Method)...")
    chunker = TextChunker(method="semantic")
    chunks = chunker.chunk_documents(docs)
    print(f"   -> Created {len(chunks)} chunks.")

    # ─── 3. Formatting ────────────────────────────────────────
    print("\n📝 Formatting into Instruction Pairs (E-Commerce Templates)...")
    formatter = InstructFormatter(template="alpaca")
    pairs = formatter.format_chunks(chunks, domain="e-commerce product")
    print(f"   -> Generated {len(pairs)} candidate instruction pairs.")

    # ─── 4. Live Judging (Quality Scoring) ────────────────────
    print("\n" + "=" * 60)
    print("⚖️  LIVE JUDGING (QUALITY SCORING)")
    print("=" * 60)

    scorer = QualityScorer(min_quality_score=0.4)
    time.sleep(1)  # Dramatic pause

    scored_pairs = scorer.score(pairs)

    passed_count = 0
    for i, pair in enumerate(scored_pairs):
        quality = pair["quality"]
        score = quality["overall_score"]

        print(f"\n🔹 Pair {i + 1}/{len(pairs)}")
        print(f"Q: {pair['instruction'][:80]}...")
        print(f"A: {pair['output'][:80]}...")

        # The Verdict
        print("   --- JUDGE'S VERDICT ---")
        print(f"   Overall Score: {score:.3f}")
        print(
            f"   Length Score: {quality['length_score']:.2f} | Diversity: {quality['diversity_score']:.2f}"
        )

        flags = []
        if quality["is_toxic"]:
            flags.append("TOXIC")
        if quality["is_duplicate"]:
            flags.append("DUPLICATE")
        if quality["low_quality_flags"]:
            flags.extend(quality["low_quality_flags"])

        if score >= 0.4 and not flags:
            print("   ✅ APPROVED")
            passed_count += 1
        else:
            reasons = ", ".join(flags) if flags else "Score too low"
            print(f"   ❌ REJECTED (Reason: {reasons})")

        time.sleep(0.5)  # Pause to simulate 'live' read-out

    print("\n" + "=" * 60)
    print(f"🏁 JUDGING COMPLETE: {passed_count}/{len(pairs)} Pairs Approved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
