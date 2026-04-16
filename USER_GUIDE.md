# DataClean1 User Guide

This guide covers setup, web usage, API usage, pagination scraping, and troubleshooting.

## 1. Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

Optional for browser-rendered scraping:

```bash
pip install playwright
playwright install chromium
```

### Environment variables

```bash
export SECRET_KEY="change-me"
export DATABASE_URL="sqlite:///pipeline_users.db"
```

## 2. Run the App

```bash
python app.py
```

Open:

- `http://127.0.0.1:8080/login`

## 3. Web Workflow (LLM Pipeline)

1. Sign up or log in.
2. Open `/llm`.
3. Stage 1 Ingest:
- Upload files and/or add URLs.
- For search/listing URLs (Amazon/Flipkart), set `max_pages` > 1 to enable pagination scraping.
4. Stage 2 Process:
- Choose template (`alpaca`, `chatml`, `sharegpt`).
- Choose quality threshold (`Min Quality`).
- Processing uses the internal fixed pipeline:
  `scraper -> raw records -> rule-based extraction -> LLM extraction -> validation -> normalization -> deduplication -> dataset split -> training -> evaluation -> inference API`
- E-commerce samples are automatically deduplicated and normalized.
- E-commerce outputs favor concise structured JSON fields to reduce token cost.
5. Stage 3 Export:
- Provide `version`, `model`, and `method`.
- Download `training_data.jsonl`, `training_config.json`, and training script.

## 4. Python Usage

### A) Product page scraping

```python
from data_pipeline.ecommerce_scraper import EcommerceScraper

scraper = EcommerceScraper(use_playwright=False, max_pages=3)
products = scraper.scrape(["https://www.amazon.in/dp/B0XXXXXXX"])
docs = scraper.to_documents(products)
```

### B) Listing/search pagination scraping

```python
from data_pipeline.ecommerce_scraper import EcommerceScraper

scraper = EcommerceScraper(use_playwright=True, max_pages=10)
products = scraper.scrape_listings(["https://www.amazon.in/s?k=laptop"], max_pages=5)
docs = scraper.to_documents(products)
```

### C) LLM pipeline ingest from listing URLs

```python
from data_pipeline.llm_pipeline import LLMPipeline

llm = LLMPipeline()
docs = llm.ingest_ecommerce_listings(
    ["https://www.flipkart.com/search?q=headphones"],
    max_pages=5,
    use_playwright=False
)
```

## 5. API Quick Reference

### Ingest

`POST /api/llm/ingest` (`multipart/form-data`)

- `files`: uploaded files (optional)
- `urls`: JSON array string (optional), e.g. `["https://www.amazon.in/s?k=laptop"]`
- `max_pages`: integer, e.g. `5`

### Process

`POST /api/llm/process` (`application/json`)

Example:

```json
{
  "session_id": "abcd1234",
  "chunk_method": "sliding_window",
  "chunk_size": 512,
  "template": "alpaca",
  "min_quality": 0.4
}
```

### Architecture

`GET /api/llm/architecture`

### Export

`POST /api/llm/export` (`application/json`)

```json
{
  "session_id": "abcd1234",
  "version": "v1.0.0",
  "model": "meta-llama/Meta-Llama-3-8B",
  "method": "lora"
}
```

## 6. Common Errors and Fixes

### "0 pairs passed the quality filter..."

- Lower `Min Quality` and run Process again.
- Then run Export again.

### Ingest returns 0 docs

- Check URL accessibility.
- For Amazon/Flipkart listing pages, increase `max_pages`.
- If blocked, try `use_playwright=true`.

## 7. Dataset Quality Behavior (E-commerce)

- Duplicate instructions are capped and exact duplicates are removed.
- Repeated noise like `off off`, ad phrases, and delivery promo fragments are cleaned.
- Outputs are length-limited to keep training examples concise.
- Instruction styles are diversified (extraction, summary, pricing, specs, availability, pros/cons).

## 8. Test Commands

```bash
pytest -q test_pagination_scraping.py test_ecommerce_quality.py test_llm_processor.py test_llm_pipeline.py test_llm_export.py
```
