from data_pipeline.document_ingestor import DocumentIngestor


def test_normalize_sources_splits_concatenated_duplicate_url():
    ingestor = DocumentIngestor()
    raw = (
        "https://www.flipkart.com/search?q=17%20pro%20max&as=off"
        "https://www.flipkart.com/search?q=17%20pro%20max&as=off"
    )
    normalized = ingestor._normalize_sources([raw])
    assert normalized == ["https://www.flipkart.com/search?q=17%20pro%20max&as=off"]


def test_normalize_sources_splits_multiple_urls_and_dedupes():
    ingestor = DocumentIngestor()
    url1 = "https://www.flipkart.com/search?q=17%20pro%20max"
    url2 = "https://www.flipkart.com/search?q=iphone%2016"
    raw = f"{url1}{url2}"
    normalized = ingestor._normalize_sources([raw, url1])
    assert normalized == [url1, url2]
