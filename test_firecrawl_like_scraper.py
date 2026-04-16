import os
import uuid

import pytest

from app import create_app
from extensions import db


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "test_firecrawl.db"
    old_db_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    app = create_app()
    app.config.update(
        TESTING=True,
        SECRET_KEY="test-secret-key-at-least-32-bytes-long",
    )

    with app.app_context():
        db.drop_all()
        db.create_all()

    with app.test_client() as test_client:
        yield test_client

    if old_db_url is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = old_db_url


def _signup(client):
    suffix = uuid.uuid4().hex[:8]
    resp = client.post(
        "/signup",
        json={
            "username": f"firecrawl_{suffix}",
            "email": f"firecrawl_{suffix}@example.com",
            "password": "password123",
        },
    )
    assert resp.status_code == 200


def test_firecrawl_scrape_endpoint(monkeypatch, client):
    _signup(client)

    pages = {
        "https://example.com": """
        <html><head><title>Home</title></head>
        <body>
          <main>
            <h1>Example Home</h1>
            <p>Welcome to test crawling.</p>
            <a href="/about">About</a>
          </main>
        </body></html>
        """,
        "https://example.com/about": """
        <html><head><title>About</title></head><body><main><p>About page.</p></main></body></html>
        """,
    }

    from data_pipeline import firecrawl_scraper

    def fake_get(url, timeout=None, headers=None):
        html = pages.get(url)
        if html is None:
            return _FakeResponse("", status_code=404)
        return _FakeResponse(html, status_code=200)

    monkeypatch.setattr(firecrawl_scraper.requests, "get", fake_get)

    resp = client.post(
        "/api/firecrawl/v2/scrape",
        json={
            "url": "https://example.com",
            "formats": ["markdown", "links", "metadata"],
            "onlyMainContent": True,
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert "Example Home" in body["data"]["markdown"]
    assert "https://example.com/about" in body["data"]["links"]
    assert body["data"]["metadata"]["sourceURL"] == "https://example.com"


def test_firecrawl_map_endpoint(monkeypatch, client):
    _signup(client)

    pages = {
        "https://example.com": """
        <html><body>
          <a href="/a">A</a>
          <a href="/b">B</a>
          <a href="/private/hidden">Hidden</a>
          <a href="https://external.com/out">Out</a>
        </body></html>
        """,
        "https://example.com/a": "<html><body><a href='/a/1'>A1</a></body></html>",
        "https://example.com/b": "<html><body><p>B page</p></body></html>",
        "https://example.com/a/1": "<html><body><p>A1 page</p></body></html>",
    }

    from data_pipeline import firecrawl_scraper

    def fake_get(url, timeout=None, headers=None):
        html = pages.get(url)
        if html is None:
            return _FakeResponse("", status_code=404)
        return _FakeResponse(html, status_code=200)

    monkeypatch.setattr(firecrawl_scraper.requests, "get", fake_get)

    resp = client.post(
        "/api/firecrawl/v2/map",
        json={
            "url": "https://example.com",
            "limit": 10,
            "maxDepth": 2,
            "excludePaths": ["/private"],
            "allowExternalLinks": False,
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    links = body["links"]
    assert body["success"] is True
    assert "https://example.com/a" in links
    assert "https://example.com/b" in links
    assert all("/private" not in u for u in links)
    assert all("external.com" not in u for u in links)


def test_firecrawl_crawl_job_endpoint(monkeypatch, client):
    _signup(client)

    pages = {
        "https://example.com": """
        <html><body>
          <main>
            <h1>Docs Home</h1>
            <a href="/docs/getting-started">Start</a>
            <a href="/blog/post-1">Blog</a>
          </main>
        </body></html>
        """,
        "https://example.com/docs/getting-started": """
        <html><body><main><h2>Getting Started</h2><p>Install and run.</p></main></body></html>
        """,
        "https://example.com/blog/post-1": """
        <html><body><main><h2>Post</h2><p>Blog content.</p></main></body></html>
        """,
    }

    from data_pipeline import firecrawl_scraper

    def fake_get(url, timeout=None, headers=None):
        html = pages.get(url)
        if html is None:
            return _FakeResponse("", status_code=404)
        return _FakeResponse(html, status_code=200)

    monkeypatch.setattr(firecrawl_scraper.requests, "get", fake_get)

    create_resp = client.post(
        "/api/firecrawl/v2/crawl",
        json={
            "url": "https://example.com",
            "limit": 10,
            "maxDepth": 2,
            "includePaths": ["/docs"],
            "scrapeOptions": {"formats": ["markdown", "metadata"]},
        },
    )
    assert create_resp.status_code == 200
    create_body = create_resp.get_json()
    assert create_body["success"] is True
    job_id = create_body["id"]

    status_resp = client.get(f"/api/firecrawl/v2/crawl/{job_id}")
    assert status_resp.status_code == 200
    status_body = status_resp.get_json()
    assert status_body["success"] is True
    assert status_body["status"] == "completed"
    assert status_body["total"] >= 1
    assert all(
        page["metadata"]["sourceURL"].startswith("https://example.com/docs")
        for page in status_body["data"]
    )


def test_firecrawl_engine_fallback_when_scrapy_unavailable(monkeypatch, client):
    _signup(client)

    pages = {
        "https://example.com": """
        <html><head><title>Home</title></head>
        <body><main><h1>Hello</h1><p>Engine fallback</p></main></body></html>
        """
    }

    from data_pipeline import firecrawl_scraper

    def fake_get(url, timeout=None, headers=None):
        html = pages.get(url)
        if html is None:
            return _FakeResponse("", status_code=404)
        return _FakeResponse(html, status_code=200)

    monkeypatch.setattr(firecrawl_scraper.requests, "get", fake_get)
    monkeypatch.setattr(firecrawl_scraper, "HAS_SCRAPY", False)

    resp = client.post(
        "/api/firecrawl/v2/scrape",
        json={
            "url": "https://example.com",
            "formats": ["markdown", "metadata"],
            "engine": "scrapy",
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    assert body["data"]["engine"] == "requests"
    assert body["data"]["metadata"]["engine"] == "requests"
