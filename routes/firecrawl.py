import uuid
from threading import Lock
from typing import Any, Dict

from flask import Blueprint, jsonify, request, g

from data_pipeline.firecrawl_scraper import FirecrawlLikeScraper
from utils import jwt_required


firecrawl_bp = Blueprint("firecrawl", __name__)

_CRAWL_JOBS: Dict[str, Dict[str, Any]] = {}
_CRAWL_LOCK = Lock()


def _body() -> Dict[str, Any]:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _requested_engine(data: Dict[str, Any]) -> str:
    raw = data.get("engine")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    if "useScrapy" in data:
        return "scrapy" if _parse_bool(data.get("useScrapy")) else "requests"
    return "auto"


@firecrawl_bp.route("/api/firecrawl/v2/scrape", methods=["POST"])
@jwt_required
def firecrawl_scrape():
    data = _body()
    url = data.get("url")
    if not isinstance(url, str) or not url.strip():
        return jsonify({"error": "url is required"}), 400

    scraper = FirecrawlLikeScraper(
        timeout=int(data.get("timeout", 20)),
        engine=_requested_engine(data),
    )
    try:
        page = scraper.scrape(
            url=url.strip(),
            formats=data.get("formats"),
            only_main_content=_parse_bool(data.get("onlyMainContent"), True),
            include_tags=data.get("includeTags"),
            exclude_tags=data.get("excludeTags"),
            headers=data.get("headers"),
        )
        return jsonify({"success": True, "data": page})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@firecrawl_bp.route("/api/firecrawl/v2/map", methods=["POST"])
@jwt_required
def firecrawl_map():
    data = _body()
    url = data.get("url")
    if not isinstance(url, str) or not url.strip():
        return jsonify({"error": "url is required"}), 400

    scraper = FirecrawlLikeScraper(
        timeout=int(data.get("timeout", 20)),
        engine=_requested_engine(data),
    )
    try:
        links = scraper.map(
            url=url.strip(),
            limit=int(data.get("limit", 100)),
            max_depth=int(data.get("maxDepth", 2)),
            include_paths=data.get("includePaths"),
            exclude_paths=data.get("excludePaths"),
            allow_external_links=_parse_bool(data.get("allowExternalLinks"), False),
            headers=data.get("headers"),
        )
        return jsonify(
            {
                "success": True,
                "links": links,
                "total": len(links),
                "engine": scraper.engine,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@firecrawl_bp.route("/api/firecrawl/v2/crawl", methods=["POST"])
@jwt_required
def firecrawl_crawl():
    data = _body()
    url = data.get("url")
    if not isinstance(url, str) or not url.strip():
        return jsonify({"error": "url is required"}), 400

    scraper = FirecrawlLikeScraper(
        timeout=int(data.get("timeout", 20)),
        engine=_requested_engine(data),
    )
    job_id = str(uuid.uuid4())

    with _CRAWL_LOCK:
        _CRAWL_JOBS[job_id] = {"id": job_id, "status": "running", "created_by": g.current_user.id}

    try:
        result = scraper.crawl(
            url=url.strip(),
            limit=int(data.get("limit", 25)),
            max_depth=int(data.get("maxDepth", 2)),
            scrape_options=data.get("scrapeOptions") or {},
            include_paths=data.get("includePaths"),
            exclude_paths=data.get("excludePaths"),
            allow_external_links=_parse_bool(data.get("allowExternalLinks"), False),
            headers=data.get("headers"),
        )
        with _CRAWL_LOCK:
            _CRAWL_JOBS[job_id].update(result)
            _CRAWL_JOBS[job_id]["status"] = "completed"
        return jsonify(
            {
                "success": True,
                "id": job_id,
                "status": "completed",
                "url": f"/api/firecrawl/v2/crawl/{job_id}",
            }
        )
    except Exception as exc:
        with _CRAWL_LOCK:
            _CRAWL_JOBS[job_id]["status"] = "failed"
            _CRAWL_JOBS[job_id]["error"] = str(exc)
        return jsonify({"error": str(exc), "id": job_id}), 400


@firecrawl_bp.route("/api/firecrawl/v2/crawl/<job_id>", methods=["GET"])
@jwt_required
def firecrawl_crawl_status(job_id: str):
    with _CRAWL_LOCK:
        job = _CRAWL_JOBS.get(job_id)

    if not job:
        return jsonify({"error": "crawl job not found"}), 404
    if int(job.get("created_by", 0)) != int(g.current_user.id):
        return jsonify({"error": "access denied"}), 403

    return jsonify({"success": True, **job})
