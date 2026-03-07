import io
import os
import uuid

import pytest

from app import create_app
from extensions import db


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "test_pipeline_users.db"
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


def _signup_and_login(client):
    user_id = uuid.uuid4().hex[:8]
    payload = {
        "username": f"demotest_{user_id}",
        "email": f"demotest_{user_id}@example.com",
        "password": "password123",
    }
    resp = client.post("/signup", json=payload)
    assert resp.status_code == 200


def _ingest_small_text(client):
    data = {
        "files": (io.BytesIO(b"LLMs are neural networks trained on large text corpora."), "sample.txt"),
        "urls": "[]",
        "max_pages": "1",
    }
    resp = client.post("/api/llm/ingest", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["success"] is True
    return body["session_id"]


def test_export_error_when_all_pairs_filtered(client):
    _signup_and_login(client)
    session_id = _ingest_small_text(client)

    process_resp = client.post(
        "/api/llm/process",
        json={
            "session_id": session_id,
            "chunk_method": "sliding_window",
            "chunk_size": 256,
            "template": "alpaca",
            "min_quality": 1.1,
        },
    )
    assert process_resp.status_code == 200

    export_resp = client.post(
        "/api/llm/export",
        json={
            "session_id": session_id,
            "version": "vTEST-EMPTY",
            "model": "meta-llama/Meta-Llama-3-8B",
            "method": "lora",
        },
    )
    assert export_resp.status_code == 400
    body = export_resp.get_json()
    assert "0 pairs passed the quality filter" in body.get("error", "")


def test_export_success(client):
    _signup_and_login(client)
    session_id = _ingest_small_text(client)

    process_resp = client.post(
        "/api/llm/process",
        json={
            "session_id": session_id,
            "chunk_method": "sliding_window",
            "chunk_size": 256,
            "template": "alpaca",
            "min_quality": 0.0,
        },
    )
    assert process_resp.status_code == 200
    process_body = process_resp.get_json()
    assert process_body["success"] is True
    assert process_body["filtered_pairs"] > 0

    export_resp = client.post(
        "/api/llm/export",
        json={
            "session_id": session_id,
            "version": "vTEST-OK",
            "model": "meta-llama/Meta-Llama-3-8B",
            "method": "lora",
        },
    )
    assert export_resp.status_code == 200
    body = export_resp.get_json()
    assert body["success"] is True
    assert body["sample_count"] > 0
