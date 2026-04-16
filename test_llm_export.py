import io
import json
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


def _ingest_product_text(client):
    text = (
        "Product: Apple iPhone 17 Pro Max (Silver, 256 GB)\n"
        "Price: INR 149,900\n"
        "Rating: 4.8/5 (552 reviews)\n"
        "Description:\n"
        "256 GB ROM\n"
        "17.53 cm (6.9 inch) Super Retina XDR Display\n"
        "48MP + 48MP + 48MP | 18MP Front Camera\n"
        "A19 Chip, 6 Core Processor Processor\n"
        "Apple One (1) Year Limited Warranty\n"
    )
    data = {
        "files": (io.BytesIO(text.encode("utf-8")), "product.txt"),
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
    assert "run_id" in body

    runs_resp = client.get("/api/llm/runs")
    assert runs_resp.status_code == 200
    runs_body = runs_resp.get_json()
    assert isinstance(runs_body.get("runs"), list)
    assert any(r.get("run_id") == body["run_id"] for r in runs_body["runs"])

    summary_resp = client.get("/api/llm/db/summary")
    assert summary_resp.status_code == 200
    summary_body = summary_resp.get_json()
    assert summary_body["success"] is True
    assert summary_body["llm_run_count"] >= 1

    del_resp = client.delete(f"/api/llm/runs/{body['run_id']}")
    assert del_resp.status_code == 200
    del_body = del_resp.get_json()
    assert del_body["success"] is True


def test_file_manager_can_manage_llm_export_files(client):
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

    export_resp = client.post(
        "/api/llm/export",
        json={
            "session_id": session_id,
            "version": "vTEST-FILES",
            "model": "meta-llama/Meta-Llama-3-8B",
            "method": "lora",
        },
    )
    assert export_resp.status_code == 200

    files_resp = client.get("/api/user_files")
    assert files_resp.status_code == 200
    files = files_resp.get_json()
    assert isinstance(files, list)

    export_file = next(
        (
            f
            for f in files
            if f.get("path", "").startswith(f"llm_exports/{session_id}/")
            and f.get("name") == "training_data.jsonl"
        ),
        None,
    )
    assert export_file is not None

    download_resp = client.get(
        "/api/download_user_file", query_string={"path": export_file["path"]}
    )
    assert download_resp.status_code == 200
    content_disposition = download_resp.headers.get("Content-Disposition", "")
    assert "training_data.jsonl" in content_disposition

    delete_resp = client.post(
        "/api/delete_files", json={"filenames": [export_file["path"]]}
    )
    assert delete_resp.status_code == 200
    delete_body = delete_resp.get_json()
    assert export_file["path"] in delete_body.get("deleted", [])

    files_after_resp = client.get("/api/user_files")
    assert files_after_resp.status_code == 200
    files_after = files_after_resp.get_json()
    assert all(f.get("path") != export_file["path"] for f in files_after)


def test_file_manager_delete_supports_legacy_basename_payload(client):
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

    export_resp = client.post(
        "/api/llm/export",
        json={
            "session_id": session_id,
            "version": "vTEST-LEGACY-DELETE",
            "model": "meta-llama/Meta-Llama-3-8B",
            "method": "lora",
        },
    )
    assert export_resp.status_code == 200

    # Simulate old cached frontend that only sends basename.
    delete_resp = client.post(
        "/api/delete_files", json={"filenames": ["training_data.jsonl"]}
    )
    assert delete_resp.status_code == 200
    body = delete_resp.get_json()
    deleted = body.get("deleted", [])
    errors = body.get("errors", [])
    assert any(item.endswith("/training_data.jsonl") for item in deleted) or any(
        "Ambiguous file name" in err for err in errors
    )


def test_labeling_api_drives_best_jsonl_export(client):
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

    label_payload = {
        "session_id": session_id,
        "include_unlabeled": False,
        "labels": [{"pair_index": 0, "label": "accept"}],
    }
    label_resp = client.post("/api/llm/label", json=label_payload)
    assert label_resp.status_code == 200
    label_body = label_resp.get_json()
    assert label_body["success"] is True
    assert label_body["best_pairs_count"] == 1

    export_resp = client.post(
        "/api/llm/export",
        json={
            "session_id": session_id,
            "version": "vTEST-LABELED",
            "model": "meta-llama/Meta-Llama-3-8B",
            "method": "lora",
        },
    )
    assert export_resp.status_code == 200
    export_body = export_resp.get_json()
    assert export_body["success"] is True
    assert export_body["sample_count"] == 1
    assert export_body["export_mode"] == "labeled_best"

    download_resp = client.get(f"/api/llm/download/{session_id}/training_data.jsonl")
    assert download_resp.status_code == 200
    lines = [line for line in download_resp.data.decode("utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert "instruction" in record
    assert "output" in record


def test_export_with_use_llm_direct_mode_fallback_template(client):
    _signup_and_login(client)
    session_id = _ingest_product_text(client)

    process_resp = client.post(
        "/api/llm/process",
        json={
            "session_id": session_id,
            "use_llm": True,
            "domain": "general",
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
            "version": "vTEST-DIRECT-FALLBACK",
            "model": "meta-llama/Meta-Llama-3-8B",
            "method": "lora",
        },
    )
    assert export_resp.status_code == 200
    export_body = export_resp.get_json()
    assert export_body["success"] is True
    assert export_body["sample_count"] > 0
