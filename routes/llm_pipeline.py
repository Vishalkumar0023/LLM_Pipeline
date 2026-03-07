import os
import json
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, g, send_file

from utils import jwt_required, get_user_folder

llm_bp = Blueprint("llm", __name__)


def get_llm_session_path(user_id):
    return os.path.join(get_user_folder(user_id), "llm_sessions.json")


def get_llm_runs_path(user_id):
    return os.path.join(get_user_folder(user_id), "llm_runs.json")


def load_llm_sessions(user_id):
    path = get_llm_session_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            pass
    return {}


def save_llm_sessions(user_id, sessions):
    path = get_llm_session_path(user_id)
    with open(path, "w") as f:
        json.dump(sessions, f)


def load_llm_runs(user_id):
    path = get_llm_runs_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            pass
    return []


def save_llm_runs(user_id, runs):
    path = get_llm_runs_path(user_id)
    with open(path, "w") as f:
        json.dump(runs, f)


@llm_bp.route("/llm")
@jwt_required
def llm_page():
    return render_template("llm.html", user=g.current_user)


@llm_bp.route("/api/llm/ingest", methods=["POST"])
@jwt_required
def llm_ingest():
    from data_pipeline.document_ingestor import DocumentIngestor

    session_id = str(uuid.uuid4())[:8]
    user_folder = get_user_folder(g.current_user.id)
    llm_folder = os.path.join(user_folder, "llm_temp", session_id)
    os.makedirs(llm_folder, exist_ok=True)

    sources = []
    files = request.files.getlist("files")
    for f in files:
        if f.filename:
            path = os.path.join(llm_folder, f.filename)
            f.save(path)
            sources.append(path)

    urls_json = request.form.get("urls", "[]")
    try:
        url_list = json.loads(urls_json)
        sources.extend(url_list)
    except:
        pass

    if not sources:
        return jsonify({"error": "No files or URLs provided"}), 400

    try:
        ingestor = DocumentIngestor()
        docs = ingestor.ingest(sources)
        stats = ingestor.get_stats()

        sessions = load_llm_sessions(g.current_user.id)
        sessions[session_id] = {
            "documents": docs,
            "user_id": g.current_user.id,
            "folder": llm_folder,
        }
        save_llm_sessions(g.current_user.id, sessions)

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "total_docs": len(docs),
                "total_chars": stats.get("total_chars", 0),
                "total_words": stats.get("total_words", 0),
                "source_types": list(stats.get("by_type", {}).keys()),
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@llm_bp.route("/api/llm/process", methods=["POST"])
@jwt_required
def llm_process():
    from data_pipeline.text_chunker import TextChunker
    from data_pipeline.instruct_formatter import InstructFormatter
    from data_pipeline.quality_scorer import QualityScorer

    data = request.json
    session_id = data.get("session_id")
    sessions = load_llm_sessions(g.current_user.id)

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session. Run ingestion first."}), 400
    session = sessions[session_id]
    if session["user_id"] != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403

    docs = session["documents"]

    try:
        chunker = TextChunker(
            method=data.get("chunk_method", "sliding_window"),
            chunk_size=data.get("chunk_size", 512),
            overlap=64,
        )
        chunks = chunker.chunk_documents(docs)

        formatter = InstructFormatter(template=data.get("template", "alpaca"))
        pairs = formatter.format_chunks(
            chunks,
            domain=data.get("domain", "general"),
            generate_qa=True,
            pairs_per_chunk=2,
        )

        min_quality = data.get("min_quality", 0.4)
        scorer = QualityScorer(min_quality_score=min_quality)
        scored = scorer.score(pairs)
        filtered = scorer.filter(scored, min_score=min_quality)

        scores = [p.get("quality", {}).get("overall_score", 0) for p in filtered]
        avg_quality = sum(scores) / len(scores) if scores else 0

        session["chunks"] = chunks
        session["pairs"] = pairs
        session["filtered_pairs"] = filtered
        session["template"] = data.get("template", "alpaca")
        session["avg_quality"] = avg_quality
        save_llm_sessions(g.current_user.id, sessions)

        sample = [
            {
                k: v
                for k, v in p.items()
                if k in ("instruction", "output", "input", "messages", "quality")
            }
            for p in filtered[:10]
        ]

        return jsonify(
            {
                "success": True,
                "total_chunks": len(chunks),
                "total_pairs": len(pairs),
                "filtered_pairs": len(filtered),
                "avg_quality": avg_quality,
                "sample_pairs": sample,
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@llm_bp.route("/api/llm/export", methods=["POST"])
@jwt_required
def llm_export():
    from data_pipeline.dataset_registry import DatasetRegistry
    from data_pipeline.finetune_config import FineTuneConfig
    from data_pipeline.instruct_formatter import InstructFormatter

    data = request.json
    session_id = data.get("session_id")
    sessions = load_llm_sessions(g.current_user.id)

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session. Run the pipeline first."}), 400
    session = sessions[session_id]
    if session["user_id"] != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403

    filtered = session.get("filtered_pairs", [])
    if not filtered:
        return jsonify({"error": "No processed data. Run process first."}), 400

    try:
        user_folder = get_user_folder(g.current_user.id)
        export_dir = os.path.join(user_folder, "llm_exports", session_id)
        os.makedirs(export_dir, exist_ok=True)

        data_path = os.path.join(export_dir, "training_data.jsonl")
        formatter = InstructFormatter(template=session.get("template", "alpaca"))
        formatter.export_jsonl(filtered, data_path, include_metadata=False)

        version = data.get("version", "v1.0.0")
        description = data.get("description", "")
        registry = DatasetRegistry(os.path.join(user_folder, "llm_registry"))
        try:
            registry.register(filtered, version=version, description=description)
        except:
            pass

        model = data.get("model", "meta-llama/Meta-Llama-3-8B")
        method = data.get("method", "lora")
        config = FineTuneConfig(model_name=model, method=method, backend="trl")
        config.export(export_dir, dataset_path="./training_data.jsonl")

        runs = load_llm_runs(g.current_user.id)
        runs.append(
            {
                "run_id": f"{version}-{session_id}",
                "session_id": session_id,
                "user_id": g.current_user.id,
                "sample_count": len(filtered),
                "avg_quality": session.get("avg_quality", 0),
                "template": session.get("template", "alpaca"),
                "model": model,
                "method": method,
                "version": version,
                "timestamp": datetime.now().isoformat(),
            }
        )
        save_llm_runs(g.current_user.id, runs)

        model_short = model.split("/")[-1] if "/" in model else model
        if len(model_short) > 15:
            model_short = model_short[:15] + "…"
        files = [
            {"name": "training_data.jsonl", "label": "Training Data (JSONL)"},
            {"name": "training_config.json", "label": "Config (JSON)"},
            {"name": "train.py", "label": "Train Script"},
            {"name": "requirements_training.txt", "label": "Requirements"},
        ]

        return jsonify(
            {
                "success": True,
                "sample_count": len(filtered),
                "version": version,
                "model_short": model_short,
                "method": method,
                "files": files,
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@llm_bp.route("/api/llm/download/<session_id>/<filename>")
@jwt_required
def llm_download(session_id, filename):
    user_folder = get_user_folder(g.current_user.id)
    file_path = os.path.join(user_folder, "llm_exports", session_id, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True, download_name=filename)


@llm_bp.route("/api/llm/runs")
@jwt_required
def llm_runs():
    runs = load_llm_runs(g.current_user.id)
    return jsonify({"runs": runs})
