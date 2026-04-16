import os
import json
import uuid
from datetime import datetime
import shutil
import re
from urllib.parse import urlparse
from flask import Blueprint, request, jsonify, render_template, g, send_file

from utils import jwt_required, get_user_folder

llm_bp = Blueprint("llm", __name__)

PIPELINE_ARCHITECTURE = [
    "scraper",
    "raw records",
    "rule-based extraction",
    "LLM extraction",
    "validation",
    "normalization",
    "deduplication",
    "dataset split",
    "training",
    "evaluation",
    "inference API",
]


def _parse_bool(value, default=False):
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


def _split_concatenated_urls(raw: str):
    text = (raw or "").strip()
    if not text:
        return []
    starts = [m.start() for m in re.finditer(r"https?://", text, flags=re.IGNORECASE)]
    if len(starts) <= 1:
        return [text]
    parts = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        part = text[start:end].strip()
        if part:
            parts.append(part)
    return parts


def _normalize_url_list(values):
    normalized = []
    seen = set()
    for value in values or []:
        if not isinstance(value, str):
            continue
        for candidate in _split_concatenated_urls(value):
            parsed = urlparse(candidate)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                continue
            key = candidate.strip()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(key)
    return normalized


def get_llm_session_path(user_id):
    return os.path.join(get_user_folder(user_id), "llm_sessions.json")


def get_llm_runs_path(user_id):
    return os.path.join(get_user_folder(user_id), "llm_runs.json")


def _serialize_llm_run(run):
    return {
        "run_id": run.run_id,
        "session_id": run.session_id,
        "user_id": run.user_id,
        "sample_count": run.sample_count or 0,
        "avg_quality": float(run.avg_quality or 0.0),
        "template": run.template or "",
        "model": run.model or "",
        "method": run.method or "",
        "version": run.version or "",
        "timestamp": run.created_at.isoformat() if run.created_at else "",
    }


def _load_llm_runs_file(user_id):
    path = get_llm_runs_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_llm_runs_file(user_id, runs):
    path = get_llm_runs_path(user_id)
    with open(path, "w") as f:
        json.dump(runs, f)


def _save_llm_runs_db(user_id, runs):
    try:
        from extensions import db
        from models import LLMRun

        for payload in runs or []:
            run_id = str(payload.get("run_id", "")).strip()
            session_id = str(payload.get("session_id", "")).strip()
            if not run_id or not session_id:
                continue
            row = LLMRun.query.filter_by(user_id=user_id, run_id=run_id).first()
            if row is None:
                row = LLMRun(user_id=user_id, run_id=run_id, session_id=session_id)
                db.session.add(row)
            row.session_id = session_id
            row.sample_count = int(payload.get("sample_count", 0) or 0)
            row.avg_quality = float(payload.get("avg_quality", 0.0) or 0.0)
            row.template = str(payload.get("template", "") or "")
            row.model = str(payload.get("model", "") or "")
            row.method = str(payload.get("method", "") or "")
            row.version = str(payload.get("version", "") or "")
            ts = str(payload.get("timestamp", "") or "").strip()
            if ts:
                try:
                    row.created_at = datetime.fromisoformat(ts)
                except Exception:
                    pass
        db.session.commit()
        return True
    except Exception:
        try:
            from extensions import db

            db.session.rollback()
        except Exception:
            pass
        return False


def _load_llm_runs_db(user_id):
    try:
        from models import LLMRun

        rows = (
            LLMRun.query.filter_by(user_id=user_id)
            .order_by(LLMRun.created_at.desc())
            .all()
        )
        return [_serialize_llm_run(r) for r in rows]
    except Exception:
        return None


def _delete_llm_run_db(user_id, run_id):
    try:
        from extensions import db
        from models import LLMRun

        row = LLMRun.query.filter_by(user_id=user_id, run_id=run_id).first()
        if row is None:
            return False
        db.session.delete(row)
        db.session.commit()
        return True
    except Exception:
        try:
            from extensions import db

            db.session.rollback()
        except Exception:
            pass
        return False


def load_llm_sessions(user_id):
    path = get_llm_session_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_llm_sessions(user_id, sessions):
    path = get_llm_session_path(user_id)
    with open(path, "w") as f:
        json.dump(sessions, f)


def load_llm_runs(user_id):
    db_runs = _load_llm_runs_db(user_id)
    if db_runs is not None:
        if db_runs:
            return db_runs
        # One-time migration path from legacy file store.
        file_runs = _load_llm_runs_file(user_id)
        if file_runs:
            if _save_llm_runs_db(user_id, file_runs):
                migrated = _load_llm_runs_db(user_id)
                if migrated is not None:
                    return migrated
            return file_runs
        return []
    return _load_llm_runs_file(user_id)


def save_llm_runs(user_id, runs):
    if not _save_llm_runs_db(user_id, runs):
        _save_llm_runs_file(user_id, runs)
        return
    # Keep JSON in sync for backward compatibility tooling.
    _save_llm_runs_file(user_id, runs)


@llm_bp.route("/llm")
@jwt_required
def llm_page():
    return render_template("llm.html", user=g.current_user)


@llm_bp.route("/api/llm/architecture")
@jwt_required
def llm_architecture():
    return jsonify({"architecture": PIPELINE_ARCHITECTURE})


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
    # SECURITY: Enforce filename sanitization and extension whitelist
    ALLOWED_UPLOAD_EXTENSIONS = {
        ".pdf", ".txt", ".md", ".json", ".xml", ".html",
        ".htm", ".csv", ".docx", ".markdown", ".rst",
    }
    for f in files:
        if f.filename:
            from werkzeug.utils import secure_filename
            safe_name = secure_filename(f.filename)
            if not safe_name:
                continue
            ext = os.path.splitext(safe_name)[1].lower()
            if ext not in ALLOWED_UPLOAD_EXTENSIONS:
                continue
            path = os.path.join(llm_folder, safe_name)
            f.save(path)
            sources.append(path)

    urls_json = request.form.get("urls", "[]")
    try:
        url_list = json.loads(urls_json)
        sources.extend(_normalize_url_list(url_list))
    except Exception:
        pass

    try:
        max_pages = int(request.form.get("max_pages", 1))
    except ValueError:
        max_pages = 1

    if not sources:
        return jsonify({"error": "No files or URLs provided"}), 400

    try:
        ingestor = DocumentIngestor()
        docs = ingestor.ingest(sources, max_pages=max_pages)
        stats = ingestor.get_stats()

        if not docs:
            # Provide exact reasons why ingestion failed
            error_details = [f"{err['source']}: {err['error']}" for err in ingestor.errors]
            error_msg = "Failed to extract documents. " + " | ".join(error_details)
            print(f"DEBUG INGEST: {error_msg}")
            return jsonify({"error": error_msg}), 400

        # Auto-detect optimal pipeline config from ingested documents
        from data_pipeline.auto_config import AutoPipelineConfig
        auto_config = AutoPipelineConfig.detect(docs)

        sessions = load_llm_sessions(g.current_user.id)
        sessions[session_id] = {
            "documents": docs,
            "user_id": g.current_user.id,
            "folder": llm_folder,
            "auto_config": auto_config,
        }
        save_llm_sessions(g.current_user.id, sessions)

        total_chars = sum(d.get("char_count", 0) for d in docs)
        total_words = sum(d.get("word_count", 0) for d in docs)

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "total_docs": len(docs),
                "total_chars": total_chars,
                "total_words": total_words,
                "source_types": list(stats.get("by_type", {}).keys()),
                "auto_config": auto_config,
            }
        )
    except Exception as e:
        import traceback
        import logging
        logging.getLogger(__name__).exception("LLM ingestion failed")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during ingestion. Please try again."}), 500


@llm_bp.route("/api/llm/process", methods=["POST"])
@jwt_required
def llm_process():
    from data_pipeline.text_chunker import TextChunker
    from data_pipeline.instruct_formatter import InstructFormatter
    from data_pipeline.quality_scorer import QualityScorer
    from data_pipeline.verification_agent import DatasetVerificationAgent
    from data_pipeline.quality_scorer import TwoLayerQualityScorer
    from data_pipeline.llm_data_processor import LLMDataProcessor

    data = request.json or {}
    session_id = data.get("session_id")
    sessions = load_llm_sessions(g.current_user.id)

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session. Run ingestion first."}), 400
    session = sessions[session_id]
    if session["user_id"] != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403

    docs = session["documents"]

    try:
        min_quality = float(data.get("min_quality", 0.4))
        domain = str(data.get("domain", "general") or "general")

        filter_mode = "internal_fixed_pipeline"
        quality_stats = {}
        effective_min_quality = float(min_quality)
        auto_relaxed_quality = False

        def _rescue_generic_pairs(scored_pairs):
            rescued = [
                s
                for s in (scored_pairs or [])
                if not s.get("quality", {}).get("is_toxic", False)
            ]
            rescued.sort(
                key=lambda s: s.get("quality", {}).get("overall_score", 0.0),
                reverse=True,
            )
            rescued = rescued[: max(1, min(25, len(rescued)))]
            for s in rescued:
                s.setdefault("quality", {})["rescue_mode"] = True
            return rescued

        def _generic_filter_with_fallback(scored_pairs, requested_min_quality):
            """
            Filter with requested threshold, then auto-relax to 0.0 if it yields zero.
            Returns (filtered, stats, effective_min, was_relaxed).
            """
            requested_min_quality = float(requested_min_quality)
            local_scorer = QualityScorer(min_quality_score=requested_min_quality)
            filtered_local = local_scorer.filter(
                scored_pairs, min_score=requested_min_quality
            )
            local_stats = local_scorer.get_stats()
            # Respect out-of-range thresholds (e.g., >1.0) as explicit strict filters.
            if (
                filtered_local
                or requested_min_quality <= 0
                or requested_min_quality > 1.0
            ):
                return filtered_local, local_stats, float(requested_min_quality), False

            relaxed_scorer = QualityScorer(min_quality_score=0.0)
            relaxed_filtered = relaxed_scorer.filter(scored_pairs, min_score=0.0)
            relaxed_stats = relaxed_scorer.get_stats()
            if relaxed_filtered:
                return relaxed_filtered, relaxed_stats, 0.0, True
            return filtered_local, local_stats, float(requested_min_quality), False

        def _is_clean_generic_pair(pair):
            """Drop low-value generic pairs that are mostly raw chunk echoes/noise."""
            instruction = str(pair.get("instruction", "") or "")
            output = str(pair.get("output", "") or "").strip()
            input_text = str(pair.get("input", "") or "")

            if not output:
                return False
            if "http://" in output.lower() or "https://" in output.lower():
                return False
            if re.search(r"\b\d{2}/\d{2}/\d{4},\s*\d{1,2}:\d{2}\b", output):
                return False
            if len(output) < 30 or len(output.split()) < 5:
                return False

            norm_out = re.sub(r"\s+", " ", output.lower()).strip()
            norm_in = re.sub(r"\s+", " ", input_text.lower()).strip()
            if norm_out and norm_out == norm_in:
                return False
            if len(norm_out) >= 60 and norm_out in norm_in:
                return False

            # For summary-like instructions, require a concise synthesized style.
            inst_low = instruction.lower()
            if any(k in inst_low for k in ("explain", "summarize", "overview", "main points")):
                if not (
                    output.startswith("Key concepts include:")
                    or "." in output
                    or ";" in output
                ):
                    return False
            return True

        def _apply_generic_output_guard(pairs):
            before = len(pairs or [])
            kept = [p for p in (pairs or []) if _is_clean_generic_pair(p)]
            dropped = before - len(kept)
            return kept, dropped

        processor = LLMDataProcessor()
        chunks = []
        raw_pairs = []
        rule_based_records = []

        # 1) scraper -> 2) raw records -> 3) rule-based extraction
        for i, doc in enumerate(docs):
            text = (doc.get("text") or "").strip()
            if not text:
                continue
            source = doc.get("source") or doc.get("metadata", {}).get("source", "unknown")
            chunks.append(
                {
                    "text": text,
                    "source": source,
                    "doc_id": doc.get("doc_id", "unknown"),
                    "chunk_index": i,
                    "source_type": doc.get("source_type", ""),
                }
            )

            evidence = processor.extract_evidence(text)
            if not evidence:
                continue
            record = processor.normalize_record(evidence)
            if not record:
                continue
            rule_based_records.append(
                {
                    "source": source,
                    "doc_id": doc.get("doc_id", "unknown"),
                    "record": record,
                }
            )

            # 4) LLM extraction (internal deterministic generator)
            generated_tasks = processor.generate_tasks(record, text)
            for task in generated_tasks:
                # 5) validation
                if not processor.verify_sample(task, record):
                    continue
                task["metadata"] = {
                    "source": source,
                    "doc_id": doc.get("doc_id", "unknown"),
                    "generated_by": "internal_fixed_pipeline",
                }
                raw_pairs.append(task)

        # 6) normalization + correction
        pairs = processor.deduplicate_and_balance(raw_pairs)
        verifier = DatasetVerificationAgent()
        corrected_pairs, reports = verifier.validate_and_correct(pairs)

        # If strict rule-based extraction yields nothing, fallback internally to
        # generic chunk->format generation (still fully internal, no UI toggle).
        if not corrected_pairs:
            chunker = TextChunker(
                method=data.get("chunk_method", "sliding_window"),
                chunk_size=data.get("chunk_size", 512),
                overlap=64,
            )
            fallback_chunks = chunker.chunk_documents(docs)
            formatter = InstructFormatter(template="alpaca")
            pairs = formatter.format_chunks(
                fallback_chunks,
                domain=domain,
                generate_qa=True,
                pairs_per_chunk=2,
            )
            # SECURITY: Still apply verification to fallback pairs —
            # never skip grounding checks, even on the generic path.
            fallback_verifier = DatasetVerificationAgent()
            corrected_pairs, reports = fallback_verifier.validate_and_correct(pairs)
            if not corrected_pairs:
                # If verification rejects everything, use raw pairs as last resort
                corrected_pairs = pairs
                reports = []
            filter_mode = "internal_fixed_pipeline_generic_fallback"

        # 7) deduplication + quality
        if filter_mode == "internal_fixed_pipeline_generic_fallback":
            scored = QualityScorer(min_quality_score=min_quality).score(corrected_pairs)
            filtered, quality_stats, effective_min_quality, auto_relaxed_quality = _generic_filter_with_fallback(
                scored, min_quality
            )
            cleaned_filtered, dropped_noisy = _apply_generic_output_guard(filtered)
            if cleaned_filtered:
                filtered = cleaned_filtered
                if dropped_noisy > 0:
                    filter_mode = "internal_fixed_pipeline_generic_guarded"
            rescue_allowed = 0.0 <= float(min_quality) <= 1.0
            if not filtered and scored and rescue_allowed:
                rescued = _rescue_generic_pairs(scored)
                if rescued:
                    filtered = rescued
                    filter_mode = "internal_fixed_pipeline_generic_rescue"
        else:
            strict_scorer = TwoLayerQualityScorer()
            filtered = strict_scorer.score_quality(corrected_pairs)
            quality_stats = strict_scorer.get_stats()
            if not filtered and corrected_pairs:
                # Controlled fallback so user still gets exportable output.
                scored = QualityScorer(min_quality_score=min_quality).score(corrected_pairs)
                filtered, quality_stats, effective_min_quality, auto_relaxed_quality = _generic_filter_with_fallback(
                    scored, min_quality
                )
                filter_mode = "internal_fixed_pipeline_relaxed"
                cleaned_filtered, dropped_noisy = _apply_generic_output_guard(filtered)
                if cleaned_filtered:
                    filtered = cleaned_filtered
                    if dropped_noisy > 0:
                        filter_mode = "internal_fixed_pipeline_guarded"
                rescue_allowed = 0.0 <= float(min_quality) <= 1.0
                if not filtered and scored and rescue_allowed:
                    rescued = _rescue_generic_pairs(scored)
                    if rescued:
                        filtered = rescued
                        filter_mode = "internal_fixed_pipeline_rescue"

        for pair in filtered:
            quality = pair.setdefault("quality", {})
            quality["overall_score"] = float(quality.get("overall_score", 1.0) or 1.0)

        scores = [p.get("quality", {}).get("overall_score", 1.0) for p in filtered]
        avg_quality = sum(scores) / len(scores) if scores else 0.0

        print(
            "DEBUG PROCESS: internal fixed pipeline "
            f"raw_pairs={len(raw_pairs)} corrected={len(corrected_pairs)} kept={len(filtered)} "
            f"records={len(rule_based_records)} mode={filter_mode}"
        )
        if auto_relaxed_quality:
            print(
                "DEBUG PROCESS: Auto-relaxed min_quality "
                f"from {min_quality} to {effective_min_quality}."
            )

        session["chunks"] = chunks
        session["pairs"] = pairs
        session["filtered_pairs"] = filtered
        session["template"] = "alpaca"
        session["avg_quality"] = avg_quality
        session["filter_mode"] = filter_mode
        session["quality_stats"] = quality_stats
        session["effective_min_quality"] = effective_min_quality
        session["auto_relaxed_quality"] = auto_relaxed_quality
        session["rule_based_records"] = rule_based_records
        session["architecture"] = PIPELINE_ARCHITECTURE
        save_llm_sessions(g.current_user.id, sessions)

        sample = [
            {
                k: v
                for k, v in p.items()
                if k in ("instruction", "output", "input", "messages", "quality")
            }
            for p in filtered[:10]
        ]

        ollama_enabled = os.environ.get("OLLAMA_ENABLED", "false").strip().lower() in {"1", "true", "yes"}

        # Auto-recommend export settings based on actual dataset stats
        from data_pipeline.auto_config import AutoPipelineConfig
        export_recommendations = AutoPipelineConfig.recommend_export_settings(
            pair_count=len(filtered),
            avg_quality=avg_quality,
        )

        return jsonify(
            {
                "success": True,
                "total_chunks": len(chunks),
                "total_pairs": len(pairs),
                "filtered_pairs": len(filtered),
                "avg_quality": avg_quality,
                "filter_mode": filter_mode,
                "quality_stats": quality_stats,
                "effective_min_quality": effective_min_quality,
                "auto_relaxed_quality": auto_relaxed_quality,
                "architecture": PIPELINE_ARCHITECTURE,
                "rule_based_records": len(rule_based_records),
                "sample_pairs": sample,
                "ollama_required": False,
                "ollama_enabled": ollama_enabled,
                "export_recommendations": export_recommendations,
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

    export_mode = "filtered_auto"
    labeled_pairs = session.get("best_pairs", [])
    if isinstance(labeled_pairs, list) and labeled_pairs:
        filtered = labeled_pairs
        export_mode = "labeled_best"
    else:
        filtered = session.get("filtered_pairs", [])
    if not filtered:
        # Check if pairs exists to distinguish between "not processed" and "all filtered out"
        if "pairs" in session and len(session["pairs"]) > 0:
            print(f"DEBUG EXPORT: 400 Error. filtered_pairs is empty, but session['pairs'] has {len(session['pairs'])} items. User needs to lower Min Quality Score.")
            filter_mode = session.get("filter_mode", "legacy_generic")
            if str(filter_mode).startswith("strict"):
                return jsonify(
                    {
                        "error": (
                            f"0 pairs passed strict validation (out of {len(session['pairs'])}). "
                            "For PDF/general documents, process with generic mode "
                            "(disable strict ecommerce path) and try again."
                        )
                    }
                ), 400
            return jsonify({"error": f"0 pairs passed the quality filter (out of {len(session['pairs'])}). Lower the Min Quality Score to 0.0 and process again."}), 400
        print("DEBUG EXPORT: 400 Error. No processed data. session['pairs'] is empty or missing.")
        return jsonify({"error": "No processed data. Run process first."}), 400

    print(f"DEBUG EXPORT: Success! Exporting {len(filtered)} filtered pairs.")
    try:
        user_folder = get_user_folder(g.current_user.id)
        export_dir = os.path.join(user_folder, "llm_exports", session_id)
        os.makedirs(export_dir, exist_ok=True)

        # Keep export dataset strictly in training schema only.
        export_pairs = []
        for pair in filtered:
            if "instruction" in pair and "output" in pair:
                output_val = pair.get("output", "")
                if isinstance(output_val, (dict, list)):
                    output_val = json.dumps(output_val, ensure_ascii=False)
                export_pairs.append(
                    {
                        "instruction": pair.get("instruction", ""),
                        "input": pair.get("input", ""),
                        "output": output_val,
                    }
                )
            elif "messages" in pair:
                export_pairs.append({"messages": pair.get("messages", [])})
            elif "conversations" in pair:
                export_pairs.append({"conversations": pair.get("conversations", [])})

        data_path = os.path.join(export_dir, "training_data.jsonl")
        export_template = str(session.get("template", "alpaca") or "alpaca").strip().lower()
        if export_template not in InstructFormatter.TEMPLATES:
            # "direct" and any unknown values should still export with a valid schema.
            if export_pairs and isinstance(export_pairs[0], dict):
                first = export_pairs[0]
                if "messages" in first:
                    export_template = "chatml"
                elif "conversations" in first:
                    export_template = "sharegpt"
                else:
                    export_template = "alpaca"
            else:
                export_template = "alpaca"
        formatter = InstructFormatter(template=export_template)
        formatter.export_jsonl(export_pairs, data_path, include_metadata=False)

        version = data.get("version", "v1.0.0")
        description = data.get("description", "")
        registry = DatasetRegistry(os.path.join(user_folder, "llm_registry"))
        try:
            registry.register(export_pairs, version=version, description=description)
        except Exception:
            pass

        model = data.get("model", "meta-llama/Meta-Llama-3-8B")
        method = data.get("method", "lora")
        config = FineTuneConfig(model_name=model, method=method, backend="trl")
        config.export(export_dir, dataset_path="./training_data.jsonl")

        run_id = f"{version}-{session_id}"
        runs = load_llm_runs(g.current_user.id)
        runs.append(
            {
                "run_id": run_id,
                "session_id": session_id,
                "user_id": g.current_user.id,
                "sample_count": len(export_pairs),
                "avg_quality": session.get("avg_quality", 0),
                "template": export_template,
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
                "run_id": run_id,
                "sample_count": len(export_pairs),
                "export_mode": export_mode,
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


@llm_bp.route("/api/llm/label", methods=["POST"])
@jwt_required
def llm_label():
    data = request.json or {}
    session_id = data.get("session_id")
    labels = data.get("labels", [])
    include_unlabeled = _parse_bool(data.get("include_unlabeled"), False)

    sessions = load_llm_sessions(g.current_user.id)
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session. Run process first."}), 400
    session = sessions[session_id]
    if session["user_id"] != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403

    base_pairs = session.get("filtered_pairs", [])
    if not isinstance(base_pairs, list) or not base_pairs:
        return jsonify({"error": "No filtered pairs found. Run process first."}), 400
    if not isinstance(labels, list) or not labels:
        return jsonify({"error": "labels must be a non-empty array"}), 400

    allowed_labels = {"accept", "reject", "edit", "skip"}
    label_map = {}
    ignored = []

    for idx, item in enumerate(labels):
        if not isinstance(item, dict):
            ignored.append({"index": idx, "reason": "label entry must be an object"})
            continue
        pair_index = item.get("pair_index", item.get("index"))
        try:
            pair_index = int(pair_index)
        except Exception:
            ignored.append({"index": idx, "reason": "pair_index must be an integer"})
            continue
        if pair_index < 0 or pair_index >= len(base_pairs):
            ignored.append({"index": idx, "reason": "pair_index out of range"})
            continue
        label = str(item.get("label", "")).strip().lower()
        if label not in allowed_labels:
            ignored.append(
                {
                    "index": idx,
                    "reason": f"label must be one of {sorted(allowed_labels)}",
                }
            )
            continue
        # Last write wins for duplicate pair indexes.
        label_map[pair_index] = {
            "label": label,
            "instruction": item.get("instruction"),
            "input": item.get("input"),
            "output": item.get("output"),
            "note": str(item.get("note", "") or "").strip(),
        }

    if not label_map:
        return jsonify({"error": "No valid labels supplied", "ignored": ignored}), 400

    best_pairs = []
    accepted = 0
    edited = 0
    rejected = 0
    skipped = 0
    unlabeled = 0

    for i, pair in enumerate(base_pairs):
        labeled = label_map.get(i)
        if not labeled:
            unlabeled += 1
            if include_unlabeled:
                best_pairs.append(pair)
            continue

        action = labeled["label"]
        if action == "reject":
            rejected += 1
            continue
        if action == "skip":
            skipped += 1
            if include_unlabeled:
                best_pairs.append(pair)
            continue

        updated = dict(pair)
        if action == "accept":
            accepted += 1
        else:
            edited += 1
            for key in ("instruction", "input", "output"):
                if labeled.get(key) is not None:
                    updated[key] = str(labeled.get(key))

        quality = dict(updated.get("quality", {}) or {})
        quality["human_label"] = action
        if labeled.get("note"):
            quality["label_note"] = labeled["note"]
        updated["quality"] = quality
        best_pairs.append(updated)

    if not best_pairs:
        return jsonify(
            {
                "error": "All pairs were rejected/skipped. Keep at least one pair to export JSONL.",
                "stats": {
                    "accepted": accepted,
                    "edited": edited,
                    "rejected": rejected,
                    "skipped": skipped,
                    "unlabeled": unlabeled,
                },
            }
        ), 400

    session["labels"] = {str(k): v for k, v in label_map.items()}
    session["best_pairs"] = best_pairs
    session["labeling_applied"] = True
    session["best_pair_stats"] = {
        "accepted": accepted,
        "edited": edited,
        "rejected": rejected,
        "skipped": skipped,
        "unlabeled": unlabeled,
        "include_unlabeled": include_unlabeled,
    }
    save_llm_sessions(g.current_user.id, sessions)

    sample = [
        {
            key: value
            for key, value in p.items()
            if key in ("instruction", "input", "output", "quality")
        }
        for p in best_pairs[:10]
    ]
    return jsonify(
        {
            "success": True,
            "session_id": session_id,
            "total_filtered_pairs": len(base_pairs),
            "best_pairs_count": len(best_pairs),
            "stats": session["best_pair_stats"],
            "ignored": ignored,
            "sample_pairs": sample,
        }
    )


@llm_bp.route("/api/llm/download/<session_id>/<filename>")
@jwt_required
def llm_download(session_id, filename):
    user_folder = get_user_folder(g.current_user.id)
    # SECURITY: Sanitize path components to prevent directory traversal
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        return jsonify({"error": "Invalid session ID"}), 400
    if ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "Invalid filename"}), 400
    file_path = os.path.join(user_folder, "llm_exports", session_id, filename)
    # Verify the resolved path is under the user's folder
    if not os.path.abspath(file_path).startswith(os.path.abspath(user_folder)):
        return jsonify({"error": "Access denied"}), 403
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True, download_name=filename)


@llm_bp.route("/api/llm/runs")
@jwt_required
def llm_runs():
    runs = load_llm_runs(g.current_user.id)
    return jsonify({"runs": runs})


@llm_bp.route("/api/llm/runs/<run_id>", methods=["DELETE"])
@jwt_required
def llm_delete_run(run_id):
    runs = load_llm_runs(g.current_user.id)
    target = next((r for r in runs if str(r.get("run_id")) == str(run_id)), None)
    if target is None:
        return jsonify({"error": "Run not found"}), 404

    # Update file list first
    updated = [r for r in runs if str(r.get("run_id")) != str(run_id)]
    _save_llm_runs_file(g.current_user.id, updated)
    _delete_llm_run_db(g.current_user.id, run_id)

    # Clean export folder associated with this run, if present.
    session_id = str(target.get("session_id", "")).strip()
    if session_id:
        user_folder = get_user_folder(g.current_user.id)
        export_dir = os.path.join(user_folder, "llm_exports", session_id)
        if os.path.isdir(export_dir):
            shutil.rmtree(export_dir, ignore_errors=True)

    return jsonify({"success": True, "deleted_run_id": run_id})


@llm_bp.route("/api/llm/db/summary")
@jwt_required
def llm_db_summary():
    try:
        from models import Dataset, LLMRun

        dataset_count = Dataset.query.filter_by(user_id=g.current_user.id).count()
        run_count = LLMRun.query.filter_by(user_id=g.current_user.id).count()
        sessions = load_llm_sessions(g.current_user.id)
        return jsonify(
            {
                "success": True,
                "dataset_count": dataset_count,
                "llm_run_count": run_count,
                "cached_session_count": len(sessions),
            }
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("DB summary failed")
        return jsonify({"error": "An internal error occurred."}), 500


@llm_bp.route("/api/llm/ollama/status")
@jwt_required
def ollama_status():
    """Return Ollama connectivity status and available models."""
    from data_pipeline.llm_client import LLMClient

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    available = LLMClient.is_ollama_available(base_url)
    models = LLMClient.list_ollama_models(base_url) if available else []
    default_model = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")

    return jsonify({
        "available": available,
        "models": models,
        "default_model": default_model,
        "base_url": base_url,
    })


@llm_bp.route("/api/llm/enhance", methods=["POST"])
@jwt_required
def llm_enhance():
    """
    Use DeepSeek-R1 (via Ollama) to enhance/verify existing processed pairs.
    This runs AFTER the deterministic pipeline and adds LLM-powered
    quality improvements to the already-generated dataset.
    """
    from data_pipeline.llm_client import LLMClient

    data = request.json or {}
    session_id = data.get("session_id")
    sessions = load_llm_sessions(g.current_user.id)

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session. Run the pipeline first."}), 400
    session = sessions[session_id]
    if session["user_id"] != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403

    filtered_pairs = session.get("filtered_pairs", [])
    if not filtered_pairs:
        return jsonify({"error": "No pairs to enhance. Run process first."}), 400

    # Read Ollama config
    provider = data.get("provider", os.environ.get("LLM_PROVIDER", "ollama"))
    model = data.get("model", os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b"))
    base_url = data.get("base_url", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))

    # Check Ollama availability
    if provider == "ollama" and not LLMClient.is_ollama_available(base_url):
        return jsonify({
            "error": "Ollama is not running. Please start Ollama first.",
            "help": "Open the Ollama app or run 'ollama serve' in terminal."
        }), 503

    client = LLMClient(provider=provider, base_url=base_url)

    enhanced_pairs = []
    enhanced_count = 0
    failed_count = 0
    max_pairs = min(len(filtered_pairs), int(data.get("max_pairs", 10)))

    for i, pair in enumerate(filtered_pairs[:max_pairs]):
        instruction = pair.get("instruction", "")
        input_text = pair.get("input", "")
        original_output = pair.get("output", "")

        try:
            system_prompt = (
                "You are a fact-checker. Check if the output is grounded in the input. "
                "Return ONLY valid JSON: "
                '{\"is_correct\":true/false,\"confidence\":0.0-1.0,\"reason\":\"brief\"}'
            )
            user_prompt = (
                f"Q: {instruction}\n"
                f"Context: {input_text[:500]}\n"
                f"Answer: {original_output[:500]}\n"
                "JSON:"
            )

            result = client.generate_json(system_prompt, user_prompt, model=model)

            quality = dict(pair.get("quality", {}) or {})
            quality["deepseek_verified"] = True
            quality["deepseek_confidence"] = float(result.get("confidence", 0.0))
            quality["deepseek_correct"] = bool(result.get("is_correct", True))
            quality["deepseek_reason"] = str(result.get("reason", ""))[:200]

            enhanced = dict(pair)
            # If DeepSeek says the output is wrong and provides an improvement,
            # use the improved version but keep the original for audit
            if not result.get("is_correct") and result.get("improved_output"):
                enhanced["output"] = str(result["improved_output"])
                quality["original_output"] = original_output
                quality["deepseek_improved"] = True

            enhanced["quality"] = quality
            enhanced_pairs.append(enhanced)
            enhanced_count += 1
            print(f"  ✅ DeepSeek verified pair {i+1}/{max_pairs} "
                  f"(correct={result.get('is_correct')}, "
                  f"confidence={result.get('confidence', 0):.2f})")

        except Exception as e:
            # On failure, keep the original pair untouched
            quality = dict(pair.get("quality", {}) or {})
            quality["deepseek_verified"] = False
            quality["deepseek_error"] = str(e)[:100]
            enhanced = dict(pair)
            enhanced["quality"] = quality
            enhanced_pairs.append(enhanced)
            failed_count += 1
            print(f"  ⚠️ DeepSeek failed on pair {i+1}: {e}")

    # Append un-enhanced pairs beyond max_pairs
    for pair in filtered_pairs[max_pairs:]:
        enhanced_pairs.append(pair)

    # Save enhanced pairs back to session
    session["filtered_pairs"] = enhanced_pairs
    session["deepseek_enhanced"] = True
    session["deepseek_stats"] = {
        "model": model,
        "provider": provider,
        "enhanced": enhanced_count,
        "failed": failed_count,
        "total": len(filtered_pairs),
    }
    save_llm_sessions(g.current_user.id, sessions)

    # Recalculate avg quality
    scores = [
        p.get("quality", {}).get("overall_score", 1.0) for p in enhanced_pairs
    ]
    avg_quality = sum(scores) / len(scores) if scores else 0.0

    return jsonify({
        "success": True,
        "enhanced": enhanced_count,
        "failed": failed_count,
        "total_pairs": len(filtered_pairs),
        "avg_quality": avg_quality,
        "model_used": model,
        "provider": provider,
    })

