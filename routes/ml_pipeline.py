from flask import (
    Blueprint,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    g,
    send_file,
    send_from_directory,
)
import os
import json
import io
from datetime import datetime

from extensions import db
from models import Dataset
from utils import (
    jwt_required,
    get_current_user,
    get_user_folder,
    generate_plots,
)

# Load pandas for ml routes
import pandas as pd
import numpy as np

ml_bp = Blueprint("ml", __name__)


@ml_bp.route("/")
def index():
    """Home page - landing page for visitors, dashboard for logged-in users."""
    user = get_current_user()
    if user:
        return redirect(url_for("ml.dashboard"))
    return render_template("index.html")


@ml_bp.route("/dashboard")
@jwt_required
def dashboard():
    """User dashboard with their datasets."""
    datasets = (
        Dataset.query.filter_by(user_id=g.current_user.id)
        .order_by(Dataset.created_at.desc())
        .all()
    )
    return render_template("dashboard.html", datasets=datasets, user=g.current_user)


@ml_bp.route("/upload", methods=["POST"])
@jwt_required
def upload_file():
    """Handle file upload."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith((".csv", ".xlsx", ".xls")):
        return jsonify({"error": "Only CSV and Excel files are supported"}), 400

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        for col in df.columns:
            if hasattr(df[col].dtype, "name") and df[col].dtype.name in (
                "string",
                "String",
            ):
                df[col] = df[col].astype("object")
            elif hasattr(df[col].dtype, "numpy_dtype"):
                df[col] = df[col].astype(df[col].dtype.numpy_dtype)

        user_folder = get_user_folder(g.current_user.id)
        temp_path = os.path.join(user_folder, "temp_upload.csv")
        df.to_csv(temp_path, index=False)

        columns = df.columns.tolist()
        dtypes = {col: str(df[col].dtype) for col in columns}
        missing = {col: int(df[col].isnull().sum()) for col in columns}

        sample_df = df.head(5).replace({np.nan: None})
        sample = sample_df.to_dict("records")
        for row in sample:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif hasattr(value, "item"):
                    row[key] = value.item()

        return jsonify(
            {
                "success": True,
                "filename": file.filename,
                "shape": list(df.shape),
                "columns": columns,
                "dtypes": dtypes,
                "missing": missing,
                "sample": sample,
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@ml_bp.route("/process", methods=["POST"])
@jwt_required
def process_data():
    """Process uploaded data through the pipeline."""
    from data_pipeline import DataPipeline, DataCleaner

    try:
        data = request.json
        target_col = data.get("target_column")
        problem_type = data.get("problem_type", "regression")
        dataset_name = data.get("dataset_name", "Untitled Dataset")
        original_filename = data.get("original_filename", "unknown.csv")

        user_folder = get_user_folder(g.current_user.id)
        temp_path = os.path.join(user_folder, "temp_upload.csv")

        if not os.path.exists(temp_path):
            return jsonify(
                {"error": "No file uploaded. Please upload a file first."}
            ), 400

        pipeline = DataPipeline()
        pipeline.load(temp_path)
        validation = pipeline.validate()

        raw_cleaner = DataCleaner(pipeline.raw_df)
        initial_quality = raw_cleaner.validate_quality()
        suggestions = raw_cleaner.generate_suggestions()

        pipeline.clean()
        cleaning_summary = pipeline.cleaner.get_cleaning_summary()
        final_quality = pipeline.cleaner.validate_quality()

        if target_col and target_col in pipeline.cleaned_df.columns:
            pipeline.engineer_features(target_col=target_col, problem_type=problem_type)
        else:
            pipeline.engineer_features()

        feature_summary = pipeline.engineer.get_summary() if pipeline.engineer else {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaned_filename = f"cleaned_{timestamp}.csv"
        final_filename = f"final_{timestamp}.csv"

        cleaned_path = os.path.join(user_folder, cleaned_filename)
        final_path = os.path.join(user_folder, final_filename)

        pipeline.cleaned_df.to_csv(cleaned_path, index=False)
        pipeline.final_df.to_csv(final_path, index=False)

        dataset = Dataset(
            name=dataset_name,
            original_filename=original_filename,
            cleaned_path=cleaned_path,
            final_path=final_path,
            original_rows=validation["shape"][0],
            original_cols=validation["shape"][1],
            cleaned_rows=pipeline.cleaned_df.shape[0],
            cleaned_cols=pipeline.cleaned_df.shape[1],
            final_rows=pipeline.final_df.shape[0],
            final_cols=pipeline.final_df.shape[1],
            target_column=target_col,
            problem_type=problem_type,
            processing_log=json.dumps(
                {
                    "cleaning": cleaning_summary["operations"],
                    "feature_engineering": feature_summary.get("transformations", []),
                    "quality_impact": {
                        "before": initial_quality,
                        "after": final_quality,
                    },
                    "suggestions": suggestions,
                    "row_changes": cleaning_summary.get("row_changes", []),
                }
            ),
            user_id=g.current_user.id,
        )
        db.session.add(dataset)
        db.session.commit()

        raw_filename = f"raw_{dataset.id}.csv"
        raw_path = os.path.join(user_folder, raw_filename)
        pipeline.raw_df.to_csv(raw_path, index=False)

        row_changes_df = pd.DataFrame(cleaning_summary.get("row_changes", []))
        row_changes_filename = f"row_changes_{dataset.id}.csv"
        row_changes_path = os.path.join(user_folder, row_changes_filename)
        if not row_changes_df.empty:
            row_changes_df.to_csv(row_changes_path, index=False)
        else:
            pd.DataFrame(
                columns=[
                    "index",
                    "column",
                    "old_value",
                    "new_value",
                    "operation",
                    "reason",
                ]
            ).to_csv(row_changes_path, index=False)

        plots = generate_plots(pipeline.cleaned_df, target_col)

        try:
            os.remove(temp_path)
        except OSError:
            pass

        return jsonify(
            {
                "success": True,
                "dataset_id": dataset.id,
                "row_changes_csv": row_changes_filename,
                "validation": {
                    "original_shape": validation["shape"],
                    "missing_count": validation["missing_values"][
                        "total_missing_cells"
                    ],
                    "duplicate_count": validation["duplicates"]["count"],
                },
                "cleaning": {
                    "final_shape": list(pipeline.cleaned_df.shape),
                    "operations": cleaning_summary["operations"],
                    "row_changes": cleaning_summary.get("row_changes", []),
                },
                "feature_engineering": {
                    "final_shape": list(pipeline.final_df.shape),
                    "transformations": feature_summary.get("transformations", []),
                },
                "quality_impact": {"before": initial_quality, "after": final_quality},
                "suggestions": suggestions,
                "plots": plots,
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@ml_bp.route("/dataset/<int:dataset_id>")
@jwt_required
def view_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403

    cleaned_sample = []
    cleaned_columns = []
    final_sample = []
    final_columns = []

    try:
        if os.path.exists(dataset.cleaned_path):
            df_clean = pd.read_csv(dataset.cleaned_path)
            cleaned_columns = df_clean.columns.tolist()
            cleaned_sample = (
                df_clean.head(10).replace({np.nan: None}).to_dict("records")
            )

        if os.path.exists(dataset.final_path):
            df_final = pd.read_csv(dataset.final_path)
            final_columns = df_final.columns.tolist()
            final_sample = df_final.head(10).replace({np.nan: None}).to_dict("records")
    except Exception as e:
        print(f"Error loading dataset samples: {e}")

    other_datasets = Dataset.query.filter(
        Dataset.user_id == g.current_user.id, Dataset.id != dataset_id
    ).all()

    return render_template(
        "view_dataset.html",
        dataset=dataset,
        cleaned_columns=cleaned_columns,
        cleaned_sample=cleaned_sample,
        final_columns=final_columns,
        final_sample=final_sample,
        processing_log=json.loads(dataset.processing_log)
        if dataset.processing_log
        else {},
        other_datasets=other_datasets,
    )


@ml_bp.route("/dataset/<int:dataset_id>/download/<file_type>")
@jwt_required
def download_dataset(dataset_id, file_type):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403

    file_format = request.args.get("format", "csv")
    if file_type == "cleaned":
        path = dataset.cleaned_path
        base_filename = f"{dataset.name}_cleaned"
    elif file_type == "final":
        path = dataset.final_path
        base_filename = f"{dataset.name}_model_ready"
    elif file_type == "model":
        path = dataset.model_path
        filename = f"{dataset.name}_model.pkl"
        if not path or not os.path.exists(path):
            return jsonify({"error": "File not found"}), 404
        return send_file(path, as_attachment=True, download_name=filename)
    else:
        return jsonify({"error": "Invalid file type"}), 400

    if not path or not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404

    if file_format == "csv":
        return send_file(path, as_attachment=True, download_name=f"{base_filename}.csv")
    elif file_format == "xlsx":
        try:
            df = pd.read_csv(path)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            return send_file(
                output,
                as_attachment=True,
                download_name=f"{base_filename}.xlsx",
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            return jsonify({"error": f"Error converting to Excel: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid format"}), 400


@ml_bp.route("/dataset/<int:dataset_id>/delete", methods=["POST"])
@jwt_required
def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403

    try:
        user_folder = get_user_folder(g.current_user.id)
        for path in [
            dataset.cleaned_path,
            dataset.final_path,
            dataset.model_path,
            os.path.join(user_folder, f"raw_{dataset.id}.csv"),
            os.path.join(user_folder, f"model_report_{dataset.id}.md"),
            os.path.join(user_folder, f"model_report_{dataset.id}.html"),
        ]:
            if path and os.path.exists(path):
                os.remove(path)

        db.session.delete(dataset)
        db.session.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ml_bp.route("/api/datasets")
@jwt_required
def api_datasets():
    datasets = (
        Dataset.query.filter_by(user_id=g.current_user.id)
        .order_by(Dataset.created_at.desc())
        .all()
    )
    return jsonify(
        [
            {
                "id": d.id,
                "name": d.name,
                "original_filename": d.original_filename,
                "original_rows": d.original_rows,
                "original_cols": d.original_cols,
                "final_rows": d.final_rows,
                "final_cols": d.final_cols,
                "target_column": d.target_column,
                "problem_type": d.problem_type,
                "created_at": d.created_at.isoformat(),
            }
            for d in datasets
        ]
    )


@ml_bp.route("/files")
@jwt_required
def file_manager():
    return render_template("files.html")


def _classify_user_file_type(rel_path):
    rel_lower = rel_path.lower()
    name_lower = os.path.basename(rel_lower)

    if rel_lower.startswith("llm_exports/"):
        if name_lower.endswith(".jsonl"):
            return "LLM Export Data (.jsonl)"
        if name_lower.endswith(".json"):
            return "LLM Export Config (.json)"
        if name_lower.endswith(".py"):
            return "LLM Export Script (.py)"
        if name_lower.endswith(".txt"):
            return "LLM Export Notes (.txt)"
        return "LLM Export File"

    if name_lower.endswith(".pkl"):
        return "Model (.pkl)"
    if name_lower.endswith(".csv"):
        if "cleaned" in name_lower:
            return "Cleaned Data (.csv)"
        if "final" in name_lower:
            return "Model-Ready Data (.csv)"
        return "Raw Data (.csv)"
    if name_lower in {"llm_runs.json", "llm_sessions.json"}:
        return "LLM Metadata (.json)"
    if name_lower.endswith(".json"):
        return "JSON (.json)"
    if name_lower.endswith(".md"):
        return "Report (.md)"
    return "Unknown"


def _resolve_user_relative_path(user_folder, relative_path):
    if not isinstance(relative_path, str):
        return None

    clean = relative_path.strip().replace("\\", "/")
    if not clean or clean.startswith("/"):
        return None

    normalized = os.path.normpath(clean)
    if normalized in {"", "."} or normalized.startswith(".."):
        return None

    root = os.path.abspath(user_folder)
    abs_path = os.path.abspath(os.path.join(root, normalized))
    try:
        common = os.path.commonpath([root, abs_path])
    except ValueError:
        return None
    if common != root:
        return None

    return abs_path, normalized.replace("\\", "/")


def _prune_empty_dirs(start_dir, floor_dir):
    floor_abs = os.path.abspath(floor_dir)
    current = os.path.abspath(start_dir)
    while True:
        try:
            if not current.startswith(floor_abs):
                break
            if not os.path.isdir(current):
                break
            if os.listdir(current):
                break
            os.rmdir(current)
            if current == floor_abs:
                break
            current = os.path.dirname(current)
        except Exception:
            break


def _resolve_legacy_filename(user_folder, raw_name):
    if not isinstance(raw_name, str):
        return None, "invalid"

    name = raw_name.strip()
    if not name or "/" in name or "\\" in name:
        return None, "invalid"

    top_level = os.path.abspath(os.path.join(user_folder, name))
    if os.path.isfile(top_level):
        return (top_level, name), None

    matches = []
    for root, dirs, filenames in os.walk(user_folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in filenames:
            if fname != name:
                continue
            abs_path = os.path.abspath(os.path.join(root, fname))
            rel = os.path.relpath(abs_path, user_folder).replace("\\", "/")
            matches.append((abs_path, rel))
            if len(matches) > 1:
                return None, "ambiguous"

    if not matches:
        return None, "not_found"
    return matches[0], None


@ml_bp.route("/api/user_files")
@jwt_required
def get_user_files():
    user_folder = get_user_folder(g.current_user.id)
    files = []

    def _append_file(file_path, rel_path):
        try:
            stat = os.stat(file_path)
            rel_norm = rel_path.replace(os.sep, "/")
            files.append(
                {
                    "name": os.path.basename(rel_norm),
                    "path": rel_norm,
                    "size": stat.st_size,
                    "date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": _classify_user_file_type(rel_norm),
                }
            )
        except Exception:
            pass

    if os.path.exists(user_folder):
        # Top-level user files (legacy behavior).
        for entry in os.scandir(user_folder):
            if entry.is_file() and not entry.name.startswith("."):
                _append_file(entry.path, entry.name)

        # Nested LLM export files under user_data/<id>/llm_exports/<session>/*
        llm_exports_root = os.path.join(user_folder, "llm_exports")
        if os.path.isdir(llm_exports_root):
            for root, dirs, filenames in os.walk(llm_exports_root):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in filenames:
                    if name.startswith("."):
                        continue
                    path = os.path.join(root, name)
                    rel = os.path.relpath(path, user_folder)
                    _append_file(path, rel)

    files.sort(key=lambda x: x["date"], reverse=True)
    return jsonify(files)


@ml_bp.route("/api/delete_files", methods=["POST"])
@jwt_required
def delete_user_files():
    data = request.get_json(silent=True) or {}
    filenames = data.get("filenames", data.get("paths", []))
    if not isinstance(filenames, list):
        return jsonify({"deleted": [], "errors": ["Invalid payload: filenames must be a list"]}), 400

    user_folder = get_user_folder(g.current_user.id)
    llm_exports_root = os.path.abspath(os.path.join(user_folder, "llm_exports"))
    deleted, errors = [], []
    for raw_path in filenames:
        resolved = _resolve_user_relative_path(user_folder, raw_path)
        path, rel_path = None, ""
        if resolved:
            path, rel_path = resolved
            if not os.path.isfile(path):
                path = None
        if not path:
            # Backward compatibility for cached clients that send basename only.
            legacy_resolved, legacy_err = _resolve_legacy_filename(user_folder, raw_path)
            if legacy_resolved:
                path, rel_path = legacy_resolved
            elif legacy_err == "ambiguous":
                errors.append(
                    f"Ambiguous file name '{raw_path}'. Please refresh and delete by full path."
                )
                continue

        if not path or not rel_path:
            if isinstance(raw_path, str) and raw_path.strip():
                errors.append(f"File not found or invalid path: {raw_path}")
            continue

        try:
            os.remove(path)
            deleted.append(rel_path)
            abs_path = os.path.abspath(path)
            if abs_path.startswith(llm_exports_root + os.sep):
                _prune_empty_dirs(os.path.dirname(abs_path), llm_exports_root)
        except Exception as e:
            errors.append(f"Error deleting {rel_path}: {str(e)}")
    return jsonify({"deleted": deleted, "errors": errors})


@ml_bp.route("/api/download_user_file")
@jwt_required
def download_user_file():
    user_folder = get_user_folder(g.current_user.id)
    rel_path = request.args.get("path", "")
    resolved = _resolve_user_relative_path(user_folder, rel_path)
    if not resolved:
        return jsonify({"error": "Invalid file path"}), 400

    abs_path, safe_rel = resolved
    if not os.path.isfile(abs_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(abs_path, as_attachment=True, download_name=os.path.basename(safe_rel))


@ml_bp.route("/download/<filename>")
@jwt_required
def download_file(filename):
    user_folder = get_user_folder(g.current_user.id)
    # SECURITY: Reject any path traversal attempts
    if ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "Invalid filename"}), 400
    safe_path = os.path.abspath(os.path.join(user_folder, filename))
    if not safe_path.startswith(os.path.abspath(user_folder)):
        return jsonify({"error": "Access denied"}), 403
    return send_from_directory(user_folder, filename, as_attachment=True)


@ml_bp.route("/download_changes/<filename>")
@jwt_required
def download_changes(filename):
    return download_file(filename)


@ml_bp.route("/demo")
def demo_page():
    user = get_current_user()
    user_models, untrained_datasets = [], []
    if user:
        datasets = Dataset.query.filter_by(user_id=user.id).all()
        for d in datasets:
            info = {
                "id": d.id,
                "name": d.name,
                "target": d.target_column,
                "type": d.problem_type,
                "created_at": d.created_at.strftime("%Y-%m-%d"),
            }
            if d.model_path and os.path.exists(d.model_path):
                user_models.append(info)
            else:
                untrained_datasets.append(info)
    return render_template(
        "demo.html",
        user=user,
        user_models=user_models,
        untrained_datasets=untrained_datasets,
    )


# Some routes left out for brevity (model training), but we can add those to another blueprint or keep here.
