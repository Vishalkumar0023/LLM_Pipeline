import os
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, g, send_file
import pandas as pd
import numpy as np

from extensions import db
from models import Dataset
from utils import jwt_required, get_user_folder, _load_data_libs, _load_pipeline_libs

ml_models_bp = Blueprint("ml_models", __name__)


@ml_models_bp.route("/dataset/<int:dataset_id>/transform", methods=["POST"])
@jwt_required
def transform_dataset(dataset_id):
    _load_pipeline_libs()
    from data_pipeline import DataPipeline, DataCleaner

    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403

    data = request.json
    operation = data.get("operation")
    params = data.get("params", {})
    pipeline = DataPipeline()
    if os.path.exists(dataset.cleaned_path):
        pipeline.load(dataset.cleaned_path)
    else:
        return jsonify({"error": "Dataset file not found"}), 404

    cleaner = DataCleaner(pipeline.raw_df)

    try:
        if operation == "clean_numeric_text":
            cleaner.clean_numeric_text(**params)
        elif operation == "rename_columns":
            cleaner.rename_columns(**params)
        elif operation == "extract_regex":
            cleaner.extract_regex_feature(**params)
        elif operation == "remove_duplicates":
            cleaner.remove_duplicates(**params)
        elif operation == "drop_columns":
            cleaner.drop_columns(**params)
        else:
            return jsonify({"error": "Invalid operation"}), 400

        new_df = cleaner.get_cleaned_data()
        new_df.to_csv(dataset.cleaned_path, index=False)
        if os.path.exists(dataset.final_path):
            new_df.to_csv(dataset.final_path, index=False)
            dataset.final_rows = new_df.shape[0]
            dataset.final_cols = new_df.shape[1]

        dataset.cleaned_rows = new_df.shape[0]
        dataset.cleaned_cols = new_df.shape[1]

        try:
            log = json.loads(dataset.processing_log) if dataset.processing_log else {}
        except:
            log = {}

        if "cleaning" not in log:
            log["cleaning"] = []
        if cleaner.cleaning_log:
            log["cleaning"].extend(cleaner.cleaning_log[-1:])

        dataset.processing_log = json.dumps(log)
        db.session.commit()

        return jsonify(
            {
                "success": True,
                "preview": new_df.head()
                .replace({np.nan: None})
                .to_dict(orient="records"),
                "columns": new_df.columns.tolist(),
                "stats": {"rows": new_df.shape[0], "cols": new_df.shape[1]},
                "message": cleaner.cleaning_log[-1]
                if cleaner.cleaning_log
                else "Transformation applied",
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@ml_models_bp.route("/dataset/<int:dataset_id>/report")
@jwt_required
def view_model_report(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403
    user_folder = get_user_folder(g.current_user.id)
    report_path = os.path.join(user_folder, f"model_report_{dataset.id}.html")
    if not os.path.exists(report_path):
        return jsonify(
            {"error": "Report not found. Please train a new model to generate it."}
        ), 404
    return send_file(report_path)


@ml_models_bp.route("/dataset/<int:dataset_id>/download/report")
@jwt_required
def download_model_report(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403
    user_folder = get_user_folder(g.current_user.id)
    html_filename = f"model_report_{dataset.id}.html"
    html_path = os.path.join(user_folder, html_filename)
    if os.path.exists(html_path):
        return send_file(html_path, as_attachment=True, download_name=html_filename)
    report_filename = f"model_report_{dataset.id}.md"
    report_path = os.path.join(user_folder, report_filename)
    if not os.path.exists(report_path):
        return jsonify(
            {"error": "Report not found. Please train a new model to generate it."}
        ), 404
    return send_file(report_path, as_attachment=True, download_name=report_filename)


@ml_models_bp.route("/dataset/<int:dataset_id>/train", methods=["POST"])
@jwt_required
def train_model(dataset_id):
    _load_pipeline_libs()
    from data_pipeline import ModelTrainer, DataPipeline

    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403

    try:
        data_path = dataset.final_path or dataset.cleaned_path
        if not data_path or not os.path.exists(data_path):
            return jsonify({"error": "Dataset file not found"}), 404

        df = pd.read_csv(data_path)
        data = request.json or {}
        target_col = data.get("target_column", dataset.target_column)
        problem_type = data.get("problem_type", dataset.problem_type)

        user_folder = get_user_folder(g.current_user.id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(user_folder, f"model_{dataset_id}_{timestamp}.pkl")
        raw_path = os.path.join(user_folder, f"raw_{dataset_id}.csv")
        raw_df = pd.read_csv(raw_path) if os.path.exists(raw_path) else None

        trainer = ModelTrainer(
            df, target_col=target_col, problem_type=problem_type, raw_df=raw_df
        )
        results = trainer.run_full_comparison()
        trainer.export_model(model_path)

        dataset.model_path = model_path
        dataset.model_results = json.dumps(results, default=str)
        if trainer.target_col:
            dataset.target_column = trainer.target_col
        if trainer.problem_type:
            dataset.problem_type = trainer.problem_type
        db.session.commit()

        try:
            processed_log = (
                json.loads(dataset.processing_log) if dataset.processing_log else {}
            )
            dummy = DataPipeline()
            dummy.target_col = trainer.target_col
            dummy.problem_type = trainer.problem_type
            dummy.raw_df = raw_df
            dummy.cleaned_df = df
            dummy.model_results = results

            steps = []
            if "cleaning" in processed_log:
                steps.extend([f"Cleaning: {x}" for x in processed_log["cleaning"]])
            if "feature_engineering" in processed_log:
                steps.extend(
                    [f"Feature Eng: {x}" for x in processed_log["feature_engineering"]]
                )
            dummy.pipeline_report = {"preprocessing": {"steps_executed": steps}}

            dummy.generate_markdown_report(
                os.path.join(user_folder, f"model_report_{dataset_id}.md")
            )
            dummy.generate_html_report(
                os.path.join(user_folder, f"model_report_{dataset_id}.html")
            )
        except:
            pass

        return jsonify({"success": True, "results": results})
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@ml_models_bp.route("/api/user_model_info/<int:dataset_id>")
@jwt_required
def get_user_model_info(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403
    if not dataset.model_path or not os.path.exists(dataset.model_path):
        return jsonify({"error": "Model not found"}), 404

    try:
        import joblib

        model_data = joblib.load(dataset.model_path)
        return jsonify(
            {
                "success": True,
                "name": dataset.name,
                "features": model_data.get("feature_names", []),
                "target": dataset.target_column,
                "problem_type": dataset.problem_type,
                "metrics": model_data.get("metrics", {}),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ml_models_bp.route("/api/predict_user_model", methods=["POST"])
@jwt_required
def predict_user_model():
    _load_data_libs()
    try:
        data = request.json
        dataset_id = data.get("dataset_id")
        features = data.get("features")
        if not dataset_id or not features:
            return jsonify({"error": "Missing dataset_id or features"}), 400

        dataset = Dataset.query.get_or_404(dataset_id)
        if dataset.user_id != g.current_user.id:
            return jsonify({"error": "Access denied"}), 403
        if not dataset.model_path or not os.path.exists(dataset.model_path):
            return jsonify({"error": "Model not found"}), 404

        import joblib

        model_data = joblib.load(dataset.model_path)
        model = model_data.get("model")
        scaler = model_data.get("scaler")
        feature_names = model_data.get("feature_names", [])

        input_df = pd.DataFrame([features])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        if scaler:
            input_df = scaler.transform(input_df)
        prediction = model.predict(input_df)[0]
        return jsonify(
            {
                "success": True,
                "prediction": float(prediction),
                "target": dataset.target_column,
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


DEMO_MODELS_DIR = "demo_models"


def load_demo_model(model_type):
    try:
        if model_type not in ["sales", "student"]:
            return None
        path = os.path.join(DEMO_MODELS_DIR, f"{model_type}_model.pkl")
        if not os.path.exists(path):
            return None
        import joblib

        return joblib.load(path)
    except:
        return None


@ml_models_bp.route("/api/predict", methods=["POST"])
def predict_demo():
    _load_data_libs()
    try:
        data = request.json
        model_type = data.get("model_type")
        features = data.get("features")
        if not model_type or not features:
            return jsonify({"error": "Missing model_type or features"}), 400
        model = load_demo_model(model_type)
        if not model:
            return jsonify({"error": "Model not found or could not be loaded"}), 404

        if model_type == "sales":
            feature_names = ["TV_Ad_Budget", "Radio_Ad_Budget", "Newspaper_Ad_Budget"]
        elif model_type == "student":
            feature_names = ["Study_Hours", "Attendance_Percentage", "Previous_Score"]
        else:
            return jsonify({"error": "Unknown model type"}), 400

        input_df = pd.DataFrame([features])[feature_names]
        prediction = model.predict(input_df)[0]
        return jsonify(
            {"success": True, "model_type": model_type, "prediction": float(prediction)}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ml_models_bp.route("/dataset/<int:dataset_id>/drift", methods=["POST"])
@jwt_required
def check_drift(dataset_id):
    current_ds = Dataset.query.get_or_404(dataset_id)
    if current_ds.user_id != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403

    data = request.get_json()
    baseline_id = data.get("baseline_id")
    if not baseline_id:
        return jsonify({"error": "Baseline dataset ID required"}), 400

    baseline_ds = Dataset.query.get(baseline_id)
    if not baseline_ds or baseline_ds.user_id != g.current_user.id:
        return jsonify({"error": "Invalid baseline dataset"}), 400

    try:
        cur_df = pd.read_csv(current_ds.cleaned_path)
        base_df = pd.read_csv(baseline_ds.cleaned_path)
        from data_pipeline.drift_detector import DriftDetector

        detector = DriftDetector(base_df, cur_df)
        detector.run()
        return jsonify({"success": True, "report": detector.report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ml_models_bp.route("/dataset/<int:dataset_id>/synthesize", methods=["POST"])
@jwt_required
def generate_synthetic(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403

    try:
        df = pd.read_csv(dataset.cleaned_path)
        from data_pipeline.synthetic_generator import SyntheticGenerator

        gen = SyntheticGenerator(df)
        gen.fit()
        data = request.get_json() or {}
        n_rows = int(data.get("n_rows", len(df)))
        synthetic_df = gen.generate(n_rows=n_rows)

        filename = (
            f"synthetic_{dataset.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        user_folder = get_user_folder(g.current_user.id)
        path = os.path.join(user_folder, filename)
        synthetic_df.to_csv(path, index=False)

        return jsonify(
            {
                "success": True,
                "filename": filename,
                "preview": synthetic_df.head(5)
                .replace({np.nan: None})
                .to_dict("records"),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ml_models_bp.route("/dataset/<int:dataset_id>/evolve", methods=["POST"])
@jwt_required
def evolve_features(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != g.current_user.id:
        return jsonify({"error": "Access denied"}), 403

    try:
        df = pd.read_csv(dataset.cleaned_path)
        from data_pipeline.feature_engineer import FeatureEngineer

        engineer = FeatureEngineer(
            df, target_col=dataset.target_column, problem_type=dataset.problem_type
        )
        engineer.create_datetime_features()
        engineer.encode_categorical()
        engineer.auto_evolve(max_new_features=5)
        engineer.scale_features()

        final_df = engineer.get_transformed_data()
        final_df.to_csv(dataset.final_path, index=False)
        dataset.final_rows = final_df.shape[0]
        dataset.final_cols = final_df.shape[1]

        try:
            log = json.loads(dataset.processing_log)
        except:
            log = {}
        log["feature_engineering"] = engineer.get_summary()["transformations"]
        dataset.processing_log = json.dumps(log)
        db.session.commit()

        return jsonify(
            {
                "success": True,
                "features_count": final_df.shape[1],
                "transformations": log["feature_engineering"],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
