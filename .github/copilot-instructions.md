## Copilot / AI Agent Instructions for dataClean1

Goal: help an AI coding agent become productive quickly in this repository by summarizing architecture, workflows, conventions, and concrete examples.

**Big Picture**
- **Web UI & API**: The Flask UI + JWT auth lives in [app.py](app.py). It handles uploads, per-user storage (`user_data/<user_id>/`), processing, training, and demo endpoints.
- **ML Microservice**: A separate FastAPI service is in [ml_api.py](ml_api.py) (uvicorn to run). It reuses the same SQLite DB and `data_pipeline` trainer.
- **Processing Library**: Core logic is under `data_pipeline/` — key orchestrator is [data_pipeline/pipeline.py](data_pipeline/pipeline.py). Major modules: `data_loader.py`, `data_cleaner.py`, `feature_engineer.py`, `model_trainer.py`, `report_generator.py`.

**Primary data flow**
- Upload → saved to `user_data/<id>/temp_upload.csv` (see `/upload` and `/process` in [app.py](app.py)).
- `DataPipeline.load()` → `validate()` → `clean()` → `analyze()` → `engineer_features()` → save cleaned/final CSVs and DB `Dataset` row.
- Training uses `ModelTrainer` from [data_pipeline/model_trainer.py](data_pipeline/model_trainer.py) and exports a `.pkl` with keys: `model`, `scaler`, `label_encoder`, `feature_names`, `metrics`.

**Project-specific conventions & patterns**
- Lazy/heavy imports: modules intentionally delay importing heavy libs (matplotlib, sklearn) — follow existing patterns: use helper loaders like `_load_pipeline_libs()` in [app.py](app.py).
- Fluent API and logs: `DataCleaner` and `FeatureEngineer` mutate internal `df` and return `self`; they maintain `cleaning_log`, `transformations`, and `row_changes`. Use these when generating reports or UI JSON payloads.
- Memory-optimized defaults: code is written for constrained hosts (~512MB). Avoid adding heavy optional deps unless guarded by lazy imports.
- Deterministic randomness: many classes set `random_state=42` — preserve this when adding experiments/tests.

**Developer workflows (how to run things)**
- Install & prepare (Render-style): `./build.sh` (installs binaries and initializes DB) — see [build.sh](build.sh).
- Run Flask app (dev): `python app.py` (listens on port 8080 by default). The `__main__` block prints URL.
- Run FastAPI ML service: `uvicorn ml_api:app --reload --port 8000` (uvicorn may need adding to `requirements.txt`).
- Run a local pipeline test: `python run_pipeline.py` (edit top variables) — see [run_pipeline.py](run_pipeline.py).
- Simple API smoke tests: `python test_predict_api.py` or use `pytest` for test files under repo root.

**Important files to inspect for context**
- Entry points: [app.py](app.py), [ml_api.py](ml_api.py)
- Pipeline orchestrator: [data_pipeline/pipeline.py](data_pipeline/pipeline.py)
- Cleaning logic: [data_pipeline/data_cleaner.py](data_pipeline/data_cleaner.py)
- Trainer & model export: [data_pipeline/model_trainer.py](data_pipeline/model_trainer.py)
- Feature engineering: [data_pipeline/feature_engineer.py](data_pipeline/feature_engineer.py)
- Data loading: [data_pipeline/data_loader.py](data_pipeline/data_loader.py)
- Deployment script: [build.sh](build.sh)

**Patterns to follow when editing or extending code**
- Preserve lazy import style for heavy libs and add new heavy deps guarded behind helper loaders.
- Keep DB schema in [app.py](app.py) consistent (SQLAlchemy `User` / `Dataset`) — migrations are not present, so prefer additive, non-breaking changes.
- When changing I/O (CSV/Excel paths), update both Flask and FastAPI code paths — `ml_api.py` reads `final_path` directly from the DB.
- When producing artifacts, follow the existing artifact layout: `user_data/<user_id>/model_*.pkl`, `cleaned_*.csv`, `final_*.csv`, `model_report_<id>.md` / `.html`.

**Examples (copy-paste)**
- Start dev web app:

  python app.py

- Start ML microservice (dev):

  uvicorn ml_api:app --reload --port 8000

- Run the pipeline script (edit FILE_PATH first):

  python run_pipeline.py

- Initialize DB manually (same as build.sh snippet):

  python -c "from app import app, db; with app.app_context(): db.create_all()"

**Testing notes**
- Tests in the repo (e.g., `test_predict_api.py`) are lightweight HTTP clients; start the Flask app on port 8080 before running them.
- Use `pytest -q` to run unit-style tests if available; test files may also be simple scripts to exercise endpoints.

Ask me to refine any section (shorten, add more file-level examples, or include line pointers) and I will iterate.
