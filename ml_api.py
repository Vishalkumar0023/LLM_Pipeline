import os
import json
import logging
import time
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Path
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Import our existing ML engine
from data_pipeline.model_trainer import ModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Data Pipeline ML Service",
    description="High-performance, asynchronous ML training API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Connection (Reuse existing SQLite DB)
DB_PATH = os.path.abspath("instance/pipeline_users.db")
# Convert to SQLAlchemy URL
db_url = f"sqlite:///{DB_PATH}"
engine = create_engine(db_url)

# ─── Pydantic Models ───
class TrainRequest(BaseModel):
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "target_column": "target",
                "problem_type": "classification"
            }
        }

class ModelInfo(BaseModel):
    name: str
    metrics: Dict[str, Any]
    params: Dict[str, Any]

class TrainResponse(BaseModel):
    success: bool
    dataset_id: int
    best_model: ModelInfo
    reliability_score: float
    warnings: List[str]
    model_path: str

# ─── Helper Functions ───
def get_dataset_path(dataset_id: int) -> str:
    """Fetch 'final_path' from SQLite for a given dataset ID."""
    query = text("SELECT final_path FROM dataset WHERE id = :id")
    with engine.connect() as conn:
        result = conn.execute(query, {"id": dataset_id}).fetchone()
        if not result or not result[0]:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found or has no final file")
        return result[0]

def save_model_path(dataset_id: int, model_path: str, model_results: dict):
    """Update database with model path and results."""
    query = text("""
        UPDATE dataset 
        SET model_path = :path, model_results = :results 
        WHERE id = :id
    """)
    with engine.connect() as conn:
        conn.execute(query, {
            "path": model_path,
            "results": json.dumps(model_results),
            "id": dataset_id
        })
        conn.commit()

# ─── Endpoints ───

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ML Microservice"}

@app.post("/train/{dataset_id}", response_model=TrainResponse)
def train_model(
    dataset_id: int = Path(..., title="The ID of the dataset to train"),
    config: TrainRequest = None
):
    """
    Train an ML model on the specified dataset.
    This runs in a thread pool (blocking safe) and returns the results.
    """
    logger.info(f"Received training request for dataset {dataset_id}")
    start_time = time.time()
    
    # 1. Get file path
    file_path = get_dataset_path(dataset_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")
    
    # 2. Load Data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")
    
    # 3. Train Model
    try:
        trainer = ModelTrainer(
            df=df,
            target_col=config.target_column if config else None,
            problem_type=config.problem_type if config else None
        )
        results = trainer.run()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail="Internal ML training error")
        raise HTTPException(status_code=500, detail="Internal ML training error")
    
    duration = time.time() - start_time
    logger.info(f"Training completed in {duration:.2f} seconds")
    # 4. Save Model Artifact
    model_filename = f"model_dataset_{dataset_id}.pkl"
    user_folder = os.path.dirname(file_path)
    model_path = os.path.join(user_folder, model_filename)
    
    trainer.export_model(model_path)
    
    # 5. Update DB
    save_model_path(dataset_id, model_path, results)
    
    # 6. Format Response
    best = results.get('best_model', {})
    
    return {
        "success": True,
        "dataset_id": dataset_id,
        "best_model": {
            "name": best.get('name', 'Unknown'),
            "metrics": best.get('metrics', {}),
            "params": best.get('params', {})
        },
        "reliability_score": results.get('reliability', {}).get('score', 0),
        "warnings": results.get('warnings', []),
        "model_path": model_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
