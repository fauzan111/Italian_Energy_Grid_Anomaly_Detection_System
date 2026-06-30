from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from anomaly_detection.config import settings
from anomaly_detection.models.predict import load_latest_model, predict_records
from anomaly_detection.models.train import HybridAnomalyModel
from anomaly_detection.pipelines.retrain import retrain


class PredictionRecord(BaseModel):
    timestamp: str
    total_load_mw: float | None = None
    forecast_total_load_mw: float | None = None
    actual_generation_mw: float | None = None
    load_gap_mw: float | None = None
    load_gap_pct: float | None = None
    hour: float | None = None
    day_of_week: float | None = None
    day_of_year: float | None = None
    month: float | None = None
    is_weekend: float | None = None
    hour_sin: float | None = None
    hour_cos: float | None = None


class PredictionRequest(BaseModel):
    records: list[PredictionRecord] = Field(min_length=1)


class RetrainResponse(BaseModel):
    model_dir: str
    rows: int
    train_rows: int
    validation_rows: int
    drift: dict[str, object]


@asynccontextmanager
async def lifespan(application: FastAPI):
    model_path = Path(settings.model_dir)
    if model_path.exists():
        application.state.model = load_latest_model(model_path)
    else:
        application.state.model = None
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> dict[str, object]:
    model: HybridAnomalyModel | None = app.state.model
    if model is None:
        raise HTTPException(status_code=404, detail="No trained model is loaded")
    return {
        "seq_length": model.seq_length,
        "threshold": model.threshold,
        "feature_count": len(model.feature_columns),
        "trained_at": model.metadata.get("trained_at"),
        "features": model.feature_columns,
    }


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, object]:
    model: HybridAnomalyModel | None = app.state.model
    if model is None:
        raise HTTPException(status_code=404, detail="No trained model is loaded")
    frame = pd.DataFrame([record.model_dump() for record in payload.records])
    predictions = predict_records(model, frame.to_dict(orient="records"))
    return {"predictions": predictions.reset_index().rename(columns={"index": "timestamp"}).to_dict(orient="records")}


@app.post("/retrain", response_model=RetrainResponse)
def run_retrain() -> dict[str, object]:
    result = retrain()
    app.state.model = load_latest_model(settings.model_dir)
    return result
