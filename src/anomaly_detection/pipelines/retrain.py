from __future__ import annotations

from pathlib import Path

from anomaly_detection.config import settings
from anomaly_detection.data.ingest import build_feature_frame, train_validation_split
from anomaly_detection.models.train import HybridAnomalyModel, save_model, train_hybrid_model
from anomaly_detection.monitoring.drift import generate_drift_report


def retrain(raw_dir: str | Path = settings.raw_data_dir, model_dir: str | Path = settings.model_dir) -> dict[str, object]:
    feature_frame = build_feature_frame(raw_dir)
    train_frame, validation_frame = train_validation_split(feature_frame)
    model: HybridAnomalyModel = train_hybrid_model(feature_frame)
    save_path = save_model(model, model_dir)
    drift = generate_drift_report(train_frame, validation_frame, settings.drift_report_dir)
    return {
        "model_dir": str(save_path),
        "rows": int(len(feature_frame)),
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "drift": drift,
    }
