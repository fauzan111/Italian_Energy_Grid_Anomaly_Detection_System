from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from anomaly_detection.config import settings
from anomaly_detection.data.ingest import build_feature_frame
from anomaly_detection.models.train import HybridAnomalyModel, load_model, predict_scores


def load_latest_model(model_dir: str | Path = settings.model_dir) -> HybridAnomalyModel:
    return load_model(model_dir)


def predict_from_frame(model: HybridAnomalyModel, feature_frame: pd.DataFrame) -> pd.DataFrame:
    return predict_scores(model, feature_frame)


def predict_from_raw(model: HybridAnomalyModel, raw_dir: str | Path = settings.raw_data_dir) -> pd.DataFrame:
    feature_frame = build_feature_frame(raw_dir)
    return predict_scores(model, feature_frame)


def predict_records(model: HybridAnomalyModel, records: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(list(records))
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False)
        frame = frame.set_index("timestamp")
    frame = frame.sort_index()
    return predict_scores(model, frame)
