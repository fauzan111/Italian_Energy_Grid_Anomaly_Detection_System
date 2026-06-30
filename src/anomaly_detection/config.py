from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    app_name: str = "Grid Anomaly Detection API"
    raw_data_dir: Path = project_root / "data" / "raw" / "terna"
    model_dir: Path = project_root / "models" / "latest"
    drift_report_dir: Path = project_root / "reports" / "drift"
    seq_length: int = 48
    lstm_units: int = 64
    latent_units: int = 16
    epochs: int = 5
    batch_size: int = 128
    contamination: float = 0.05
    random_state: int = 42


settings = Settings()
