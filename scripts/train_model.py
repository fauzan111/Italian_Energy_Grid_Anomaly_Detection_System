from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anomaly_detection.config import settings
from anomaly_detection.data.ingest import build_feature_frame
from anomaly_detection.models.train import log_training_run, save_model, train_hybrid_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the anomaly detection model")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seq-length", type=int, default=settings.seq_length)
    parser.add_argument("--epochs", type=int, default=settings.epochs)
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    parser.add_argument("--lstm-units", type=int, default=settings.lstm_units)
    parser.add_argument("--latent-units", type=int, default=settings.latent_units)
    args = parser.parse_args()

    frame = build_feature_frame(settings.raw_data_dir)
    if args.max_rows > 0:
        frame = frame.iloc[: args.max_rows]

    model = train_hybrid_model(
        frame,
        seq_length=args.seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lstm_units=args.lstm_units,
        latent_units=args.latent_units,
    )
    path = save_model(model, settings.model_dir)
    log_training_run(model, path)
    print({"model_dir": str(path), "rows": len(frame)})


if __name__ == "__main__":
    main()
