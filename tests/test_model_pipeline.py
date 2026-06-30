from pathlib import Path

from anomaly_detection.config import settings
from anomaly_detection.data.ingest import build_feature_frame
from anomaly_detection.models.train import predict_scores, save_model, train_hybrid_model


def test_train_and_predict_small_slice(tmp_path: Path) -> None:
    frame = build_feature_frame(settings.raw_data_dir).iloc[:80]
    model = train_hybrid_model(
        frame,
        seq_length=8,
        epochs=1,
        batch_size=8,
        lstm_units=8,
        latent_units=4,
    )
    save_model(model, tmp_path)
    predictions = predict_scores(model, frame)
    assert not predictions.empty
    assert {"anomaly_score", "is_anomaly", "reconstruction_error", "isolation_score"}.issubset(predictions.columns)
