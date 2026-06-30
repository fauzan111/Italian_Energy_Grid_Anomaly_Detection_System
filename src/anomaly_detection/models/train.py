from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from anomaly_detection.config import settings
from anomaly_detection.data.ingest import build_feature_frame, ensure_raw_data, train_validation_split

try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None


@dataclass
class HybridAnomalyModel:
    feature_columns: list[str]
    feature_scaler: StandardScaler
    hybrid_scaler: StandardScaler
    autoencoder: keras.Model
    encoder: keras.Model
    isolation_forest: IsolationForest
    seq_length: int
    rec_error_mean: float
    rec_error_std: float
    threshold: float
    metadata: dict[str, object]


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def build_autoencoder(
    seq_length: int,
    n_features: int,
    lstm_units: int | None = None,
    latent_units: int | None = None,
) -> tuple[keras.Model, keras.Model]:
    lstm_units = lstm_units or settings.lstm_units
    latent_units = latent_units or settings.latent_units
    inputs = keras.Input(shape=(seq_length, n_features), name="sequence")
    x = keras.layers.LSTM(lstm_units, return_sequences=True, name="encoder_lstm_1")(inputs)
    x = keras.layers.LSTM(latent_units, return_sequences=False, name="encoder_lstm_2")(x)
    latent = keras.layers.Dense(latent_units, activation="relu", name="latent")(x)
    x = keras.layers.RepeatVector(seq_length, name="repeat")(latent)
    x = keras.layers.LSTM(lstm_units, return_sequences=True, name="decoder_lstm_1")(x)
    outputs = keras.layers.TimeDistributed(
        keras.layers.Dense(n_features),
        name="reconstruction",
    )(x)

    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="lstm_autoencoder")
    encoder = keras.Model(inputs=inputs, outputs=latent, name="lstm_encoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def create_sequences(values: np.ndarray, seq_length: int) -> np.ndarray:
    if len(values) < seq_length + 1:
        raise ValueError(f"Need at least {seq_length + 1} rows to build sequences")
    return np.stack([values[i : i + seq_length] for i in range(len(values) - seq_length + 1)])


def _sequence_feature_names(feature_columns: list[str]) -> list[str]:
    return feature_columns


def _hybrid_features(
    encoder: keras.Model,
    autoencoder: keras.Model,
    sequences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    latent = encoder.predict(sequences, verbose=0)
    reconstructed = autoencoder.predict(sequences, verbose=0)
    rec_error = np.mean(np.abs(sequences - reconstructed), axis=(1, 2))
    hybrid = np.concatenate([latent, rec_error.reshape(-1, 1)], axis=1)
    return hybrid, rec_error


def train_hybrid_model(
    feature_frame: pd.DataFrame,
    seq_length: int | None = None,
    contamination: float | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lstm_units: int | None = None,
    latent_units: int | None = None,
) -> HybridAnomalyModel:
    set_random_seed(settings.random_state)
    seq_length = seq_length or settings.seq_length
    contamination = contamination or settings.contamination
    epochs = epochs or settings.epochs
    batch_size = batch_size or settings.batch_size

    feature_columns = list(feature_frame.columns)
    train_frame, _ = train_validation_split(feature_frame)

    feature_scaler = StandardScaler()
    train_scaled = feature_scaler.fit_transform(train_frame.values)
    train_sequences = create_sequences(train_scaled, seq_length)

    autoencoder, encoder = build_autoencoder(
        seq_length,
        train_scaled.shape[1],
        lstm_units=lstm_units,
        latent_units=latent_units,
    )
    autoencoder.fit(
        train_sequences,
        train_sequences,
        epochs=epochs,
        batch_size=min(batch_size, len(train_sequences)),
        verbose=0,
        shuffle=False,
    )

    hybrid_train, rec_error = _hybrid_features(encoder, autoencoder, train_sequences)
    hybrid_scaler = StandardScaler()
    hybrid_train_scaled = hybrid_scaler.fit_transform(hybrid_train)

    iforest = IsolationForest(
        contamination=contamination,
        random_state=settings.random_state,
        n_estimators=200,
    )
    iforest.fit(hybrid_train_scaled)

    isolation_scores = -iforest.score_samples(hybrid_train_scaled)
    combined_scores = 0.55 * isolation_scores + 0.45 * _zscore(rec_error)
    threshold = float(np.quantile(combined_scores, 1 - contamination))

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(feature_columns),
        "sequence_count": int(len(train_sequences)),
        "seq_length": seq_length,
        "contamination": contamination,
        "columns": feature_columns,
    }
    return HybridAnomalyModel(
        feature_columns=feature_columns,
        feature_scaler=feature_scaler,
        hybrid_scaler=hybrid_scaler,
        autoencoder=autoencoder,
        encoder=encoder,
        isolation_forest=iforest,
        seq_length=seq_length,
        rec_error_mean=float(rec_error.mean()),
        rec_error_std=float(rec_error.std() or 1.0),
        threshold=threshold,
        metadata=metadata,
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std() or 1.0)
    return (values - values.mean()) / std


def save_model(model: HybridAnomalyModel, model_dir: str | Path) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model.autoencoder.save(model_dir / "autoencoder.keras", overwrite=True)
    model.encoder.save(model_dir / "encoder.keras", overwrite=True)

    with (model_dir / "feature_scaler.pkl").open("wb") as f:
        pickle.dump(model.feature_scaler, f)
    with (model_dir / "hybrid_scaler.pkl").open("wb") as f:
        pickle.dump(model.hybrid_scaler, f)
    with (model_dir / "isolation_forest.pkl").open("wb") as f:
        pickle.dump(model.isolation_forest, f)

    payload = {
        **model.metadata,
        "feature_columns": model.feature_columns,
        "seq_length": model.seq_length,
        "rec_error_mean": model.rec_error_mean,
        "rec_error_std": model.rec_error_std,
        "threshold": model.threshold,
    }
    (model_dir / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return model_dir


def log_training_run(model: HybridAnomalyModel, model_dir: str | Path) -> None:
    if mlflow is None:
        return
    model_dir = Path(model_dir)
    mlflow.set_experiment("grid-anomaly-detection")
    with mlflow.start_run(run_name=model.metadata.get("trained_at", "grid-anomaly-run")):
        mlflow.log_params(
            {
                "seq_length": model.seq_length,
                "feature_count": len(model.feature_columns),
                "contamination": model.metadata.get("contamination"),
                "sequence_count": model.metadata.get("sequence_count"),
            }
        )
        mlflow.log_metrics(
            {
                "threshold": model.threshold,
                "reconstruction_error_mean": model.rec_error_mean,
                "reconstruction_error_std": model.rec_error_std,
            }
        )
        mlflow.log_artifacts(str(model_dir))


def load_model(model_dir: str | Path) -> HybridAnomalyModel:
    model_dir = Path(model_dir)
    with (model_dir / "feature_scaler.pkl").open("rb") as f:
        feature_scaler = pickle.load(f)
    with (model_dir / "hybrid_scaler.pkl").open("rb") as f:
        hybrid_scaler = pickle.load(f)
    with (model_dir / "isolation_forest.pkl").open("rb") as f:
        iforest = pickle.load(f)

    autoencoder = keras.models.load_model(model_dir / "autoencoder.keras")
    encoder = keras.models.load_model(model_dir / "encoder.keras")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return HybridAnomalyModel(
        feature_columns=list(metadata["feature_columns"]),
        feature_scaler=feature_scaler,
        hybrid_scaler=hybrid_scaler,
        autoencoder=autoencoder,
        encoder=encoder,
        isolation_forest=iforest,
        seq_length=int(metadata["seq_length"]),
        rec_error_mean=float(metadata["rec_error_mean"]),
        rec_error_std=float(metadata["rec_error_std"]),
        threshold=float(metadata["threshold"]),
        metadata=metadata,
    )


def prepare_training_frame(raw_dir: str | Path = settings.raw_data_dir) -> pd.DataFrame:
    ensure_raw_data(raw_dir)
    return build_feature_frame(raw_dir)


def train_and_save(raw_dir: str | Path = settings.raw_data_dir, model_dir: str | Path = settings.model_dir) -> Path:
    frame = prepare_training_frame(raw_dir)
    model = train_hybrid_model(frame)
    save_path = save_model(model, model_dir)
    log_training_run(model, save_path)
    return save_path


def predict_scores(model: HybridAnomalyModel, feature_frame: pd.DataFrame) -> pd.DataFrame:
    aligned = feature_frame.reindex(columns=model.feature_columns).fillna(0).ffill().bfill()
    if len(aligned) < model.seq_length + 1:
        raise ValueError(f"Need at least {model.seq_length + 1} rows for prediction")
    scaled = model.feature_scaler.transform(aligned.values)
    sequences = create_sequences(scaled, model.seq_length)
    hybrid, rec_error = _hybrid_features(model.encoder, model.autoencoder, sequences)
    hybrid_scaled = model.hybrid_scaler.transform(hybrid)
    iforest_scores = -model.isolation_forest.score_samples(hybrid_scaled)
    rec_z = (rec_error - model.rec_error_mean) / (model.rec_error_std or 1.0)
    anomaly_score = 0.55 * iforest_scores + 0.45 * rec_z
    timestamps = aligned.index[model.seq_length - 1 :]
    return pd.DataFrame(
        {
            "anomaly_score": anomaly_score,
            "is_anomaly": anomaly_score >= model.threshold,
            "reconstruction_error": rec_error,
            "isolation_score": iforest_scores,
        },
        index=timestamps,
    )
