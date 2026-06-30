from anomaly_detection.config import settings
from anomaly_detection.data.ingest import build_feature_frame


def test_build_feature_frame_has_numeric_features() -> None:
    frame = build_feature_frame(settings.raw_data_dir)
    assert not frame.empty
    assert "total_load_mw" in frame.columns
    assert "actual_generation_mw" in frame.columns
