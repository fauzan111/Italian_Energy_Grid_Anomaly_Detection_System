from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anomaly_detection.config import settings
from anomaly_detection.data.ingest import build_feature_frame, train_validation_split
from anomaly_detection.monitoring.drift import generate_drift_report


if __name__ == "__main__":
    frame = build_feature_frame(settings.raw_data_dir)
    train_frame, validation_frame = train_validation_split(frame)
    print(generate_drift_report(train_frame, validation_frame, settings.drift_report_dir))
