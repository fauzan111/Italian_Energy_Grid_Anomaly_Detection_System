from anomaly_detection.api.main import app


def test_health_endpoint() -> None:
    assert app.title == "Grid Anomaly Detection API"
