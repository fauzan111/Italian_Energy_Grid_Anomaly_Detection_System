# Grid Anomaly Detection

Production-grade starter for anomaly detection on Italian electricity grid data.

## Stack
- Python
- LSTM + Isolation Forest
- MLflow
- Evidently AI
- Apache Airflow
- FastAPI
- Docker
- GitHub Actions
- Vercel static frontend

## Structure
- `src/anomaly_detection/` - application code
- `airflow/dags/` - retraining workflows
- `tests/` - automated tests
- `data/raw/terna/` - downloaded TERNA source data
- `scripts/download_terna_data.py` - repeatable data fetcher
- `frontend/index.html` - Vercel-ready dashboard

## Run locally
1. `python scripts/download_terna_data.py`
2. `python scripts/train_model.py --max-rows 5000 --epochs 1`
3. `uvicorn anomaly_detection.api.main:app --reload`
4. `docker compose up --build`

## Vercel
- Deploy the `frontend/` folder as a static site.
- Point the dashboard at your hosted FastAPI backend URL.
