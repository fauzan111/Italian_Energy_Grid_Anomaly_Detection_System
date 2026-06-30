# Grid Anomaly Detection

Production-grade anomaly detection for Italian electricity grid data with a React + Node.js frontend and FastAPI backend.

## Stack
- React + Vite
- Node.js tooling
- Python / FastAPI
- LSTM + Isolation Forest
- MLflow
- Evidently AI
- Apache Airflow
- Docker
- GitHub Actions
- Vercel-ready dashboard

## Structure
- `frontend/` - React UI
- `src/anomaly_detection/` - backend API and ML code
- `airflow/dags/` - retraining workflows
- `scripts/` - download/train/check commands
- `data/raw/terna/` - downloaded TERNA source data

## Run locally
1. `npm install`
2. `npm run dev`
3. In another terminal: `python scripts\\train_model.py --max-rows 5000 --epochs 1`

## Vercel
- Deploy the repository as a Vercel project.
- The React app builds from `frontend/`.
- Set `VITE_API_BASE_URL` to your hosted FastAPI backend.
