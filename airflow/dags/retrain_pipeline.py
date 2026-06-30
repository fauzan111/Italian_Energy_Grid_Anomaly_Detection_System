from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # pragma: no cover - optional dependency
    DAG = None
    PythonOperator = None

from anomaly_detection.pipelines.retrain import retrain


def _run_retrain() -> dict[str, object]:
    return retrain()


if DAG is not None and PythonOperator is not None:
    with DAG(
        dag_id="grid_anomaly_retrain",
        start_date=datetime(2025, 1, 1),
        schedule=timedelta(days=1),
        catchup=False,
        tags=["anomaly-detection", "terna", "mlops"],
        default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    ) as dag:
        retrain_task = PythonOperator(
            task_id="retrain_model",
            python_callable=_run_retrain,
        )

    retrain_task
else:
    dag = None
