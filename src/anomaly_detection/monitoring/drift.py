from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from evidently.report import Report
    from evidently.metrics import DataDriftPreset
except ImportError:  # pragma: no cover - optional dependency
    Report = None
    DataDriftPreset = None


def _psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected = expected.astype(float)
    actual = actual.astype(float)
    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(expected.quantile(quantiles).to_numpy())
    if len(edges) < 3:
        return 0.0
    expected_bins = pd.cut(expected, bins=edges, include_lowest=True)
    actual_bins = pd.cut(actual, bins=edges, include_lowest=True)
    expected_dist = expected_bins.value_counts(normalize=True).sort_index()
    actual_dist = actual_bins.value_counts(normalize=True).sort_index()
    aligned = pd.concat([expected_dist, actual_dist], axis=1).fillna(0.0001)
    aligned.columns = ["expected", "actual"]
    aligned["expected"] = aligned["expected"].replace(0, 0.0001)
    aligned["actual"] = aligned["actual"].replace(0, 0.0001)
    return float(((aligned["actual"] - aligned["expected"]) * np.log(aligned["actual"] / aligned["expected"])).sum())


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "drift_report.html"
    json_path = output_dir / "drift_report.json"

    if Report is not None and DataDriftPreset is not None:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(html_path))
        payload = report.as_dict()
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return {"engine": "evidently", "html_path": str(html_path), "json_path": str(json_path)}

    drift_rows = []
    common = [col for col in reference.columns if col in current.columns and pd.api.types.is_numeric_dtype(reference[col])]
    for column in common:
        drift_rows.append(
            {
                "feature": column,
                "psi": _psi(reference[column].dropna(), current[column].dropna()),
                "reference_mean": float(reference[column].mean()),
                "current_mean": float(current[column].mean()),
            }
        )
    payload = {"engine": "fallback", "features": drift_rows}
    html_path.write_text(
        "<html><body><h1>Drift report</h1><pre>"
        + json.dumps(payload, indent=2)
        + "</pre></body></html>",
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"engine": "fallback", "html_path": str(html_path), "json_path": str(json_path)}
