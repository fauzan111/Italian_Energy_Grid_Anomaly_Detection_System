from __future__ import annotations

import csv
import datetime as dt
import argparse
import json
import pathlib
import urllib.request
from typing import Any


API_URL = "https://dati.terna.it/api/sitecore/dati/downloadcenter/recordsv2"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "terna"
DEFAULT_DATASETS = ["TotalLoad", "ActualGeneration"]
DEFAULT_YEARS = ["2025"]


def dotnet_ticks_to_iso(value: int) -> str:
    base = dt.datetime(1, 1, 1)
    return (base + dt.timedelta(milliseconds=value)).isoformat()


def fetch_dataset(dataset: str, year: str) -> dict[str, Any]:
    payload = {
        "filterDataset": dataset,
        "filterYear": year,
        "pageSize": "1",
        "pageIndex": "0",
        "db": "dati",
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        summary = json.loads(response.read().decode("utf-8"))

    payload["pageSize"] = str(summary["Count"])
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def write_outputs(dataset: str, year: str, payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / f"{dataset}_{year}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = OUTPUT_DIR / f"{dataset}_{year}.csv"
    columns = list(payload["Columns"].keys())
    date_columns = {name for name, kind in payload["Columns"].items() if kind == "Date"}

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in payload["Data"]:
            formatted = []
            for column, value in zip(columns, row):
                if column in date_columns and isinstance(value, int):
                    formatted.append(dotnet_ticks_to_iso(value))
                else:
                    formatted.append(value)
            writer.writerow(formatted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TERNA datasets")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--years", nargs="+", default=DEFAULT_YEARS)
    args = parser.parse_args()

    for year in args.years:
        for dataset in args.datasets:
            payload = fetch_dataset(dataset, year)
            write_outputs(dataset, year, payload)


if __name__ == "__main__":
    main()
