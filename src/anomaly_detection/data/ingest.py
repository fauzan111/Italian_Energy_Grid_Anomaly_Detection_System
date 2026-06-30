from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SourceBundle:
    total_load: pd.DataFrame
    actual_generation: pd.DataFrame


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s*\[mw\]\s*", "_mw_", value)
    value = value.replace("/", "_").replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = _slug(str(column))
        renamed[column] = key
    return frame.rename(columns=renamed)


def load_total_load_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = _normalize_columns(frame)
    frame["timestamp"] = pd.to_datetime(frame["date"], utc=False)
    grouped = (
        frame.groupby("timestamp")
        .agg(
            total_load_mw=("total_load_mw", "sum"),
            forecast_total_load_mw=("forecast_total_load_mw", "sum"),
            zone_count=("bidding_zone", "nunique"),
        )
        .sort_index()
    )
    zone_pivot = (
        frame.pivot_table(
            index="timestamp",
            columns="bidding_zone",
            values="total_load_mw",
            aggfunc="sum",
        )
        .sort_index()
        .rename(columns=lambda c: f"load_zone__{_slug(str(c))}")
    )
    grouped = grouped.join(zone_pivot, how="left")
    grouped["load_gap_mw"] = grouped["forecast_total_load_mw"] - grouped["total_load_mw"]
    grouped["load_gap_pct"] = grouped["load_gap_mw"] / grouped["total_load_mw"].replace(0, np.nan)
    return grouped


def load_actual_generation_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = _normalize_columns(frame)
    frame["timestamp"] = pd.to_datetime(frame["date"], utc=False)
    grouped = (
        frame.groupby("timestamp")
        .agg(actual_generation_mw=("actual_generation", "sum"), source_count=("primary_source", "nunique"))
        .sort_index()
    )
    source_pivot = (
        frame.pivot_table(
            index="timestamp",
            columns="primary_source",
            values="actual_generation",
            aggfunc="sum",
        )
        .sort_index()
        .rename(columns=lambda c: f"generation_source__{_slug(str(c))}")
    )
    grouped = grouped.join(source_pivot, how="left")
    return grouped


def build_feature_frame(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    total_load = load_total_load_csv(raw_dir / "TotalLoad_2025.csv")
    actual_generation = load_actual_generation_csv(raw_dir / "ActualGeneration_2025.csv")

    frame = total_load.join(actual_generation, how="outer")
    frame = frame.sort_index().ffill().bfill()
    frame = add_time_features(frame)
    numeric = frame.select_dtypes(include="number").copy()
    numeric = numeric.replace([float("inf"), float("-inf")], pd.NA).ffill().bfill()
    numeric.index.name = "timestamp"
    return numeric


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    index = pd.DatetimeIndex(result.index)
    result["hour"] = index.hour
    result["day_of_week"] = index.dayofweek
    result["day_of_year"] = index.dayofyear
    result["month"] = index.month
    result["is_weekend"] = (index.dayofweek >= 5).astype(int)
    angle = 2 * np.pi * index.hour / 24.0
    result["hour_sin"] = np.sin(angle)
    result["hour_cos"] = np.cos(angle)
    return result


def train_validation_split(frame: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = max(1, int(len(frame) * train_fraction))
    return frame.iloc[:cutoff].copy(), frame.iloc[cutoff:].copy()


def available_raw_files(raw_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    return sorted(raw_dir.glob("*.csv"))


def ensure_raw_data(raw_dir: str | Path, expected: Iterable[str] = ("TotalLoad_2025.csv", "ActualGeneration_2025.csv")) -> None:
    raw_dir = Path(raw_dir)
    missing = [name for name in expected if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw data files in {raw_dir}: {', '.join(missing)}")
