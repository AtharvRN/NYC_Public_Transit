#!/usr/bin/env python3
"""
Build Component 2 travel-time statistics for taxi and Citi Bike trips.

Outputs two cache files:
1. Parquet with Gamma-fit stats per (mode, distance_bin, rush_flag, weekend_flag)
2. JSON with regression coefficients per mode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "travel_stats"

BIN_WIDTH_KM = 2.0
MAX_DISTANCE_KM = 12.0  # filter trips beyond this to keep equal-width bins
GAMMA_MIN_SAMPLES = 50
REGRESSION_MIN_SAMPLES = 200
RUSH_HOURS = [(7, 10), (16, 19)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taxi-path",
        type=Path,
        default=DATA_ROOT / "yellow_tripdata_2024-01.parquet",
        help="Taxi parquet path (Jan 2024 sample).",
    )
    parser.add_argument(
        "--bike-glob",
        type=str,
        default="202401-citibike-tripdata_*.csv",
        help="Glob under data/raw/citibike for bike CSVs.",
    )
    parser.add_argument(
        "--bike-root",
        type=Path,
        default=DATA_ROOT / "citibike",
        help="Directory containing Citi Bike CSVs.",
    )
    parser.add_argument(
        "--taxi-max-rows",
        type=int,
        default=4_000_000,
        help="Optional cap on taxi rows (None for all).",
    )
    parser.add_argument(
        "--bike-max-rows",
        type=int,
        default=2_000_000,
        help="Optional cap on first Citi Bike CSV (others read fully).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to store Parquet/JSON outputs.",
    )
    return parser.parse_args()


def is_rush_hour(hour: int) -> bool:
    return any(start <= hour < end for start, end in RUSH_HOURS)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def assign_bins(dist_km: pd.Series) -> pd.Series:
    bins = np.arange(0, MAX_DISTANCE_KM + BIN_WIDTH_KM, BIN_WIDTH_KM)
    labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}km" for i in range(len(bins) - 1)]
    return pd.cut(dist_km, bins=bins, labels=labels, right=False)


def prep_common_features(df: pd.DataFrame, event_col: str) -> pd.DataFrame:
    df = df.copy()
    df["event_time"] = pd.to_datetime(df[event_col])
    df = df.dropna(subset=["event_time", "travel_min", "distance_km"])
    df = df[
        (df["travel_min"] >= 1.0)
        & (df["travel_min"] <= 120.0)
        & (df["distance_km"] > 0)
        & (df["distance_km"] <= MAX_DISTANCE_KM)
    ]
    df["hour"] = df["event_time"].dt.hour
    df["is_weekend"] = df["event_time"].dt.dayofweek >= 5
    df["is_rush"] = df["hour"].apply(is_rush_hour)
    df["distance_bin"] = assign_bins(df["distance_km"])
    return df.dropna(subset=["distance_bin"])


def load_taxi_trips(path: Path, max_rows: int | None) -> pd.DataFrame:
    cols = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
    ]
    df = pd.read_parquet(path, columns=cols)
    if max_rows:
        df = df.head(max_rows)
    df["pickup_dt"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["dropoff_dt"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    df["travel_min"] = (df["dropoff_dt"] - df["pickup_dt"]).dt.total_seconds() / 60.0
    # TLC distance is miles; convert to km
    df["distance_km"] = df["trip_distance"].astype(float) * 1.60934
    df = prep_common_features(df, event_col="pickup_dt")
    df["mode"] = "taxi"
    return df


def load_bike_trips(root: Path, glob_pattern: str, max_rows: int | None) -> pd.DataFrame:
    files = sorted(root.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No Citi Bike files under {root} matching {glob_pattern}")
    frames = []
    for idx, csv_path in enumerate(files):
        frames.append(
            pd.read_csv(
                csv_path,
                nrows=max_rows if (max_rows and idx == 0) else None,
                usecols=[
                    "started_at",
                    "ended_at",
                    "start_lat",
                    "start_lng",
                    "end_lat",
                    "end_lng",
                ],
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["start_dt"] = pd.to_datetime(df["started_at"])
    df["end_dt"] = pd.to_datetime(df["ended_at"])
    df["travel_min"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 60.0
    df["distance_km"] = haversine_km(
        df["start_lat"],
        df["start_lng"],
        df["end_lat"],
        df["end_lng"],
    )
    df = prep_common_features(df, event_col="start_dt")
    df["mode"] = "bike"
    return df


def fit_gamma(mean_val: float, var_val: float) -> Tuple[float, float]:
    if mean_val <= 0 or var_val <= 0:
        return np.nan, np.nan
    k = (mean_val ** 2) / var_val
    theta = var_val / mean_val
    return float(k), float(theta)


def summarize_bins(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["mode", "distance_bin", "is_rush", "is_weekend"])["travel_min"]
        .agg(
            sample_count="size",
            mean_min="mean",
            var_min=lambda x: x.var(ddof=0),
        )
        .reset_index()
    )
    gamma_params = grouped.apply(
        lambda row: fit_gamma(row["mean_min"], row["var_min"])
        if row["sample_count"] >= GAMMA_MIN_SAMPLES
        else (np.nan, np.nan),
        axis=1,
    )
    grouped[["gamma_k", "gamma_theta"]] = pd.DataFrame(
        gamma_params.tolist(), index=grouped.index
    )
    return grouped


def fit_regression(df: pd.DataFrame, mode: str) -> Dict[str, float]:
    subset = df[df["mode"] == mode]
    if len(subset) < REGRESSION_MIN_SAMPLES:
        raise ValueError(
            f"Need at least {REGRESSION_MIN_SAMPLES} samples for {mode} regression."
        )
    x = np.column_stack(
        [
            np.ones(len(subset)),
            subset["distance_km"].values,
            subset["is_rush"].astype(int).values,
            subset["is_weekend"].astype(int).values,
        ]
    )
    y = subset["travel_min"].values
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    keys = ["intercept", "beta_distance", "beta_rush", "beta_weekend"]
    return dict(zip(keys, beta))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading taxi trips...")
    taxi = load_taxi_trips(args.taxi_path, args.taxi_max_rows)
    print(f"Taxi sample: {len(taxi):,} rows after filtering.")
    print("Loading bike trips...")
    bike = load_bike_trips(args.bike_root, args.bike_glob, args.bike_max_rows)
    print(f"Bike sample: {len(bike):,} rows after filtering.")

    combined = pd.concat([taxi, bike], ignore_index=True)
    bin_stats = summarize_bins(combined)
    stats_path = args.output_dir / "travel_bins.parquet"
    bin_stats.to_parquet(stats_path, index=False)
    print(f"Wrote bin summaries to {stats_path}")

    regression = {}
    for mode in ["taxi", "bike"]:
        try:
            coeffs = fit_regression(combined, mode)
            regression[mode] = coeffs
        except ValueError as exc:
            print(f"Skipping regression for {mode}: {exc}")
    regr_path = args.output_dir / "travel_regression.json"
    regr_path.write_text(json.dumps(regression, indent=2))
    print(f"Wrote regression coefficients to {regr_path}")


if __name__ == "__main__":
    main()
