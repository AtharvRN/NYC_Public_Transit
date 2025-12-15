#!/usr/bin/env python3
"""
Build travel-time statistics for taxi and Citi Bike trips.

Outputs cache files:
1. Parquet with Gamma-fit stats per (mode, distance_bin, rush_flag, weekend_flag)
2. JSON with lognormal GLM coefficients/metrics per mode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived" / "travel_stats"

SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from modeling.travel_diagnostics import (  # noqa: E402
    BIN_WIDTH_KM,
    GAMMA_MIN_SAMPLES,
    MAX_DISTANCE_KM,
    REGRESSION_MIN_SAMPLES,
    fit_lognormal_glm,
    load_bike_trips,
    load_taxi_trips,
    summarize_bins,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taxi-path",
        type=Path,
        default=None,
        help="Taxi parquet path (default: all yellow_tripdata_2024-*.parquet).",
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


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.taxi_path is None:
        taxi_files = sorted(DATA_ROOT.glob("yellow_tripdata_2024-*.parquet"))
    else:
        taxi_files = [args.taxi_path]
    taxi_frames = []
    for path in taxi_files:
        if not path.exists():
            print(f"[builder] Warning: {path} missing, skipping.")
            continue
        print(f"Loading taxi trips from {path.name}...")
        taxi_frames.append(load_taxi_trips(path, args.taxi_max_rows))
    if not taxi_frames:
        raise FileNotFoundError("No taxi data files were loaded. Check --taxi-path or data/raw.")
    taxi = pd.concat(taxi_frames, ignore_index=True)
    print(f"Taxi sample: {len(taxi):,} rows after filtering.")
    print("Loading bike trips...")
    bike = load_bike_trips(args.bike_root, args.bike_glob, args.bike_max_rows)
    # print("bike : ", bike.head())
    print(f"Bike sample: {len(bike):,} rows after filtering.")

    combined = pd.concat([taxi, bike], ignore_index=True)
    bin_stats = summarize_bins(combined)
    # print("bin_stats", bin_stats)
    stats_path = args.output_dir / "travel_bins.parquet"
    bin_stats.to_parquet(stats_path, index=False)
    print(f"Wrote bin summaries to {stats_path}")

    lognormal_models, lognormal_metrics = fit_lognormal_glm(combined)
    print("lognormal models : ", lognormal_models)
    print("lognormal metrics : ", lognormal_metrics)
    
    if lognormal_models:
        lognormal_payload = {
            "design_columns": ["const", "distance_km", "distance_sq", "is_rush", "is_weekend"],
            "models": lognormal_models,
            "metrics": lognormal_metrics,
        }
        lognormal_path = args.output_dir / "travel_lognormal_glm.json"
        lognormal_path.write_text(json.dumps(lognormal_payload, indent=2))
        print(f"Wrote lognormal GLM coefficients to {lognormal_path}")
    else:
        print("No lognormal GLM models were fit (insufficient samples).")


if __name__ == "__main__":
    main()
