#!/usr/bin/env python3
"""
Precompute wait-time statistics for taxi and Citi Bike pickups.

This script loads the raw Parquet/CSV files defined in modeling.wait_times and
persists the per-zone/station Poisson+wait summaries to `outputs/wait_stats`.
The Streamlit app and other tools can then load the cached stats without
touching raw data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from modeling.wait_times import compute_wait_stats, set_taxi_paths_override  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "wait_stats",
        help="Directory where wait-time stats will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if cache files already exist.",
    )
    parser.add_argument(
        "--taxi-paths",
        type=Path,
        nargs="+",
        help="Override list of taxi Parquet files (default: yellow_tripdata_2024-01.parquet).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.taxi_paths:
        set_taxi_paths_override(args.taxi_paths)

    modes = ("taxi", "bike")
    for mode in modes:
        print(f"[wait_stats] Computing {mode} summaries...")
        df = compute_wait_stats(mode, cache_dir=args.cache_dir, force_recompute=args.force)
        print(f"[wait_stats] {mode} rows: {len(df):,}.")


if __name__ == "__main__":
    main()
