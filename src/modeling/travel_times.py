"""
Component 2 travel-time modeling helper.

Loads bin-level Gamma stats (from scripts/component2_build_travel_stats.py) and
regression fallbacks so the Streamlit app can query travel minutes per mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATS_DIR = PROJECT_ROOT / "outputs" / "travel_stats"
BIN_WIDTH_KM = 2.0
MAX_DISTANCE_KM = 12.0
GAMMA_MIN_SAMPLES = 50


@dataclass(frozen=True)
class TravelEstimate:
    source: str
    minutes: float
    details: Dict[str, float]


def assign_distance_bin(dist_km: float) -> Optional[str]:
    if dist_km <= 0 or dist_km > MAX_DISTANCE_KM:
        return None
    idx = int(dist_km // BIN_WIDTH_KM)
    low = idx * BIN_WIDTH_KM
    high = low + BIN_WIDTH_KM
    return f"{int(low)}-{int(high)}km"


def _load_bin_stats(stats_dir: Path) -> pd.DataFrame:
    path = stats_dir / "travel_bins.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing travel bins cache at {path}. "
            "Run scripts/component2_build_travel_stats.py first."
        )
    df = pd.read_parquet(path)
    return df


def _load_regression(stats_dir: Path) -> Dict[str, Dict[str, float]]:
    path = stats_dir / "travel_regression.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def get_stats(stats_dir: Path = DEFAULT_STATS_DIR):
    stats_dir = Path(stats_dir)
    bin_df = _load_bin_stats(stats_dir)
    reg = _load_regression(stats_dir)
    return bin_df, reg


def lookup_gamma_mean(
    mode: str,
    distance_bin: str,
    is_rush: bool,
    is_weekend: bool,
    stats_dir: Path = DEFAULT_STATS_DIR,
) -> Optional[TravelEstimate]:
    bin_df, _ = get_stats(stats_dir)
    subset = bin_df[
        (bin_df["mode"] == mode)
        & (bin_df["distance_bin"] == distance_bin)
        & (bin_df["is_rush"] == is_rush)
        & (bin_df["is_weekend"] == is_weekend)
    ]
    if subset.empty:
        return None
    row = subset.iloc[0]
    if row["sample_count"] < GAMMA_MIN_SAMPLES or not np.isfinite(row["mean_min"]):
        return None
    return TravelEstimate(
        source="gamma_bin",
        minutes=float(row["mean_min"]),
        details={
            "sample_count": float(row["sample_count"]),
            "gamma_k": float(row["gamma_k"]) if pd.notna(row["gamma_k"]) else np.nan,
            "gamma_theta": float(row["gamma_theta"]) if pd.notna(row["gamma_theta"]) else np.nan,
            "mean": float(row["mean_min"]),
            "var": float(row["var_min"]),
        },
    )


def regression_fallback(
    mode: str,
    distance_km: float,
    is_rush: bool,
    is_weekend: bool,
    stats_dir: Path = DEFAULT_STATS_DIR,
    min_minutes: float = 2.0,
) -> Optional[TravelEstimate]:
    _, reg = get_stats(stats_dir)
    coeffs = reg.get(mode)
    if not coeffs:
        return None
    minutes = (
        coeffs["intercept"]
        + coeffs["beta_distance"] * distance_km
        + coeffs["beta_rush"] * (1 if is_rush else 0)
        + coeffs["beta_weekend"] * (1 if is_weekend else 0)
    )
    minutes = max(min_minutes, float(minutes))
    return TravelEstimate(
        source="regression",
        minutes=minutes,
        details=coeffs,
    )


def legacy_speed_fallback(distance_km: float, speed_kmh: float) -> TravelEstimate:
    minutes = (distance_km / speed_kmh) * 60.0 if speed_kmh > 0 else float("nan")
    return TravelEstimate(
        source="speed_heuristic",
        minutes=float(minutes),
        details={"speed_kmh": speed_kmh},
    )


def estimate_travel_minutes(
    mode: str,
    distance_km: float,
    *,
    is_rush: bool,
    is_weekend: bool,
    stats_dir: Path = DEFAULT_STATS_DIR,
    speed_fallback_kmh: float = 10.0,
) -> TravelEstimate:
    distance_bin = assign_distance_bin(distance_km)
    if distance_bin:
        gamma_est = lookup_gamma_mean(
            mode, distance_bin, is_rush, is_weekend, stats_dir=stats_dir
        )
        if gamma_est:
            return gamma_est
    reg_est = regression_fallback(
        mode, distance_km, is_rush, is_weekend, stats_dir=stats_dir
    )
    if reg_est:
        return reg_est
    return legacy_speed_fallback(distance_km, speed_fallback_kmh)


__all__ = [
    "TravelEstimate",
    "estimate_travel_minutes",
    "assign_distance_bin",
]
