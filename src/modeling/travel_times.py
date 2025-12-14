"""
Travel-time modeling helper.

Loads bin-level Gamma stats and lognormal GLM coefficients so the Streamlit app
can query travel minutes per mode.
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
DEFAULT_STATS_DIR = PROJECT_ROOT / "data" / "derived" / "travel_stats"
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
        columns = [
            "mode",
            "distance_bin",
            "is_rush",
            "is_weekend",
            "sample_count",
            "mean_min",
            "var_min",
            "gamma_k",
            "gamma_theta",
        ]
        print(f"[travel_times] Warning: {path} missing. Gamma fallback unavailable.")
        return pd.DataFrame(columns=columns)
    df = pd.read_parquet(path)
    return df


def _load_lognormal(stats_dir: Path) -> Dict[str, Dict]:
    path = stats_dir / "travel_lognormal_glm.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def get_stats(stats_dir: Path = DEFAULT_STATS_DIR):
    stats_dir = Path(stats_dir)
    bin_df = _load_bin_stats(stats_dir)
    lognormal = _load_lognormal(stats_dir)
    return bin_df, lognormal


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


def lognormal_prediction(
    mode: str,
    distance_km: float,
    is_rush: bool,
    is_weekend: bool,
    *,
    is_ebike: bool = False,
    stats_dir: Path = DEFAULT_STATS_DIR,
) -> Optional[TravelEstimate]:
    _, payload = get_stats(stats_dir)
    models = (payload or {}).get("models", {})
    metrics = (payload or {}).get("metrics", {})
    model = models.get(mode)
    if not model:
        return None
    coeffs = model.get("coefficients", {})
    sigma = float(model.get("sigma", float("nan")))
    features = {
        "const": 1.0,
        "distance_km": distance_km,
        "distance_sq": distance_km ** 2,
        "is_rush": 1.0 if is_rush else 0.0,
        "is_weekend": 1.0 if is_weekend else 0.0,
        "is_ebike": 1.0 if is_ebike else 0.0,
    }
    mu = 0.0
    for name, value in features.items():
        mu += float(coeffs.get(name, 0.0)) * value
    if not np.isfinite(mu):
        return None
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.0
    mean_minutes = float(np.exp(mu + 0.5 * sigma ** 2))
    metric_info = metrics.get(mode, {})
    details = {
        "sigma_log": sigma,
        "coefficients": coeffs,
        "samples": float(metric_info.get("samples", 0.0)),
        "r_squared_log": float(metric_info.get("r_squared_log", float("nan"))),
    }
    return TravelEstimate(
        source="lognormal_glm",
        minutes=mean_minutes,
        details=details,
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
    rideable_type: Optional[str] = None,
    stats_dir: Path = DEFAULT_STATS_DIR,
    speed_fallback_kmh: float = 10.0,
) -> TravelEstimate:
    is_ebike = bool(rideable_type and str(rideable_type).lower() == "electric_bike")
    lognormal_est = lognormal_prediction(
        mode,
        distance_km,
        is_rush,
        is_weekend,
        is_ebike=is_ebike,
        stats_dir=stats_dir,
    )
    if lognormal_est:
        return lognormal_est

    distance_bin = assign_distance_bin(distance_km)
    if distance_bin:
        gamma_est = lookup_gamma_mean(
            mode, distance_bin, is_rush, is_weekend, stats_dir=stats_dir
        )
        if gamma_est:
            return gamma_est
    return legacy_speed_fallback(distance_km, speed_fallback_kmh)


__all__ = [
    "TravelEstimate",
    "estimate_travel_minutes",
    "assign_distance_bin",
]
