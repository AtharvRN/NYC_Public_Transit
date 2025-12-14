"""
Wait-time modeling helpers shared between notebooks and the Streamlit app.

This module centralizes the logic that was originally prototyped in
`notebooks/wait_times/combined_dashboard.ipynb`.  It can summarize taxi or Citi
Bike arrivals per location/hour and cache the resulting wait-time statistics.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from modeling.poisson_zone import load_taxi_pickups

ModeName = Literal["taxi", "bike"]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs" / "wait_stats"
_TAXI_PATHS_OVERRIDE: Optional[Tuple[Path, ...]] = None


def _is_rush_hour(hour: int, ranges=((7, 10), (16, 19))) -> bool:
    return any(lo <= hour < hi for lo, hi in ranges)


@dataclass(frozen=True)
class ModeConfig:
    name: ModeName
    loader: Callable[[], pd.DataFrame]
    cache_stub: str
    location_dtype: Callable[[object], object]


def _resolve_taxi_paths() -> Tuple[Path, ...]:
    if _TAXI_PATHS_OVERRIDE:
        return _TAXI_PATHS_OVERRIDE
    paths = sorted((PROJECT_ROOT / "data" / "raw").glob("yellow_tripdata_2024-*.parquet"))
    if not paths:
        raise FileNotFoundError("No taxi Parquet files found under data/raw.")
    return tuple(paths)


def set_taxi_paths_override(paths: Sequence[Path]) -> None:
    global _TAXI_PATHS_OVERRIDE
    _TAXI_PATHS_OVERRIDE = tuple(Path(p) for p in paths)


def _load_taxi_events(max_rows: int = 4_000_000) -> pd.DataFrame:
    """Return taxi pickup events with columns [location_id, event_time]."""
    frames = []
    for path in _resolve_taxi_paths():
        if not path.exists():
            print(f"[wait_times] Warning: {path} missing, skipping.")
            continue
        trips = load_taxi_pickups(path, max_rows=max_rows)
        frames.append(trips)
    if not frames:
        raise FileNotFoundError("No taxi Parquet files were loaded.")
    trips = pd.concat(frames, ignore_index=True)
    trips["event_time"] = trips["event_time"].dt.tz_convert(None)
    trips["hour"] = trips["event_time"].dt.hour
    trips["is_weekend"] = trips["event_time"].dt.dayofweek >= 5
    trips["is_rush"] = trips["hour"].apply(_is_rush_hour)
    trips = trips.rename(columns={"PULocationID": "location_id"})
    return trips[["location_id", "event_time", "hour", "is_weekend", "is_rush"]].dropna(subset=["location_id"])


def _load_bike_events(max_rows: int = 5_000_000) -> pd.DataFrame:
    """Return Citi Bike trips with columns [location_id, event_time]."""
    data_root = PROJECT_ROOT / "data" / "raw" / "citibike"
    files = sorted(data_root.glob("202401-citibike-tripdata_*.csv"))
    if not files:
        raise FileNotFoundError(f"No Citi Bike CSVs found under {data_root}")
    frames = []
    for idx, csv_path in enumerate(files):
        frames.append(
            pd.read_csv(
                csv_path,
                nrows=max_rows if (max_rows and idx == 0) else None,
                dtype={"start_station_id": str, "end_station_id": str},
                usecols=["started_at", "start_station_id"],
                low_memory=False,
            )
        )
    trips = pd.concat(frames, ignore_index=True)
    trips = trips.rename(columns={"start_station_id": "location_id"})
    trips["location_id"] = trips["location_id"].astype(str)
    trips["event_time"] = pd.to_datetime(trips["started_at"])
    trips = trips.dropna(subset=["location_id", "event_time"])
    trips["hour"] = trips["event_time"].dt.hour
    trips["is_weekend"] = trips["event_time"].dt.dayofweek >= 5
    trips["is_rush"] = trips["hour"].apply(_is_rush_hour)
    return trips[["location_id", "event_time", "hour", "is_weekend", "is_rush"]]


MODE_CONFIG: Dict[ModeName, ModeConfig] = {
    "taxi": ModeConfig(
        name="taxi",
        loader=_load_taxi_events,
        cache_stub="taxi_hourly_waits",
        location_dtype=lambda v: int(v),
    ),
    "bike": ModeConfig(
        name="bike",
        loader=_load_bike_events,
        cache_stub="bike_hourly_waits",
        location_dtype=lambda v: str(v),
    ),
}


def _fit_nb(mean: float, variance: float) -> Tuple[float, float]:
    """Method-of-moments negative-binomial parameters."""
    if mean <= 0 or variance <= mean:
        return np.nan, np.nan
    r = mean**2 / (variance - mean)
    p = r / (r + mean)
    return float(r), float(p)


def _summarize_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate arrivals per location/hour to compute mean & variance."""
    events = events.copy()
    if "hour" not in events:
        events["hour"] = events["event_time"].dt.hour
    if "is_weekend" not in events:
        events["is_weekend"] = events["event_time"].dt.dayofweek >= 5
    if "is_rush" not in events:
        events["is_rush"] = events["hour"].apply(_is_rush_hour)
    events["bucket"] = events["event_time"].dt.floor("H")

    hourly_counts = (
        events.groupby(["location_id", "hour", "is_weekend", "is_rush", "bucket"])
        .size()
        .rename("arrivals")
        .reset_index()
    )

    stats = (
        hourly_counts.groupby(["location_id", "hour", "is_weekend", "is_rush"])["arrivals"]
        .agg(
            mean_arrivals="mean",
            variance=lambda x: x.var(ddof=0),
            obs_hours="size",
        )
        .reset_index()
    )
    stats["variance"] = stats["variance"].fillna(0.0)
    return stats


def _summarize_waits(events: pd.DataFrame) -> pd.DataFrame:
    """Compute empirical wait minutes per location/hour."""
    events = events.copy()
    if "hour" not in events:
        events["hour"] = events["event_time"].dt.hour
    if "is_weekend" not in events:
        events["is_weekend"] = events["event_time"].dt.dayofweek >= 5
    if "is_rush" not in events:
        events["is_rush"] = events["hour"].apply(_is_rush_hour)
    events = events.sort_values(
        ["location_id", "is_weekend", "is_rush", "hour", "event_time"]
    )
    events["prev_event"] = events.groupby(
        ["location_id", "is_weekend", "is_rush", "hour"]
    )["event_time"].shift(1)
    events["wait_minutes"] = (
        events["event_time"] - events["prev_event"]
    ).dt.total_seconds() / 60.0

    valid = events["wait_minutes"].between(0.01, 180)
    waits = (
        events.loc[valid]
        .groupby(["location_id", "hour", "is_weekend", "is_rush"])["wait_minutes"]
        .agg(
            mean_wait="mean",
            median_wait="median",
            wait_count="size",
        )
        .reset_index()
    )
    return waits


def _build_stats(events: pd.DataFrame, mode: ModeName) -> pd.DataFrame:
    counts = _summarize_counts(events)
    waits = _summarize_waits(events)
    merged = counts.merge(
        waits,
        on=["location_id", "hour", "is_weekend", "is_rush"],
        how="left",
    )

    merged["dispersion"] = np.where(
        merged["mean_arrivals"] > 0,
        merged["variance"] / merged["mean_arrivals"],
        np.nan,
    )
    nb = merged.apply(
        lambda row: _fit_nb(row["mean_arrivals"], row["variance"]), axis=1
    )
    merged[["nb_r", "nb_p"]] = pd.DataFrame(nb.tolist(), index=merged.index)
    merged["poisson_wait"] = np.where(
        merged["mean_arrivals"] > 0, 60.0 / merged["mean_arrivals"], np.nan
    )
    merged["mode"] = mode
    return merged[
        [
            "mode",
            "location_id",
            "hour",
            "is_weekend",
            "is_rush",
            "mean_arrivals",
            "variance",
            "dispersion",
            "nb_r",
            "nb_p",
            "obs_hours",
            "mean_wait",
            "median_wait",
            "wait_count",
            "poisson_wait",
        ]
    ]


def compute_wait_stats(
    mode: ModeName,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Calculate and cache wait-time stats for the requested mode.

    Returns a DataFrame keyed by (location_id, hour, is_weekend, is_rush).
    """
    config = MODE_CONFIG[mode]
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / f"{config.cache_stub}.parquet"
    json_path = cache_dir / f"{config.cache_stub}.json"

    if parquet_path.exists() and not force_recompute:
        df = pd.read_parquet(parquet_path)
        return df

    events = config.loader()
    events["location_id"] = events["location_id"].apply(config.location_dtype)
    stats = _build_stats(events, mode)
    stats.to_parquet(parquet_path, index=False)
    json_path.write_text(stats.to_json(orient="records"))
    return stats


@lru_cache(maxsize=None)
def load_wait_stats(
    mode: ModeName,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Load cached stats (computing if needed)."""
    cache_dir = Path(cache_dir)
    config = MODE_CONFIG[mode]
    parquet_path = cache_dir / f"{config.cache_stub}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return compute_wait_stats(mode, cache_dir=cache_dir, force_recompute=False)


def _normalize_key(mode: ModeName, location_id: object) -> object:
    """Apply config-specific casting for location IDs."""
    return MODE_CONFIG[mode].location_dtype(location_id)


def get_wait_summary(
    mode: ModeName,
    location_id: object,
    hour: int,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    is_weekend: bool = False,
    is_rush: bool = False,
) -> Optional[dict]:
    """
    Return wait stats for a single mode/location/hour.

    The result dictionary is suitable for feeding into UI code:
    {
        'mean_wait': float,
        'poisson_wait': float,
        'dispersion': float,
        'nb_r': float,
        'nb_p': float,
        ...
    }
    """
    df = load_wait_stats(mode, cache_dir=cache_dir)
    location_id = _normalize_key(mode, location_id)
    subset = df[
        (df["location_id"] == location_id)
        & (df["hour"] == int(hour))
        & (df["is_weekend"] == bool(is_weekend))
        & (df["is_rush"] == bool(is_rush))
    ]
    if subset.empty:
        return None
    return subset.iloc[0].to_dict()


__all__ = [
    "compute_wait_stats",
    "load_wait_stats",
    "get_wait_summary",
    "set_taxi_paths_override",
]
