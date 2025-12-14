"""
Reusable helpers for the wait-time exploratory notebook.

The original `notebooks/wait_times.ipynb` mixed data-loading, aggregation,
plotting, and ipywidget wiring.  This module centralizes the heavy lifting so
the notebook can focus on narrative structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Image, Markdown, display
from plotly.subplots import make_subplots

try:  # ipywidgets is optional outside notebooks
    import ipywidgets as widgets
except ImportError:  # pragma: no cover - notebook dependency
    widgets = None  # type: ignore[assignment]

from modeling.poisson_zone import (
    attach_zone_metadata,
    bucket_counts_by_group,
    load_taxi_pickups,
)
from scipy.stats import nbinom as sp_nbinom, poisson as sp_poisson


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"

DEFAULT_FREQ_OPTIONS = ("5min", "15min", "30min", "1H")
DEFAULT_BUCKET_BASE = "15min"


def is_rush(hour: int, ranges: Iterable[Tuple[int, int]] = ((7, 10), (16, 19))) -> bool:
    """Return True if `hour` is in a rush hour window."""
    return any(lo <= hour < hi for lo, hi in ranges)


def cohort_label(is_weekend: pd.Series, is_rush_flags: pd.Series) -> pd.Series:
    """Vectorized cohort labeler: weekday/weekend + rush/offpeak."""
    weekend = np.where(is_weekend, "weekend", "weekday")
    rush = np.where(is_rush_flags, "rush", "offpeak")
    return pd.Series(weekend + "_" + rush, index=is_weekend.index)


def _screen_locations(
    trips: pd.DataFrame,
    *,
    group_col: str,
    base_freq: str,
    min_mean: float,
    min_nonzero: float,
) -> List[str]:
    """Return active locations that pass the mean/nonzero thresholds."""
    counts = bucket_counts_by_group(trips, freq=base_freq, group_cols=group_col)
    means = counts.mean()
    nonzero = (counts > 0).mean()
    active = [
        loc
        for loc in counts.columns
        if means[loc] >= min_mean and nonzero[loc] >= min_nonzero
    ]
    if not active:
        return sorted(trips[group_col].dropna().unique())
    return sorted(active)


def _fit_nb(series: pd.Series) -> Tuple[float, float]:
    """Method-of-moments NB fit returning (r, p)."""
    mean = series.mean()
    var = series.var(ddof=0)
    if var <= mean or mean <= 0:
        return np.nan, np.nan
    r = mean**2 / (var - mean)
    p = r / (r + mean)
    return float(r), float(p)


@dataclass
class SelectionMetrics:
    """Bundle of statistics for a location/cohort/frequency combination."""

    location: str
    cohort: str
    freq: str
    series: pd.Series
    mean: float
    variance: float
    dispersion: float
    nb_r: float
    nb_p: float
    nonzero_frac: float
    diffs: pd.Series
    wait_mean: float

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "location": self.location,
                    "cohort": self.cohort,
                    "freq": self.freq,
                    "mean_arrivals": self.mean,
                    "variance": self.variance,
                    "dispersion": self.dispersion,
                    "nb_r": self.nb_r,
                    "nb_p": self.nb_p,
                    "mean_wait_min": self.wait_mean if not self.diffs.empty else np.nan,
                    "nonzero_frac": self.nonzero_frac,
                }
            ]
        )


@dataclass
class WaitDataset:
    """Encapsulates trips + cached aggregations for widget-driven exploration."""

    name: str
    trips: pd.DataFrame
    group_col: str
    locations: List[str]
    min_mean: float
    min_nonzero: float
    freq_options: Sequence[str] = DEFAULT_FREQ_OPTIONS
    default_freq: str = DEFAULT_BUCKET_BASE
    _count_cache: Dict[Tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    _diff_cache: Dict[Tuple[str, str], pd.Series] = field(default_factory=dict)

    @property
    def cohort_options(self) -> List[str]:
        return ["All trips"] + sorted(self.trips["cohort"].unique())

    def _subset(self, cohort: str) -> pd.DataFrame:
        if cohort == "All trips":
            return self.trips
        return self.trips[self.trips["cohort"] == cohort]

    def counts(self, cohort: str, freq: str) -> pd.DataFrame:
        key = (cohort, freq)
        if key not in self._count_cache:
            subset = self._subset(cohort)
            if subset.empty:
                self._count_cache[key] = pd.DataFrame()
            else:
                self._count_cache[key] = bucket_counts_by_group(
                    subset, freq=freq, group_cols=self.group_col
                )
        return self._count_cache[key]

    def wait_diffs(self, cohort: str, location: str) -> pd.Series:
        key = (cohort, location)
        if key not in self._diff_cache:
            subset = self._subset(cohort)
            if subset.empty:
                self._diff_cache[key] = pd.Series(dtype=float)
            else:
                location_subset = subset[
                    subset[self.group_col] == location
                ].sort_values("event_time")
                diffs = (
                    location_subset["event_time"]
                    .diff()
                    .dropna()
                    .dt.total_seconds()
                    / 60
                )
                diffs = diffs[(diffs > 0) & (diffs < 180)]
                self._diff_cache[key] = diffs
        return self._diff_cache[key]

    def compute_selection(
        self, location: str, cohort: str, freq: str
    ) -> Tuple[SelectionMetrics | None, str | None]:
        counts = self.counts(cohort, freq)
        if counts.empty:
            return None, "No trips for this cohort selection."
        if location not in counts.columns:
            return None, "Location missing for this selection."
        series = counts[location]
        nonzero_frac = (series > 0).mean()
        mean = series.mean()
        if mean < self.min_mean or nonzero_frac < self.min_nonzero:
            return (
                None,
                f"Selection too sparse (mean={mean:.2f}, nonzero={nonzero_frac:.2f}).",
            )
        var = series.var(ddof=0)
        disp = var / mean if mean > 0 else np.nan
        nb_r, nb_p = _fit_nb(series)
        diffs = self.wait_diffs(cohort, location)
        wait_mean = diffs.mean() if not diffs.empty else np.nan
        metrics = SelectionMetrics(
            location=location,
            cohort=cohort,
            freq=freq,
            series=series,
            mean=mean,
            variance=var,
            dispersion=disp,
            nb_r=nb_r,
            nb_p=nb_p,
            nonzero_frac=nonzero_frac,
            diffs=diffs,
            wait_mean=wait_mean,
        )
        return metrics, None


def create_combined_figure(
    metrics: SelectionMetrics, title_prefix: str | None = None
) -> go.Figure:
    """Create the arrivals histogram + wait-time overlay figure."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Arrivals per Bucket", "Wait Time Distribution"),
        horizontal_spacing=0.15,
    )

    series = metrics.series
    grid_max = max(series.max(), int(series.quantile(0.99)) + 5)
    grid = np.arange(0, grid_max + 1)
    obs = series.value_counts().reindex(grid, fill_value=0).values
    fig.add_trace(
        go.Bar(
            x=grid,
            y=obs,
            name="Observed",
            marker=dict(color="#4B6BFB"),
            opacity=0.75,
        ),
        row=1,
        col=1,
    )

    pois_exp = sp_poisson.pmf(grid, metrics.mean) * len(series)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=pois_exp,
            mode="lines",
            name="Poisson",
            line=dict(color="#FFA500", width=2),
        ),
        row=1,
        col=1,
    )

    if np.isfinite(metrics.nb_r) and np.isfinite(metrics.nb_p):
        if metrics.nb_r > 0 and 0 < metrics.nb_p < 1:
            nb_exp = sp_nbinom.pmf(grid, metrics.nb_r, metrics.nb_p) * len(series)
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=nb_exp,
                    mode="lines",
                    name="Neg-Bin",
                    line=dict(color="#D62728", width=3),
                ),
                row=1,
                col=1,
            )

    diffs = metrics.diffs
    if not diffs.empty and metrics.wait_mean and metrics.wait_mean > 0:
        fig.add_trace(
            go.Histogram(
                x=diffs,
                nbinsx=100,
                name="Empirical Wait",
                marker=dict(color="#4B6BFB"),
                opacity=0.75,
                histnorm="probability density",
            ),
            row=1,
            col=2,
        )
        lam = 1 / metrics.wait_mean
        x = np.linspace(0, diffs.max(), 200)
        pdf = lam * np.exp(-lam * x)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=pdf,
                mode="lines",
                name="Exponential",
                line=dict(color="#99FF00", width=2),
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Arrivals", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Minutes", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=2)

    title = title_prefix or f"{metrics.location} ({metrics.cohort}, {metrics.freq})"
    fig.update_layout(
        height=450,
        showlegend=True,
        template="plotly_white",
        title_text=title,
        title_x=0.5,
        title_xanchor="center",
    )
    return fig


class WaitDashboard:
    """Thin wrapper that wires `WaitDataset` into ipywidgets."""

    def __init__(self, dataset: WaitDataset, title: str):
        if widgets is None:
            raise ImportError("ipywidgets is required to build the wait dashboard.")
        self.dataset = dataset
        self.title = title
        self.widget = self._build()

    def _build(self) -> widgets.VBox:
        location_dd = widgets.Dropdown(
            options=self.dataset.locations, description="Location"
        )
        cohort_dd = widgets.Dropdown(
            options=self.dataset.cohort_options, description="Cohort"
        )
        freq_dd = widgets.Dropdown(
            options=self.dataset.freq_options,
            value=self.dataset.default_freq,
            description="Bucket",
        )
        out = widgets.Output()

        def refresh(*_):
            out.clear_output(wait=True)
            with out:
                metrics, message = self.dataset.compute_selection(
                    location_dd.value, cohort_dd.value, freq_dd.value
                )
                if message:
                    print(message)
                    return
                fig = create_combined_figure(
                    metrics,
                    title_prefix=f"{location_dd.value} "
                    f"({cohort_dd.value}, {freq_dd.value})",
                )
                display(fig)
                display(metrics.summary_frame())

        location_dd.observe(lambda *_: refresh(), names="value")
        cohort_dd.observe(lambda *_: refresh(), names="value")
        freq_dd.observe(lambda *_: refresh(), names="value")

        refresh()
        return widgets.VBox(
            [
                widgets.HTML(f"<h3>{self.title}</h3>"),
                widgets.HBox([location_dd, cohort_dd]),
                freq_dd,
                out,
            ]
        )


def _prepare_taxi_trips(
    paths: Sequence[Path],
    lookup_csv: Path | None,
    *,
    max_rows: int,
    bucket_base: str,
) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            print(f"  Warning: {path.name} not found, skipping...")
            continue
        print(f"  Reading {path.name}...")
        frames.append(load_taxi_pickups(path, max_rows=max_rows))
    if not frames:
        raise FileNotFoundError("No taxi data files were successfully loaded")

    trips = pd.concat(frames, ignore_index=True)
    if lookup_csv and lookup_csv.exists():
        trips = attach_zone_metadata(trips, lookup_csv).dropna(subset=["Zone"])
    else:
        if lookup_csv:
            print(
                f"  Warning: taxi zone lookup {lookup_csv} missing; "
                "falling back to raw Location IDs."
            )
        trips = trips.copy()
        trips["Zone"] = trips["PULocationID"].astype(int).astype(str)
    trips["event_time"] = trips["event_time"].dt.tz_convert(None)
    trips["hour"] = trips["event_time"].dt.hour
    trips["is_weekend"] = trips["event_time"].dt.dayofweek >= 5
    trips["is_rush"] = trips["hour"].apply(is_rush)
    trips["cohort"] = cohort_label(trips["is_weekend"], trips["is_rush"])
    trips["bucket_start"] = trips["event_time"].dt.floor(bucket_base)
    return trips


def load_taxi_dataset(
    *,
    paths: Sequence[Path] | None = None,
    lookup_csv: Path | None = None,
    max_rows: int = 4_000_000,
    bucket_base: str = DEFAULT_BUCKET_BASE,
    min_mean: float = 1.0,
    min_nonzero: float = 0.3,
) -> WaitDataset:
    """Load taxi trips and return a `WaitDataset` with caching."""
    if paths is None:
        paths = [
            DATA_RAW / f"yellow_tripdata_2024-{month:02d}.parquet"
            for month in range(1, 7)
        ]
    if lookup_csv is None:
        default_lookup = DATA_RAW / "taxi_zone_lookup.csv"
        if default_lookup.exists():
            lookup_csv = default_lookup
        else:
            print(
                "  Warning: taxi_zone_lookup.csv not found; "
                "locations will be labeled by numeric IDs."
            )

    print(f"Loading {len(paths)} taxi files...")
    trips = _prepare_taxi_trips(
        paths,
        lookup_csv,
        max_rows=max_rows,
        bucket_base=bucket_base,
    )
    locations = _screen_locations(
        trips,
        group_col="Zone",
        base_freq=bucket_base,
        min_mean=min_mean,
        min_nonzero=min_nonzero,
    )
    print(
        f"Loaded {len(trips):,} taxi trips across {len(locations)} active zones (Jan-Jun 2024)"
    )
    return WaitDataset(
        name="Taxi",
        trips=trips,
        group_col="Zone",
        locations=locations,
        min_mean=min_mean,
        min_nonzero=min_nonzero,
        default_freq=bucket_base,
    )


def load_bike_dataset(
    *,
    data_root: Path | None = None,
    glob: str = "2024*-citibike-tripdata_*.csv",
    max_rows: int = 5_000_000,
    bucket_base: str = DEFAULT_BUCKET_BASE,
    min_mean: float = 0.5,
    min_nonzero: float = 0.2,
) -> WaitDataset:
    """Load Citi Bike data and return a `WaitDataset` with caching."""
    if data_root is None:
        data_root = DATA_RAW / "citibike"
    files = sorted(data_root.glob(glob))
    if not files:
        raise FileNotFoundError(f"No Citi Bike CSVs matching {glob}")

    frames = []
    for idx, csv_path in enumerate(files):
        frames.append(
            pd.read_csv(
                csv_path,
                nrows=max_rows if (max_rows and idx == 0) else None,
                dtype={"start_station_id": str, "end_station_id": str},
                low_memory=False,
            )
        )
    bike = pd.concat(frames, ignore_index=True)
    bike = bike.dropna(subset=["start_station_id", "start_lat", "start_lng"])
    bike["start_station_id"] = bike["start_station_id"].astype(str)
    bike["event_time"] = pd.to_datetime(bike["started_at"])
    bike["hour"] = bike["event_time"].dt.hour
    bike["is_weekend"] = bike["event_time"].dt.dayofweek >= 5
    bike["is_rush"] = bike["hour"].apply(is_rush)
    bike["cohort"] = cohort_label(bike["is_weekend"], bike["is_rush"])
    bike["bucket_start"] = bike["event_time"].dt.floor(bucket_base)
    bike = bike.rename(columns={"start_station_id": "StationID"})

    locations = _screen_locations(
        bike,
        group_col="StationID",
        base_freq=bucket_base,
        min_mean=min_mean,
        min_nonzero=min_nonzero,
    )
    print(f"Loaded {len(bike):,} bike trips across {len(locations)} active stations")
    return WaitDataset(
        name="Citi Bike",
        trips=bike,
        group_col="StationID",
        locations=locations,
        min_mean=min_mean,
        min_nonzero=min_nonzero,
        default_freq=bucket_base,
    )


def build_snapshot_figure(
    dataset: WaitDataset,
    location: str,
    *,
    cohort: str = "All trips",
    freq: str = DEFAULT_BUCKET_BASE,
    title_prefix: str = "Snapshot",
) -> Tuple[go.Figure | None, SelectionMetrics | None, str | None]:
    """Return (figure, metrics, message) for a static wait-time snapshot."""
    metrics, message = dataset.compute_selection(location, cohort, freq)
    if message or metrics is None:
        return None, None, message
    title = f"{title_prefix}: {location} ({cohort}, {freq})"
    fig = create_combined_figure(metrics, title_prefix=title)
    return fig, metrics, None


def render_wait_snapshot(
    dataset: WaitDataset,
    location: str,
    *,
    cohort: str = "All trips",
    freq: str = DEFAULT_BUCKET_BASE,
    title_prefix: str = "Snapshot",
    image_kwargs: Dict[str, int] | None = None,
) -> None:
    """
    Display a static PNG snapshot of arrivals + exponential overlay.

    The figure is rasterized so GitHub can show a representative chart without
    running the ipywidgets.
    """
    image_kwargs = image_kwargs or {}
    fig, _, message = build_snapshot_figure(
        dataset, location, cohort=cohort, freq=freq, title_prefix=title_prefix
    )
    if fig is None:
        display(Markdown(message or "Snapshot unavailable."))
        return
    png = fig.to_image(format="png", width=900, height=450, scale=2, **image_kwargs)
    display(Image(png))


__all__ = [
    "WaitDataset",
    "WaitDashboard",
    "SelectionMetrics",
    "load_taxi_dataset",
    "load_bike_dataset",
    "create_combined_figure",
    "build_snapshot_figure",
    "render_wait_snapshot",
]
