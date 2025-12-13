#!/usr/bin/env python3
"""Per-station Poisson diagnostics for Citi Bike trips."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly import io as pio
from scipy.stats import chisquare
from scipy.stats import nbinom as sp_nbinom
from scipy.stats import poisson as sp_poisson


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Poisson vs NB fit for Citi Bike station pickup counts."
    )
    parser.add_argument(
        "--csvs",
        type=Path,
        nargs="+",
        required=True,
        help="List of Citi Bike trip CSV files (use shell glob).",
    )
    parser.add_argument("--freq", default="15min", help="Bucket size, e.g., 15min or 1H.")
    parser.add_argument(
        "--station-limit",
        type=int,
        default=20,
        help="Number of busiest stations to summarize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/citibike_poisson"),
        help="Directory for summary CSV + histograms.",
    )
    return parser.parse_args()


def load_trips(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    trips = pd.concat(frames, ignore_index=True)
    trips["started_at"] = pd.to_datetime(trips["started_at"], utc=False)
    trips["ended_at"] = pd.to_datetime(trips["ended_at"], utc=False)
    trips["trip_minutes"] = (trips["ended_at"] - trips["started_at"]).dt.total_seconds() / 60
    mask = (trips["trip_minutes"] > 0) & (trips["trip_minutes"] < 240)
    trips = trips[mask & trips["start_station_id"].notna()].copy()
    trips["start_station_id"] = trips["start_station_id"].astype(str)
    return trips


def bucket_counts(trips: pd.DataFrame, freq: str) -> pd.DataFrame:
    grouped = (
        trips.groupby([
            trips["start_station_id"],
            trips["started_at"].dt.floor(freq),
        ])
        .size()
        .rename("rides")
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index="started_at", columns="start_station_id", values="rides", fill_value=0
    )
    return pivot


def poisson_summary(series: pd.Series) -> Dict[str, float]:
    lam = series.mean()
    variance = series.var(ddof=0)
    dispersion = variance / lam if lam > 0 else np.nan
    value_counts = series.value_counts()
    xs = value_counts.index.values
    obs = value_counts.values
    expected = sp_poisson.pmf(xs, lam) * len(series)
    mask = expected > 5
    if mask.any():
        obs_masked = obs[mask]
        exp_masked = expected[mask]
        scale = obs_masked.sum() / exp_masked.sum()
        exp_masked *= scale
        chi2_stat, chi2_p = chisquare(obs_masked, exp_masked)
    else:
        chi2_stat = np.nan
        chi2_p = np.nan
    return {
        "mean": float(lam),
        "variance": float(variance),
        "dispersion_index": float(dispersion),
        "chi2_stat": float(chi2_stat),
        "chi2_pvalue": float(chi2_p),
    }


def estimate_nb_params(series: pd.Series) -> Tuple[float, float]:
    mean = series.mean()
    var = series.var(ddof=0)
    if var <= mean or mean <= 0:
        return np.nan, np.nan
    r = mean ** 2 / (var - mean)
    p = r / (r + mean)
    return float(r), float(p)


def make_histogram(
    series: pd.Series, station: str, output_dir: Path, freq: str, nb_params: Tuple[float, float]
) -> None:
    lam = series.mean()
    grid = np.arange(0, max(series.max(), int(series.quantile(0.99)) + 5) + 1)
    observed = series.value_counts().reindex(grid, fill_value=0).values
    fig = go.Figure()
    fig.add_bar(x=grid, y=observed, name="Observed", marker=dict(color="#4B6BFB"), opacity=0.75)
    poisson_expected = sp_poisson.pmf(grid, lam) * len(series)
    fig.add_scatter(
        x=grid, y=poisson_expected, mode="lines", name="Poisson", line=dict(color="#FFA500", width=2)
    )
    nb_r, nb_p = nb_params
    if np.isfinite(nb_r) and np.isfinite(nb_p) and nb_r > 0 and 0 < nb_p < 1:
        nb_expected = sp_nbinom.pmf(grid, nb_r, nb_p) * len(series)
        fig.add_scatter(
            x=grid,
            y=nb_expected,
            mode="lines",
            name="Neg-Bin",
            line=dict(color="#D62728", width=3),
        )
    fig.update_layout(
        title=f"Station {station} ({freq})",
        xaxis_title="Arrivals per bucket",
        yaxis_title="Frequency",
        template="plotly_white",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base = station.replace("/", "_").replace(" ", "_")
    html = output_dir / f"{base}.html"
    pdf = output_dir / f"{base}.pdf"
    fig.write_html(html)
    try:
        pio.write_image(fig, pdf)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not export histogram for {station}: {exc}")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trips = load_trips(args.csvs)
    counts = bucket_counts(trips, args.freq)
    station_totals = counts.sum().sort_values(ascending=False)
    top = station_totals.head(args.station_limit).index

    summaries = {}
    hist_dir = args.output / "histograms"
    for station in top:
        series = counts[station]
        stats = poisson_summary(series)
        nb_r, nb_p = estimate_nb_params(series)
        stats["nb_r"] = nb_r
        stats["nb_p"] = nb_p
        summaries[station] = stats
        make_histogram(series, station, hist_dir, args.freq, (nb_r, nb_p))

    out_csv = args.output / "citibike_poisson_summary.csv"
    pd.DataFrame.from_dict(summaries, orient="index").to_csv(out_csv)
    print(f"Wrote station summaries to {out_csv}")
    print(f"Histograms saved under {hist_dir}")


if __name__ == "__main__":
    main()
