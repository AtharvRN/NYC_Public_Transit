"""Travel-time diagnostic helpers for notebooks and scripts.

This module centralizes the logic used by `scripts/build_travel_stats.py`
plus the rich visualizations from `notebooks/travel_times.ipynb`.  By exposing
reusable loaders, aggregations, and ipywidget dashboards we can keep
`notebooks/mode_diagnostics.ipynb` lightweight while still sharing logic with
CLI tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from IPython.display import Image, Markdown, display
from scipy.special import gammaln
from scipy.stats import lognorm

try:  # ipywidgets is optional (Streamlit/tests do not need it)
    import ipywidgets as widgets
except ImportError:  # pragma: no cover - notebook dependency
    widgets = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"

BIN_WIDTH_KM = 2.0
MAX_DISTANCE_KM = 12.0
GAMMA_MIN_SAMPLES = 50
REGRESSION_MIN_SAMPLES = 200
RUSH_HOURS = ((7, 10), (16, 19))


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


def load_taxi_trips(path: Path, max_rows: Optional[int]) -> pd.DataFrame:
    cols = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
    ]
    df = pd.read_parquet(path, columns=cols)
    if max_rows:
        df = df.head(max_rows)
    df["pickup_dt"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["dropoff_dt"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    df["travel_min"] = (df["dropoff_dt"] - df["pickup_dt"]).dt.total_seconds() / 60.0
    df["distance_km"] = df["trip_distance"].astype(float) * 1.60934
    df = prep_common_features(df, event_col="pickup_dt")
    df["mode"] = "taxi"
    df["rideable_type"] = "taxi"
    df["is_ebike"] = False
    return df


def load_bike_trips(root: Path, glob_pattern: str, max_rows: Optional[int]) -> pd.DataFrame:
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
                    "rideable_type",
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
    df["rideable_type"] = (
        df["rideable_type"].fillna("classic_bike").astype(str).str.lower()
    )
    df["is_ebike"] = df["rideable_type"].eq("electric_bike")
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


def build_lognormal_design(df: pd.DataFrame) -> pd.DataFrame:
    design = pd.DataFrame(
        {
            "distance_km": df["distance_km"],
            "distance_sq": df["distance_km"] ** 2,
            "is_rush": df["is_rush"].astype(int),
            "is_weekend": df["is_weekend"].astype(int),
            "is_ebike": df.get("is_ebike", False).astype(int),
        }
    )
    return sm.add_constant(design, has_constant="add")


def _fit_lognormal_internal(
    df: pd.DataFrame,
    *,
    annotate: bool,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Optional[pd.DataFrame]]:
    models: Dict[str, Dict[str, float]] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    annotated = df.copy() if annotate else None
    if annotate:
        annotated["log_glm_pred_minutes"] = np.nan
        annotated["log_glm_log_mu"] = np.nan
        annotated["log_glm_residual"] = np.nan

    for mode in sorted(df["mode"].unique()):
        subset = df[df["mode"] == mode]
        if len(subset) < REGRESSION_MIN_SAMPLES:
            continue
        design = build_lognormal_design(subset)
        log_minutes = np.log(subset["travel_min"].values)
        model = sm.GLM(log_minutes, design, family=sm.families.Gaussian())
        result = model.fit()

        mu_hat = result.predict(design)
        sigma2 = float(result.scale)
        residuals = log_minutes - mu_hat

        models[mode] = {
            "sigma": float(np.sqrt(sigma2)),
            "coefficients": {key: float(val) for key, val in result.params.items()},
        }
        metrics[mode] = {
            "samples": int(len(subset)),
            "sigma": float(np.sqrt(sigma2)),
            "r_squared_log": float(
                1
                - np.sum((log_minutes - mu_hat) ** 2)
                / np.sum((log_minutes - log_minutes.mean()) ** 2)
            ),
            "mae_log": float(np.mean(np.abs(residuals))),
        }

        if annotate and annotated is not None:
            pred_minutes = np.exp(mu_hat + 0.5 * sigma2)
            annotated.loc[subset.index, "log_glm_pred_minutes"] = pred_minutes
            annotated.loc[subset.index, "log_glm_log_mu"] = mu_hat
            annotated.loc[subset.index, "log_glm_residual"] = subset["travel_min"].values - pred_minutes

    return models, metrics, annotated


def fit_lognormal_glm(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    models, metrics, _ = _fit_lognormal_internal(df, annotate=False)
    return models, metrics


def fit_lognormal_with_predictions(df: pd.DataFrame) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    pd.DataFrame,
]:
    models, metrics, annotated = _fit_lognormal_internal(df, annotate=True)
    if annotated is None:
        annotated = df.copy()
    return models, metrics, annotated


def load_travel_samples(
    taxi_paths: Sequence[Path],
    bike_root: Path,
    bike_glob: str,
    taxi_max_rows: Optional[int],
    bike_max_rows: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    taxi_frames = []
    for path in taxi_paths:
        if not path.exists():
            print(f"[taxi] Missing {path.name}, skipping")
            continue
        print(f"[taxi] Loading {path.name} ...")
        taxi_frames.append(load_taxi_trips(path, taxi_max_rows))
    if not taxi_frames:
        raise FileNotFoundError("No taxi files found. Check TAXI_PATHS configuration.")
    taxi = pd.concat(taxi_frames, ignore_index=True)

    print("Loading Citi Bike trips ...")
    bike = load_bike_trips(bike_root, bike_glob, bike_max_rows)

    combined = pd.concat([taxi, bike], ignore_index=True)
    if "is_ebike" not in combined.columns:
        combined["is_ebike"] = False
    else:
        combined["is_ebike"] = combined["is_ebike"].fillna(False).astype(bool)
    combined["distance_bin_label"] = combined["distance_bin"].astype(str)
    combined.loc[combined["distance_bin"].isna(), "distance_bin_label"] = np.nan
    return taxi, bike, combined


def _parse_distance_start(label: str) -> float:
    try:
        return float(label.replace("km", "").split("-")[0])
    except Exception:
        return math.inf


@dataclass
class TravelDiagnostics:
    taxi: pd.DataFrame
    bike: pd.DataFrame
    combined: pd.DataFrame
    bin_stats: pd.DataFrame

    def __post_init__(self) -> None:
        self.bin_stats = self.bin_stats.copy()
        self.bin_stats["distance_bin"] = self.bin_stats["distance_bin"].astype(str)
        self.bin_stats["meets_gamma"] = (
            self.bin_stats["sample_count"] >= GAMMA_MIN_SAMPLES
        )
        self.bin_stats["distance_start"] = self.bin_stats["distance_bin"].apply(
            _parse_distance_start
        )
        self.distance_levels = sorted(
            [
                label
                for label in self.bin_stats["distance_bin"].unique()
                if isinstance(label, str) and label != "nan"
            ],
            key=_parse_distance_start,
        )
        self.combined = self.combined.copy()
        self.combined["distance_bin_label"] = self.combined["distance_bin"].astype(str)
        (
            models,
            metrics,
            annotated,
        ) = fit_lognormal_with_predictions(self.combined)
        self.lognormal_models = models
        self.lognormal_metrics = metrics
        self.lognormal_coeffs = {
            mode: payload.get("coefficients", {}) for mode, payload in models.items()
        }
        self.lognormal_scales = {
            mode: float(payload.get("sigma", float("nan"))) for mode, payload in models.items()
        }
        self.lognormal_summary_df = pd.DataFrame(
            [
                {
                    "mode": mode,
                    "samples": stats.get("samples", 0),
                    "sigma_log": stats.get("sigma", float("nan")),
                    "r_squared_log": stats.get("r_squared_log", float("nan")),
                    "mae_log": stats.get("mae_log", float("nan")),
                }
                for mode, stats in metrics.items()
            ]
        )
        self.combined = annotated

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------
    def bin_stats_summary(self) -> pd.DataFrame:
        summary = (
            self.bin_stats.groupby(["mode", "distance_bin", "distance_start"])
            .agg(
                cohorts=("sample_count", "size"),
                min_samples=("sample_count", "min"),
                median_samples=("sample_count", "median"),
                max_samples=("sample_count", "max"),
            )
            .reset_index()
            .sort_values(["mode", "distance_start"])
        )
        return summary.drop(columns="distance_start")

    def low_sample_bins(self, n: int = 10) -> pd.DataFrame:
        cols = [
            "mode",
            "distance_bin",
            "is_rush",
            "is_weekend",
            "sample_count",
            "meets_gamma",
        ]
        return self.bin_stats.sort_values("sample_count")[cols].head(n)

    def gamma_coverage(self) -> pd.DataFrame:
        return (
            self.bin_stats.groupby("mode")["meets_gamma"].mean().rename("fraction_gamma_ready").to_frame()
        )

    def sample_counts_table(self) -> pd.DataFrame:
        df = self.bin_stats.copy()
        df["rush_label"] = np.where(df["is_rush"], "Rush", "Off-peak")
        df["weekend_label"] = np.where(df["is_weekend"], "Weekend", "Weekday")
        cols = [
            "mode",
            "distance_bin",
            "rush_label",
            "weekend_label",
            "sample_count",
            "meets_gamma",
        ]
        return df.sort_values(["mode", "distance_start", "rush_label", "weekend_label"])[cols]

    # ------------------------------------------------------------------
    # Gamma diagnostics
    # ------------------------------------------------------------------
    def _subset(self, mode: str, distance_bin: str, is_rush: bool, is_weekend: bool) -> pd.DataFrame:
        return self.combined[
            (self.combined["mode"] == mode)
            & (self.combined["distance_bin_label"] == distance_bin)
            & (self.combined["is_rush"] == is_rush)
            & (self.combined["is_weekend"] == is_weekend)
        ]

    def _bin_row(self, mode: str, distance_bin: str, is_rush: bool, is_weekend: bool) -> pd.Series | None:
        mask = (
            (self.bin_stats["mode"] == mode)
            & (self.bin_stats["distance_bin"] == distance_bin)
            & (self.bin_stats["is_rush"] == is_rush)
            & (self.bin_stats["is_weekend"] == is_weekend)
        )
        subset = self.bin_stats.loc[mask]
        if subset.empty:
            return None
        return subset.iloc[0]

    def gamma_figure(
        self,
        mode: str,
        distance_bin: str,
        is_rush: bool,
        is_weekend: bool,
    ) -> Tuple[go.Figure, Dict[str, float]]:
        subset = self._subset(mode, distance_bin, is_rush, is_weekend)
        row = self._bin_row(mode, distance_bin, is_rush, is_weekend)
        if row is None or subset.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No samples for this combination",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(height=400)
            return fig, {}

        bin_width = 1.0
        max_minutes = max(15, subset["travel_min"].max())
        bins = np.arange(0, max_minutes + bin_width, bin_width)
        counts, edges = np.histogram(subset["travel_min"], bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])

        fig = go.Figure()
        fig.add_bar(
            x=centers,
            y=counts,
            name="Empirical histogram",
            marker_color="#1f77b4",
            opacity=0.75,
        )

        k_val, theta_val = row["gamma_k"], row["gamma_theta"]
        if (
            np.isfinite(k_val)
            and np.isfinite(theta_val)
            and row["sample_count"] >= GAMMA_MIN_SAMPLES
        ):
            x_grid = np.linspace(0, max_minutes, 250)
            pdf_scaled = (
                (x_grid ** (k_val - 1) * np.exp(-x_grid / theta_val))
                / (math.gamma(k_val) * (theta_val ** k_val))
            ) * row["sample_count"] * bin_width
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=pdf_scaled,
                    name=f"Gamma fit (k={k_val:.2f}, θ={theta_val:.2f})",
                    line=dict(color="#d62728", width=3),
                )
            )
        else:
            fig.add_annotation(
                text="Gamma parameters suppressed (<50 samples)",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.9,
                showarrow=False,
                font=dict(color="#d62728"),
            )

        fig.update_layout(
            title=f"{mode.title()} | {distance_bin} | {'Rush' if is_rush else 'Off-peak'} | {'Weekend' if is_weekend else 'Weekday'}",
            xaxis_title="Travel minutes",
            yaxis_title="Trip count",
            bargap=0.02,
            height=500,
        )
        metrics = {
            "sample_count": int(row["sample_count"]),
            "mean_minutes": float(row["mean_min"]),
            "var_minutes": float(row["var_min"]),
            "gamma_k": float(k_val) if np.isfinite(k_val) else np.nan,
            "gamma_theta": float(theta_val) if np.isfinite(theta_val) else np.nan,
            "overall_mean": float(subset["travel_min"].mean()),
            "overall_var": float(subset["travel_min"].var(ddof=0)),
        }
        return fig, metrics

    def build_gamma_dashboard(self) -> widgets.VBox:
        if widgets is None:
            raise ImportError("ipywidgets is required for interactive dashboards.")
        mode_widget = widgets.ToggleButtons(options=sorted(self.combined["mode"].unique()), description="Mode")
        distance_widget = widgets.Dropdown(options=self.distance_levels, description="Distance bin")
        rush_widget = widgets.ToggleButtons(options=[("Off-peak", False), ("Rush", True)], description="Rush?")
        weekend_widget = widgets.ToggleButtons(options=[("Weekday", False), ("Weekend", True)], description="Weekend?")
        output = widgets.Output()

        def refresh(*_):
            with output:
                output.clear_output(wait=True)
                fig, metrics = self.gamma_figure(
                    mode_widget.value,
                    distance_widget.value,
                    rush_widget.value,
                    weekend_widget.value,
                )
                display(fig)
                if metrics:
                    display(pd.Series(metrics, name="metrics"))

        for widget in [mode_widget, distance_widget, rush_widget, weekend_widget]:
            widget.observe(refresh, names="value")

        refresh()
        return widgets.VBox([
            widgets.HBox([mode_widget, distance_widget, rush_widget, weekend_widget]),
            output,
        ])

    def render_gamma_snapshot(
        self,
        *,
        mode: str,
        distance_bin: Optional[str] = None,
        is_rush: bool = False,
        is_weekend: bool = False,
        title_prefix: str = "Gamma snapshot",
    ) -> None:
        if distance_bin is None:
            if not self.distance_levels:
                display(Markdown("No distance bins available."))
                return
            distance_bin = self.distance_levels[0]
        fig, metrics = self.gamma_figure(mode, distance_bin, is_rush, is_weekend)
        png = fig.to_image(format="png", width=900, height=500, scale=2)
        display(
            Markdown(
                f"**{title_prefix} — {mode.title()}, {distance_bin}, "
                f"{'Rush' if is_rush else 'Off-peak'}, {'Weekend' if is_weekend else 'Weekday'}**"
            )
        )
        display(Image(png))
        if metrics:
            display(pd.Series(metrics, name="metrics"))

    # ------------------------------------------------------------------
    # Lognormal overlays
    # ------------------------------------------------------------------
    def lognormal_overlay_figure(
        self,
        mode: str,
        distance_bin: str,
        is_rush: bool,
        is_weekend: bool,
        bike_type: str = "all",
    ) -> Tuple[go.Figure, Optional[Dict[str, float]]]:
        subset = self._subset(mode, distance_bin, is_rush, is_weekend).copy()
        if mode == "bike" and bike_type in {"classic", "ebike"}:
            mask = subset["is_ebike"]
            subset = subset.loc[mask] if bike_type == "ebike" else subset.loc[~mask]
        row = self._bin_row(mode, distance_bin, is_rush, is_weekend)
        if subset.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No samples for this combination",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(height=400)
            return fig, None
        log_vals = np.log(subset["travel_min"])
        mu = log_vals.mean()
        sigma = log_vals.std(ddof=0)
        bin_width = 1.0
        max_minutes = max(15, subset["travel_min"].max())
        bins = np.arange(0, max_minutes + bin_width, bin_width)
        counts, edges = np.histogram(subset["travel_min"], bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])

        fig = go.Figure()
        fig.add_bar(
            x=centers,
            y=counts,
            name="Empirical histogram",
            marker_color="#4c78a8",
            opacity=0.75,
        )
        x_grid = np.linspace(0.01, max_minutes, 400)

        if row is not None and row["sample_count"] >= GAMMA_MIN_SAMPLES:
            if np.isfinite(row["gamma_k"]) and np.isfinite(row["gamma_theta"]):
                gamma_scaled = (
                    (x_grid ** (row["gamma_k"] - 1) * np.exp(-x_grid / row["gamma_theta"]))
                    / (math.gamma(row["gamma_k"]) * (row["gamma_theta"] ** row["gamma_k"]))
                ) * len(subset) * bin_width
                fig.add_trace(
                    go.Scatter(
                        x=x_grid,
                        y=gamma_scaled,
                        name=f"Gamma fit (k={row['gamma_k']:.2f}, θ={row['gamma_theta']:.2f})",
                        line=dict(color="#ff7f0e", width=2),
                    )
                )

        pdf_scaled = lognorm.pdf(x_grid, s=sigma, scale=np.exp(mu)) * len(subset) * bin_width
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=pdf_scaled,
                name=f"Lognormal fit (μ={mu:.2f}, σ={sigma:.2f})",
                line=dict(color="#d62728", width=3),
            )
        )
        fig.update_layout(
            title=f"{mode.title()} | {distance_bin} | {'Rush' if is_rush else 'Off-peak'} | {'Weekend' if is_weekend else 'Weekday'}",
            xaxis_title="Travel minutes",
            yaxis_title="Trip count",
            bargap=0.02,
            height=500,
        )
        stats = {
            "mu_log": mu,
            "sigma_log": sigma,
            "samples": len(subset),
            "gamma_samples": int(row["sample_count"]) if row is not None else 0,
            "gamma_mean": float(row["mean_min"]) if row is not None else float("nan"),
            "bike_type": bike_type,
        }
        return fig, stats

    def build_lognormal_overlay_dashboard(self) -> widgets.VBox:
        if widgets is None:
            raise ImportError("ipywidgets is required for interactive dashboards.")
        mode_widget = widgets.ToggleButtons(options=sorted(self.combined["mode"].unique()), description="Mode")
        distance_widget = widgets.Dropdown(options=self.distance_levels, description="Distance bin")
        rush_widget = widgets.ToggleButtons(options=[("Off-peak", False), ("Rush", True)], description="Rush?")
        weekend_widget = widgets.ToggleButtons(options=[("Weekday", False), ("Weekend", True)], description="Weekend?")
        bike_type_widget = widgets.ToggleButtons(
            options=[("All bikes", "all"), ("Classic", "classic"), ("E-bike", "ebike")],
            description="Bike type",
        )
        output = widgets.Output()

        def refresh(*_):
            with output:
                output.clear_output(wait=True)
                fig, stats = self.lognormal_overlay_figure(
                    mode_widget.value,
                    distance_widget.value,
                    rush_widget.value,
                    weekend_widget.value,
                    bike_type=bike_type_widget.value,
                )
                display(fig)
                if stats:
                    display(pd.Series(stats, name="distribution_params"))
            bike_type_widget.disabled = mode_widget.value != "bike"

        for widget in [mode_widget, distance_widget, rush_widget, weekend_widget, bike_type_widget]:
            widget.observe(refresh, names="value")

        refresh()
        return widgets.VBox([
            widgets.HBox([mode_widget, distance_widget, rush_widget, weekend_widget, bike_type_widget]),
            output,
        ])

    def render_lognormal_snapshot(
        self,
        *,
        mode: str,
        distance_bin: Optional[str] = None,
        is_rush: bool = False,
        is_weekend: bool = False,
        title_prefix: str = "Gamma vs lognormal snapshot",
        bike_type: str = "all",
    ) -> None:
        if distance_bin is None:
            if not self.distance_levels:
                display(Markdown("No distance bins available."))
                return
            distance_bin = self.distance_levels[0]
        fig, stats = self.lognormal_overlay_figure(
            mode, distance_bin, is_rush, is_weekend, bike_type=bike_type
        )
        png = fig.to_image(format="png", width=900, height=500, scale=2)
        display(
            Markdown(
                f"**{title_prefix} — {mode.title()}, {distance_bin}, "
                f"{'Rush' if is_rush else 'Off-peak'}, {'Weekend' if is_weekend else 'Weekday'}**"
            )
        )
        display(Image(png))
        if stats:
            display(pd.Series(stats, name="distribution_params"))

    # ------------------------------------------------------------------
    # Lognormal GLM diagnostics
    # ------------------------------------------------------------------
    def lognormal_summary(self) -> pd.DataFrame:
        return self.lognormal_summary_df.copy()

    def lognormal_coeff_table(self) -> pd.DataFrame:
        table = pd.DataFrame(self.lognormal_coeffs).T
        table["sigma_log"] = pd.Series(self.lognormal_scales)
        return table

    def build_lognormal_diag_dashboard(self) -> widgets.VBox:
        if widgets is None:
            raise ImportError("ipywidgets is required for interactive dashboards.")
        modes = sorted(self.lognormal_coeffs.keys())
        if not modes:
            raise ValueError("No lognormal models fitted.")
        mode_widget = widgets.ToggleButtons(options=modes, description="Mode")
        bike_type_widget = widgets.ToggleButtons(
            options=[("All bikes", "all"), ("Classic", "classic"), ("E-bike", "ebike")],
            description="Bike type",
        )
        output = widgets.Output()

        def refresh(*_):
            with output:
                output.clear_output(wait=True)
                self._render_lognormal_diag(
                    mode_widget.value,
                    bike_type=bike_type_widget.value,
                )
            bike_type_widget.disabled = mode_widget.value != "bike"

        for widget in [mode_widget, bike_type_widget]:
            widget.observe(refresh, names="value")
        refresh()
        return widgets.VBox([widgets.HBox([mode_widget, bike_type_widget]), output])

    def _render_lognormal_diag(self, mode: str, bike_type: str = "all") -> None:
        subset = self.combined[
            (self.combined["mode"] == mode) & self.combined["log_glm_pred_minutes"].notna()
        ].copy()
        if mode == "bike" and bike_type in {"classic", "ebike"}:
            mask = subset["is_ebike"]
            subset = subset.loc[mask] if bike_type == "ebike" else subset.loc[~mask]
        if subset.empty:
            display(Markdown(f"No fitted samples for {mode}."))
            return
        subset["rush_label"] = subset["is_rush"].map({True: "Rush", False: "Off-peak"})
        subset["weekend_label"] = subset["is_weekend"].map({True: "Weekend", False: "Weekday"})
        scatter_sample = (
            subset.sample(n=min(len(subset), 6000), random_state=42)
            if len(subset) > 6000
            else subset
        )

        fig_scatter = px.scatter(
            scatter_sample,
            x="log_glm_pred_minutes",
            y="travel_min",
            color="rush_label",
            opacity=0.6,
            title=f"Lognormal GLM predictions — {mode.title()}",
            labels={
                "log_glm_pred_minutes": "Predicted minutes",
                "travel_min": "Observed minutes",
            },
        )
        max_val = max(subset["travel_min"].max(), subset["log_glm_pred_minutes"].max())
        fig_scatter.add_trace(
            go.Scatter(x=[0, max_val], y=[0, max_val], name="Ideal", line=dict(color="black", dash="dash"))
        )

        fig_residual = px.histogram(
            subset,
            x="log_glm_residual",
            color="weekend_label",
            nbins=60,
            opacity=0.7,
            title=f"Residuals (observed - predicted) — {mode.title()}",
            labels={"log_glm_residual": "Residual minutes", "weekend_label": "Weekend flag"},
        )

        coef_df = pd.Series(self.lognormal_coeffs[mode]).to_frame(name="coefficient")
        coef_df.loc["sigma_log"] = self.lognormal_scales.get(mode, float("nan"))

        type_label = {
            "all": "All bikes" if mode == "bike" else mode.title(),
            "classic": "Classic bikes",
            "ebike": "E-bikes",
        }.get(bike_type, "All samples")
        display(
            Markdown(
                f"**Samples ({type_label}):** {len(subset):,} | "
                f"σ_log={self.lognormal_scales.get(mode, float('nan')):.3f}"
            )
        )
        display(coef_df)
        display(fig_scatter)
        display(fig_residual)

    # ------------------------------------------------------------------
    # Likelihood / error metrics
    # ------------------------------------------------------------------
    def _gamma_logpdf(self, x: np.ndarray, k: np.ndarray, theta: np.ndarray) -> np.ndarray:
        valid = (x > 0) & np.isfinite(k) & np.isfinite(theta) & (k > 0) & (theta > 0)
        out = np.full_like(x, np.nan, dtype=float)
        out[valid] = (
            (k[valid] - 1) * np.log(x[valid])
            - x[valid] / theta[valid]
            - k[valid] * np.log(theta[valid])
            - gammaln(k[valid])
        )
        return out

    def _lognormal_logpdf(self, x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        valid = (x > 0) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
        out = np.full_like(x, np.nan, dtype=float)
        diff = np.log(x[valid]) - mu[valid]
        out[valid] = -0.5 * (diff ** 2) / (sigma[valid] ** 2) - np.log(
            x[valid] * sigma[valid] * np.sqrt(2 * np.pi)
        )
        return out

    def log_likelihood_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, float, int]:
        gamma_lookup = self.bin_stats[
            ["mode", "distance_bin", "is_rush", "is_weekend", "gamma_k", "gamma_theta", "sample_count"]
        ].copy()
        gamma_lookup["distance_bin"] = gamma_lookup["distance_bin"].astype(str)
        merged = self.combined.merge(
            gamma_lookup,
            how="left",
            left_on=["mode", "distance_bin_label", "is_rush", "is_weekend"],
            right_on=["mode", "distance_bin", "is_rush", "is_weekend"],
            suffixes=("", "_bin"),
        )
        merged["log_glm_sigma"] = merged["mode"].map(self.lognormal_scales)

        valid_mask = (
            merged["gamma_k"].notna()
            & merged["gamma_theta"].notna()
            & merged["sample_count"].ge(GAMMA_MIN_SAMPLES)
            & merged["log_glm_log_mu"].notna()
            & merged["log_glm_sigma"].notna()
        )
        compare = merged.loc[valid_mask].copy()
        compare["gamma_loglike"] = self._gamma_logpdf(
            compare["travel_min"].values,
            compare["gamma_k"].values,
            compare["gamma_theta"].values,
        )
        compare["lognormal_loglike"] = self._lognormal_logpdf(
            compare["travel_min"].values,
            compare["log_glm_log_mu"].values,
            compare["log_glm_sigma"].values,
        )
        ll_summary = (
            compare.groupby("mode")[["gamma_loglike", "lognormal_loglike"]]
            .sum()
            .assign(delta=lambda df: df["lognormal_loglike"] - df["gamma_loglike"])
        )
        per_trip = (
            compare.groupby("mode")[["gamma_loglike", "lognormal_loglike"]]
            .mean()
            .rename(columns={"gamma_loglike": "gamma_mean", "lognormal_loglike": "lognormal_mean"})
        )
        overall_delta = compare["lognormal_loglike"].sum() - compare["gamma_loglike"].sum()
        return ll_summary, per_trip, float(overall_delta), len(compare)

    def error_metrics_table(self) -> pd.DataFrame:
        gamma_lookup = self.bin_stats[
            ["mode", "distance_bin", "is_rush", "is_weekend", "mean_min", "sample_count"]
        ].copy()
        gamma_lookup["distance_bin"] = gamma_lookup["distance_bin"].astype(str)
        metrics_df = self.combined.merge(
            gamma_lookup,
            how="left",
            left_on=["mode", "distance_bin_label", "is_rush", "is_weekend"],
            right_on=["mode", "distance_bin", "is_rush", "is_weekend"],
            suffixes=("", "_gamma"),
        )
        metrics_df["gamma_pred"] = np.where(
            (metrics_df["sample_count"].ge(GAMMA_MIN_SAMPLES)) & metrics_df["mean_min"].notna(),
            metrics_df["mean_min"],
            np.nan,
        )
        metrics_df["lognormal_pred"] = metrics_df["log_glm_pred_minutes"]

        def summarize(residuals: pd.Series) -> Tuple[float, float]:
            mae = float(np.mean(np.abs(residuals)))
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            return mae, rmse

        records = []
        for mode in sorted(metrics_df["mode"].unique()):
            mode_df = metrics_df[metrics_df["mode"] == mode]
            log_df = mode_df.dropna(subset=["lognormal_pred"])
            gamma_df = mode_df.dropna(subset=["gamma_pred"])
            if not log_df.empty:
                resid = log_df["travel_min"] - log_df["lognormal_pred"]
                mae, rmse = summarize(resid)
                records.append(
                    {
                        "mode": mode,
                        "model": "lognormal_glm",
                        "samples": len(log_df),
                        "mae_min": mae,
                        "rmse_min": rmse,
                    }
                )
            if not gamma_df.empty:
                resid = gamma_df["travel_min"] - gamma_df["gamma_pred"]
                mae, rmse = summarize(resid)
                records.append(
                    {
                        "mode": mode,
                        "model": "gamma_bin",
                        "samples": len(gamma_df),
                        "mae_min": mae,
                        "rmse_min": rmse,
                    }
                )
        return pd.DataFrame(records)


def load_travel_diagnostics(
    *,
    taxi_paths: Optional[Sequence[Path]] = None,
    bike_root: Optional[Path] = None,
    bike_glob: str = "20240*-citibike-tripdata_*.csv",
    taxi_max_rows: Optional[int] = 4_000_000,
    bike_max_rows: Optional[int] = 2_000_000,
) -> TravelDiagnostics:
    if taxi_paths is None:
        taxi_paths = [DATA_RAW / f"yellow_tripdata_2024-{month:02d}.parquet" for month in range(1, 7)]
    if bike_root is None:
        bike_root = DATA_RAW / "citibike"
    taxi, bike, combined = load_travel_samples(
        taxi_paths=taxi_paths,
        bike_root=bike_root,
        bike_glob=bike_glob,
        taxi_max_rows=taxi_max_rows,
        bike_max_rows=bike_max_rows,
    )
    bin_stats = summarize_bins(combined)
    return TravelDiagnostics(taxi=taxi, bike=bike, combined=combined, bin_stats=bin_stats)


__all__ = [
    "BIN_WIDTH_KM",
    "MAX_DISTANCE_KM",
    "GAMMA_MIN_SAMPLES",
    "REGRESSION_MIN_SAMPLES",
    "RUSH_HOURS",
    "load_taxi_trips",
    "load_bike_trips",
    "summarize_bins",
    "fit_lognormal_glm",
    "load_travel_diagnostics",
    "TravelDiagnostics",
]
