# NYC Public Transit Travel/Wait-Time Modeling

This repo contains the notebooks, preprocessing scripts, and Streamlit dashboard used to compare NYC taxi vs Citi Bike performance using Jan–Jun 2024 data. The work has two major threads:

1. **Wait-time modeling** – Poisson / Negative-Binomial fits for arrivals plus exponential wait-time diagnostics (`notebooks/mode_diagnostics.ipynb`, `src/modeling/wait_dashboard.py`).
2. **Travel-time modeling** – Gamma-bin summaries and a continuous lognormal GLM fallback with an `is_ebike` knob for Citi Bike trips (`notebooks/mode_diagnostics.ipynb`, `scripts/build_travel_stats.py`, `src/modeling/travel_times.py`).

The quick-start below explains how to set up the data, run the notebooks, and launch the dashboard.

---

## 1. Requirements and environment setup

- Python 3.11+ (project tested on a conda env named `nyc`)
- Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `numpy`, `pandas`, `scipy`, `statsmodels`, `plotly`, `kaleido` (for embedding static PNGs), `streamlit`, `folium`.

Optional: install `nbclassic` / `jupyterlab` for running notebooks with `ipywidgets`.

---

## 2. Data layout and downloads

```
data/
├── raw/
│   ├── yellow_tripdata_2024-01.parquet … yellow_tripdata_2024-06.parquet
│   ├── citibike/
│   │   ├── 202401-citibike-tripdata_1.csv … (Jan–Jun 2024 Citi Bike CSVs)
│   ├── taxi_zone_lookup.csv, taxi_zone_centroids.csv, taxi_zones_shp/ …
├── derived/
│   ├── taxi_rates.parquet, taxi_centroids.parquet
│   ├── citibike_rates.parquet, citibike_stations.parquet
│   └── travel_stats/
│       ├── travel_bins.parquet
│       ├── travel_lognormal_glm.json
│       └── (legacy) travel_regression.json
```

### Taxi data
- Download monthly Yellow Taxi Parquet files from the NYC TLC repository: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Place the Jan–Jun 2024 files under `data/raw/` with the names shown above.

### Citi Bike data
- Download monthly Citi Bike trip CSVs from: https://s3.amazonaws.com/tripdata/index.html
- Extract them under `data/raw/citibike/`. The script/notebooks expect filenames like `202401-citibike-tripdata_1.csv`.

### Taxi zone metadata
- `taxi_zone_lookup.csv`, `taxi_zone_centroids.csv`, and `taxi_zones_shp/` come from the TLC GitHub data dictionary.

---

## 3. Precompute derived artifacts

Several files in `data/derived/` are required by the notebooks and the Streamlit app. They are produced by dedicated scripts/notebooks:

1. **Taxi/Citi Bike station summaries (wait model)** – follow the preprocessing steps defined in the wait-time notebooks or run `python scripts/build_wait_stats.py` (add `--taxi-paths data/raw/yellow_tripdata_2024-*.parquet` to span multiple months) to generate:
   - `data/derived/taxi_rates.parquet`
   - `data/derived/taxi_centroids.parquet`
   - `data/derived/citibike_rates.parquet`
   - `data/derived/citibike_stations.parquet`

2. **Travel-time stats (lognormal + Gamma)** – run the builder script:

```bash
python scripts/build_travel_stats.py \
  --bike-root data/raw/citibike \
  --bike-glob '202401-citibike-tripdata_*.csv' \
  --output-dir data/derived/travel_stats
```

This produces:
- `travel_bins.parquet` (Gamma fits per mode/distance bin/rush/weekend)
- `travel_lognormal_glm.json` (continuous GLM coefficients/metrics)

The Streamlit app now reads stats from `data/derived/travel_stats` by default.

---

## 4. Running the notebooks

Launch JupyterLab or Notebook from the repo root (ensure the conda env is activated and `ipywidgets` is installed).

### `notebooks/mode_diagnostics.ipynb`
- Unified notebook that embeds both wait-time and travel-time dashboards.
- Relies on the shared helpers in `src/modeling/wait_dashboard.py` and `src/modeling/travel_diagnostics.py`.
- Exports representative PNG snapshots (using `kaleido`) alongside ipywidgets so GitHub/nbviewer readers can see reference figures without executing the notebook.
- Includes a configuration cell (`WAIT_TAXI_MAX_ROWS`, etc.) so you can downsample raw data when running on smaller machines.
- Legacy `wait_times.ipynb` / `travel_times.ipynb` now just point to this combined notebook to avoid duplication.
- Does **not** require the derived Gamma/GLM caches; those are only needed for the Streamlit deployment path.
- For quick dataset stats (trip counts, average duration/distance, station/zone coverage), run `notebooks/data_overview.ipynb`.

**Tip:** Execute the “Static … snapshot” cells before committing so fresh PNGs are embedded for reviewers.

---

## 5. Deployment flow & Streamlit

1. **Notebook exploration** – run `notebooks/mode_diagnostics.ipynb` against raw data (downsample if needed); tune thresholds, study diagnostics, etc.
2. **Persist wait-time stats** – run `python scripts/build_wait_stats.py` to write `outputs/wait_stats/*` (Poisson/NB + wait summaries) for both taxi zones and Citi Bike stations.
3. **Persist travel-time stats** – run:

   ```bash
   python scripts/build_travel_stats.py \
     --bike-root data/raw/citibike \
     --bike-glob '20240*-citibike-tripdata_*.csv' \
     --output-dir data/derived/travel_stats
   ```

   This writes `travel_bins.parquet` and `travel_lognormal_glm.json` consumed by `src/modeling/travel_times.py`.

4. **Launch the dashboard**:

   ```bash
   streamlit run streamlit_app.py
   ```

   The app loads the cached wait/travel stats and compares taxi vs bike without touching the raw files. If a cache is missing, rerun the corresponding builder script.

---

## 6. Repository structure quick reference

```
scripts/build_wait_stats.py               # builds Poisson/wait caches for taxi + Citi Bike
scripts/build_travel_stats.py             # builds Gamma + lognormal stats
src/modeling/                             # helpers for Streamlit + modeling
notebooks/                                # analysis notebooks with embedded figures
streamlit_app.py                          # dashboard entry point
data/raw                                  # source datasets
data/derived                              # precomputed artifacts used in app/notebooks
```

---

## 7. Troubleshooting

- **`Path.cwd()` FileNotFoundError inside notebooks** – restart the kernel or set `PROJECT_ROOT` explicitly if you opened the notebook from a different directory.
- **Interactive widgets not showing figures on GitHub** – re-run the static snapshot cells; they export PNGs inline.
- **Streamlit travel-time errors** – ensure `data/derived/travel_stats/travel_lognormal_glm.json` exists (run the builder script).

---

With the raw data downloaded and the `build_wait_stats.py` + `build_travel_stats.py` scripts executed, you can explore the notebook and launch the Streamlit dashboard without additional setup. Happy modeling!
