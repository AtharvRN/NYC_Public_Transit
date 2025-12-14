# NYC Public Transit Travel/Wait-Time Modeling

This repo contains the notebooks, preprocessing scripts, and Streamlit dashboard used to compare NYC taxi vs Citi Bike performance using Jan–Jun 2024 data. The project is split into two components:

1. **Wait-time modeling (Component 1)** – Poisson / Negative-Binomial fits for arrivals plus exponential wait-time diagnostics (`notebooks/wait_times.ipynb`).
2. **Travel-time modeling (Component 2)** – Gamma-bin summaries and a continuous lognormal GLM fallback (`notebooks/travel_times.ipynb`, `scripts/component2_build_travel_stats.py`, `src/modeling/travel_times.py`).

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

1. **Taxi/Citi Bike station summaries (wait model)** – follow the preprocessing steps defined in Component 1 notebooks or custom scripts to generate:
   - `data/derived/taxi_rates.parquet`
   - `data/derived/taxi_centroids.parquet`
   - `data/derived/citibike_rates.parquet`
   - `data/derived/citibike_stations.parquet`

2. **Travel-time stats (lognormal + Gamma)** – run the builder script:

```bash
python scripts/component2_build_travel_stats.py \
  --taxi-path data/raw/yellow_tripdata_2024-01.parquet \
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

### `notebooks/wait_times.ipynb`
- Demonstrates why Poisson/Negative-Binomial arrivals and exponential wait times are reasonable.
- Requires `data/raw` taxi/citi data and derived wait-time rate files.
- Contains interactive widgets plus static PNGs (rendered via `kaleido`) so GitHub viewers can see the figures.

### `notebooks/travel_times.ipynb`
- Mirrors the wait-time analysis for travel-time fits: Gamma bins vs lognormal GLM, MAE/RMSE comparisons, log-likelihood tables, etc.
- Pulls helper functions directly from `scripts/component2_build_travel_stats.py`.
- Run the entire notebook to refresh metrics and embedded figures before committing.

**Tip:** Execute the new “Static … snapshot” cells to embed PNG outputs; otherwise GitHub will only show the ipywidget placeholders.

---

## 5. Running the Streamlit dashboard

1. Make sure derived artifacts exist (see section 3).
2. Start the app from the repo root:

```bash
streamlit run streamlit_app.py
```

The app:
- Loads wait-time caches (`taxi_rates.parquet`, etc.) from `data/derived/`.
- Uses `src/modeling/travel_times.py` to estimate travel minutes via the lognormal GLM (falls back to Gamma bins or constant speed if needed).
- Lets users pick origin/destination on the map and compares taxi vs bike travel + wait times.

If the app displays “Travel stats missing…”, rerun the builder script to regenerate `data/derived/travel_stats/*`.

---

## 6. Repository structure quick reference

```
scripts/component2_build_travel_stats.py   # builds Gamma + lognormal stats
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

With the raw data downloaded and `scripts/component2_build_travel_stats.py` executed, you can explore both notebooks and launch the Streamlit dashboard without additional setup. Happy modeling!
