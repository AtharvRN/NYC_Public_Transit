# NYC Public Transit — Wait & Travel Time Modeling

This project compares NYC Yellow Taxi and Citi Bike performance during Jan–Jun 2024.  We derive per-location wait estimates and end-to-end travel times, cache the parameters, and surface everything in a Streamlit dashboard plus a LaTeX report (`docs/report.tex`).  The modeling pipeline:

- **Waits:** For every taxi zone and Citi Bike station we compute hourly arrival rates split by weekday/weekend and rush/non-rush.  Inter-arrival gaps follow an exponential curve, so we store one $\lambda$ per slice and expose the mean wait ($60 / \lambda$ minutes) in the app.
- **Travel times:** Gamma cohorts (mode × 2 km distance bin × rush flag × weekend flag) are kept for diagnostics, while the deployed predictions come from a lognormal regression trained per mode:
  \[
  \log T = \beta_0 + \beta_1 d + \beta_2 d^2 + \beta_3 I_{\text{rush}} + \beta_4 I_{\text{weekend}} + \varepsilon.
  \]
  Bikes add an e-bike indicator.  Predictions apply $\exp(\hat{\mu} + 0.5 \hat{\sigma}^2)$ to recover minutes.

All derived parameters live under `data/derived/` and `outputs/`, so the Streamlit app never touches the raw TLC / Citi Bike files once caches are built.

---

## 1. Environment & dependencies

1. Python 3.11+ (the repo was tested with a conda env called `nyc`).
2. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
   Key libs: `numpy`, `pandas`, `pyarrow`, `statsmodels`, `scipy`, `plotly`, `kaleido`, `streamlit`, `folium`.
3. Optional: `jupyterlab`/`nbclassic` for notebooks with `ipywidgets`.

---

## 2. Data sources

```
data/
├── raw/
│   ├── yellow_tripdata_2024-01.parquet … yellow_tripdata_2024-06.parquet
│   ├── citibike/
│   │   ├── 202401-citibike-tripdata_1.csv … 202406-citibike-tripdata_1.csv
│   ├── taxi_zone_lookup.csv
│   ├── taxi_zone_centroids.csv
│   └── taxi_zones_shp/…
└── derived/  (filled by the scripts in §3)
```

- **Yellow Taxi trips:** TLC monthly parquet files — <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
- **Citi Bike trips:** Monthly CSVs — <https://s3.amazonaws.com/tripdata/index.html>
- **Zone metadata:** TLC data dictionary (lookup table, centroids, shapefile)

Drop the Jan–Jun 2024 files into `data/raw/` as shown above.

---

## 3. Build the caches

The Streamlit app requires summarized artifacts.  Run the following after downloading the raw data.

### 3.1 Wait-time summaries

```
python scripts/build_wait_stats.py \
  --cache-dir outputs/wait_stats \
  --taxi-paths data/raw/yellow_tripdata_2024-0{1,2,3,4,5,6}.parquet
```

Arguments are optional—by default the script scans `data/raw/yellow_tripdata_2024-*.parquet` and `data/raw/citibike/2024*-citibike-tripdata_*.csv`.  Outputs (default `outputs/wait_stats/`):
- `taxi_hourly_waits.parquet` (`.json`) – per-zone hourly $\lambda_{z,h}$ with weekday/weekend & rush labels
- `bike_hourly_waits.parquet` (`.json`)
- `citibike_stations.parquet`
- companion JSON exports for the Streamlit app

### 3.2 Travel-time stats

```
python scripts/build_travel_stats.py \
  --bike-root data/raw/citibike \
  --bike-glob '20240*-citibike-tripdata_*.csv' \
  --output-dir data/derived/travel_stats
```

Produces:
- `travel_bins.parquet` – Gamma cohorts (diagnostics only)
- `travel_lognormal_glm.json` – regression coefficients, $\sigma$, fit metrics

---

## 4. Notebooks

- `notebooks/mode_diagnostics.ipynb` — unified exploration of wait histograms, travel cohorts, and lognormal fits.  Exports PNG snapshots via `plotly.io.write_image` so reviewers see figures on GitHub.
- `notebooks/data_overview.ipynb` — dataset sanity checks.
- `notebooks/travel_lognormal_cohorts.ipynb` — scratchpad for the regression design matrix and comparisons.

Run notebooks from the repo root (so relative paths resolve) with your environment activated.  Execute the “Static snapshot” cells before committing to refresh the PNGs in `docs/figures/`.

---

## 5. Streamlit dashboard

Launch locally once caches exist:

```bash
streamlit run streamlit_app.py
```

The app:
1. Loads taxi / Citi Bike wait summaries and lognormal parameters from `data/derived/`.
2. Lets users drop origin/destination pins or search addresses.
3. Computes walk + wait + travel time for both modes and surfaces the faster option.
4. Falls back to nearest stations/zones if a cohort is missing data.

Deployment notes:
- `streamlit_app.py` references environment variables for Mapbox keys if you want custom tiles.
- `outputs/` can store JSON exports (`wait_cache.json`, `travel_cache.json`) to decouple the app from Parquet.

---

## 6. Repo layout

```
docs/                # report.tex + compiled PDF and published figures
notebooks/           # Jupyter analyses with cached PNGs
scripts/
  build_wait_stats.py
  build_travel_stats.py
src/
  modeling/
    wait_dashboard.py
    travel_diagnostics.py
    travel_times.py
streamlit_app.py
```

---

## 7. Troubleshooting

- **Missing `kaleido` when saving PNGs** — `pip install kaleido`.
- **Streamlit cannot find stats** — rerun the scripts in §3; check that `data/derived/travel_stats/travel_lognormal_glm.json` exists.
- **Large parquet loads** — use the down-sampling knobs at the top of the notebooks (`MAX_TRIPS`, etc.).

---

## 8. Reproducing the report

`docs/report.tex` mirrors the README narrative (wait-method, travel-method, Streamlit).  To rebuild the PDF:

```
cd docs
pdflatex report.tex
```

The repo already contains figures (`docs/figures/*.png`) exported from the diagnostics notebook.

---

Questions or suggestions?  File an issue or ping the authors via the GitHub repo: <https://github.com/AtharvRN/NYC_Public_Transit>.
