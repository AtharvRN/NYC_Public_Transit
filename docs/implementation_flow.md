# Implementation Flow: Components 1 & 2

This document captures the exact code path we built for the taxi / Citi Bike wait‑time (Component 1) and travel‑time (Component 2) enhancements before we move on to the next stage.

## Component 1 – Wait-Time Modeling

1. **Notebook Prototype**  
   - `notebooks/wait_times/combined_dashboard.ipynb` fits arrival histograms + NB overlays and empirical inter-arrival waits.  
   - Logic verified for both taxi zones and Citi Bike stations across hourly buckets.

2. **Reusable Module** (`src/modeling/wait_times.py`)  
   - Loads Jan‑2024 taxi/bike events and aggregates them by `(location_id, hour)` to compute mean arrivals, variance, dispersion, and NB `r/p`.  
   - Calculates empirical wait minutes (time between events), filters outliers, and stores `mean_wait` + exponential (1/λ) approximation (`poisson_wait`).  
   - Caches results to `outputs/wait_stats/{taxi,bike}_hourly_waits.parquet` (plus JSON) so Streamlit reads precomputed stats.

3. **Streamlit Integration** (`streamlit_app.py`)  
   - At app start we call `load_wait_stats()` for taxi/bike and turn the DataFrames into lookup tables via `build_wait_lookup()`.  
   - The `pick_wait_minutes()` helper in the app prefers `mean_wait` when there are at least 5 samples, falls back to cached `poisson_wait` otherwise, and finally to the original on-the-fly `1/λ` if no cache entry exists.  
   - This behavior is used in `recommend()` for both taxi and bike waits, surfacing more realistic headways.

## Component 2 – Travel-Time Modeling

1. **Stats Builder Script** (`scripts/component2_build_travel_stats.py`)  
   - Reads Jan‑2024 taxi parquet + Citi Bike CSVs, filters rides to 1–120 minutes and 0–12 km (straight-line for bikes, TLC distance for taxis).  
   - Assigns equal-width 2 km bins and tags rush/off-peak (`7–10`, `16–19`) plus weekday/weekend.  
   - For each `(mode, distance_bin, is_rush, is_weekend)` group, computes sample count, mean, variance, and method-of-moments Gamma shape/scale (provided ≥50 samples).  
   - Fits a per-mode linear regression (`travel_min = intercept + β_distance·km + β_rush·flag + β_weekend·flag`) when ≥200 samples exist.  
   - Writes `outputs/travel_stats/travel_bins.parquet` and `travel_regression.json`.

2. **Runtime Helper** (`src/modeling/travel_times.py`)  
   - Loads the cached Parquet/JSON lazily and exposes `estimate_travel_minutes()`.  
   - Flow per request:  
     1. Assign equal-width distance bin (2 km steps up to 12 km).  
     2. If bin stats exist and `sample_count ≥ 50`, return the Gamma mean (`TravelEstimate` with `source='gamma_bin'`).  
     3. Else, compute regression fallback (`source='regression'`).  
     4. Else, fall back to legacy constant-speed heuristic (`source='speed_heuristic'`).  
   - Includes `TravelEstimate.details` so the UI can display provenance (sample counts, coefficients, etc.).

3. **Streamlit Wiring** (`streamlit_app.py`)  
   - Adds a “Day type” radio button and derives `is_weekend` + `rush_flag` from selected hour.  
   - Replaces fixed taxi/bike speeds with `safe_travel_estimate(...)` which calls `estimate_travel_minutes()` and gracefully warns if caches are missing.  
   - The results dataframe now includes `travel_source`, and the caption clarifies we’re using Gamma/regression caches with a speed fallback.  
   - Map UI improvements: map clicks show lat/lon, clicking updates origin/destination widgets (including session-state sync) so manual inputs and map interactions stay aligned.

## Supporting UX Enhancements

- Map picker now has toggle buttons (“Click to set ORIGIN/DESTINATION”) and visually displays current coordinates, nearest taxi zones/subway stops, and the last clicked lat/lon.  
- The “Calculation Details” expander prints both Taxi and Citi Bike walk/wait/travel components with safe formatting.

These pieces ensure Component 1 + Component 2 are fully implemented, cached, and surfaced in the Streamlit interface, ready for subsequent Subway integration and Component 3 penalty work.
