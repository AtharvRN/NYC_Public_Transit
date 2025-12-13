# Component 2 – Travel-Time Modeling Plan

Goal: add lightweight, probability-friendly travel-time estimates for taxi and Citi Bike that can be slotted into the Streamlit recommender without heavy ML infrastructure.

## Design Principles
- Keep it simple: rely on distributions and statistics covered in standard probability/STA courses (Gamma/Lognormal fits, method-of-moments, linear regression on a handful of predictors).
- Cover only taxi + Citi Bike for now; subway stays on the heuristic baseline from Component 1.
- Reuse existing January 2024 taxi/Bike trip datasets so we do not add new dependencies.
- Surface parameters (shape/scale, mean/variance) in cached files for quick lookup, mirroring the wait-time workflow.

## Proposed Modeling Approach
1. **Feature Engineering**
   - Compute straight-line distance (already available via `haversine_km`) for every sample trip.
   - Derive `is_weekend`, `is_rush`, `hour`, and simple distance bins (e.g., `[0-1km, 1-3km, 3-5km, 5km+]`).
   - For bikes, keep only trips whose start/end stations have valid coordinates.

2. **Distribution Fits**
   - Use travel-time samples grouped by `(mode, distance_bin, rush_flag)`:
     * Fit **Gamma** distribution via method-of-moments: `k = mean^2 / var`, `θ = var / mean`.
     * Record empirical mean/variance and sample counts for diagnostic displays.
   - Gamma keeps us in the "positive support" territory and is a standard STA topic; optionally also compute log-normal parameters for cross-checking.

3. **Regression Backstop**
   - Fit a simple **linear model** predicting travel minutes from distance, rush flag, and weekend flag per mode. Keep coefficients so we can baseline a trip even when a bin is sparse.
   - Prediction = max(2 min, linear estimate). Use regression whenever the Gamma bin has <50 trips.

4. **Caching + API**
   - Added `src/modeling/travel_times.py` mirroring the wait helper:
     * `compute_travel_stats(mode)` → returns/caches Gamma params, regression coefficients, sample counts.
     * `get_travel_summary(mode, distance_km, rush_flag)` → returns `gamma_mean`, `gamma_scale`, `regression_mean`, etc.
   - Persist to `outputs/travel_stats/{mode}_travel.parquet` + JSON for portability.

5. **Streamlit Integration (Component 2 deliverable)**
   - Replace the fixed speeds with: `travel_min = lookup_gamma_mean(...)` when available, else regression fallback, else original speed heuristic.
   - Display the source of each estimate (Gamma vs regression vs heuristic) for transparency.

## Milestones
1. **Data prep scripts** (see `scripts/component2_build_travel_stats.py`) to derive `(distance_bin, rush_flag, weekend_flag)` groupings for both modes using equal-width 2 km bins up to 12 km and thresholds (≥50 samples for Gamma, ≥200 for regression).
2. **Modeling helper** that writes cached Gamma parameters + regression coefficients.
3. **Validation plots/notebook** comparing empirical histograms vs fitted Gamma for a few representative bins.
4. **Streamlit wiring** to consume the cached stats and expose estimate provenance.

## Open Questions / Next Steps
- Decide on exact binning strategy (equal-width distance bins vs quantile-based).
- Confirm whether we need cohort splits beyond rush/off-peak (e.g., borough pair). Keep initial scope small unless accuracy is poor.
- Determine acceptable thresholds for sample counts (defaulted to 50 for Gamma and 200 for regression training); adjust after first pass.
- Consider extending the helper later to compute percentile estimates (e.g., 75th percentile) for pessimistic ETA modes.
