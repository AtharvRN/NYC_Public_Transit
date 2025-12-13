# Component 1 – Wait-Time Modeling Plan

Goal: upgrade taxi/bike wait estimates from naive Poisson (1/lambda) to a dispersion-aware method leveraging negative-binomial fits.

## Status Snapshot (after module integration)
- ✅ Combined taxi + Citi Bike wait diagnostics into `notebooks/wait_times/combined_dashboard.ipynb` (tabs for each mode).
- ✅ For any zone/station + cohort + bucket, we now render:
  * Arrival histogram with Poisson and NB overlays + summary stats (mean, variance, dispersion, NB `r`, `p`).
  * Empirical inter-arrival histogram together with an exponential fit (`λ = 1 / mean_wait`).
- ✅ Dispersion-heavy zones visibly deviate from Poisson; bikes often sit close to exponential waits, taxis show heavier tails but still manageable with an exponential baseline.
- ✅ Extracted the wait-stat computation into `src/modeling/wait_times.py`, including a cache (`outputs/wait_stats/{taxi,bike}_hourly_waits.parquet` + JSON) so Streamlit reads precomputed stats.
- ✅ `streamlit_app.py` now calls the cached lookup: prefer empirical mean wait (fitted exponential) when ≥5 samples, fall back to cached Poisson wait otherwise, and only then to the old `1/λ` estimate if no cache entry exists.
- ⚠️ Future refinement: derive a negative-binomial-aware fallback (e.g., inflating Poisson wait by dispersion) for sparse cells once we agree on the mapping.

## Tasks
1. **Data sourcing**
   - Reuse hourly bucket counts from existing loaders (`taxi_rates`, `bike_rates`).
   - For each zone/station/hour, compute mean and variance of arrivals.

2. **Parameter fitting**
   - Method-of-moments NB fit: `r = mean^2 / (var - mean)`, `p = r / (r + mean)` when `var > mean`.
   - Flag cells where variance <= mean and fall back to Poisson.

3. **Wait-time heuristic**
   - Empirical wait = average inter-arrival minutes (plus percentiles) per zone/station/hour.
   - Fit exponential as a compact parameterization; use dispersion to warn when the exponential underestimates tails.
   - Provide helper returning `{mode, location, hour, mean_arrivals, variance, dispersion, nb_r, nb_p, mean_wait, poisson_wait}`.

4. **Implementation steps**
   - Add helper module (e.g., `src/modeling/wait_times.py`) with functions to:
     * summarize counts -> mean/variance/dispersion/NB params.
     * compute empirical waits + exponential lambda.
     * cache results to disk for fast lookup (Parquet/JSON).
   - Update Streamlit `recommend()` so taxi/bike waits come from this helper (fallback to Poisson only when counts are too sparse).

5. **Testing/validation**
   - Sanity-check a sample of zones/stations by comparing module output to notebook visuals.
   - Ensure the helper handles missing data gracefully and that Streamlit UI reflects dispersion warnings if needed.
