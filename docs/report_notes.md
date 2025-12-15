# Report Prep Notes

These notes capture findings we want to remember when drafting the final report. They are deliberately concise reminders, not polished prose.

## Travel-time modeling

- **Primary method**: keep Gamma bins + lognormal GLM as main justification (see `notebooks/mode_diagnostics.ipynb`). They provide distance-aware estimates and align with the Streamlit path.
- **Speed-only sanity check**: teammate’s `notebooks/fit_speed.ipynb` fits Weibull distributions to Citi Bike speeds per cohort (weekday/weekend × peak/off-peak). Useful for fallback when the GLM & Gamma filters drop a cohort but too coarse for the main story. We now capture the e-bike bonus explicitly via an `is_ebike` feature in the lognormal GLM.
- **Fallback integration idea**: stash the cohort speed means in `data/derived/travel_stats/bike_speed_fallback.json` and have `legacy_speed_fallback()` pull from it before using a global constant. Worth mentioning as an auxiliary improvement.
- **E-bike vs classic insight**: electric bikes average ~3.5 km/min vs ~2.5 km/min for classic bikes across all cohorts in the speed notebook. The diagnostics + GLM now include an `is_ebike` knob so we can quantify the lift; still worth calling out in the write-up and suggesting as a user-facing toggle in future product work.

## Report structure tips

- Explicitly state that wait-time and travel-time diagnostics now live in `mode_diagnostics.ipynb`; legacy notebooks exist only as pointers.
- Remind reviewers that we downsample raw data (500k row caps) in the combined notebook to keep memory manageable; for production stats we still run the full builder scripts.
