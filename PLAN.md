# Project Upgrade Plan

Interactive multi-modal ETA recommender enhancement plan:

1. **Component 1 – Wait-Time Modeling**
   - ✅ Implemented `src/modeling/wait_times.py` to cache hourly stats + empirical waits; Streamlit now consumes the cached exponential/Poisson wait blend. See `component1_wait_model.md` for details and next refinements.

2. **Component 2 – Travel-Time Modeling**
   - Fit lightweight regression models for taxi and Citi Bike travel durations using historical trips (distance, rush/off-peak flags, borough pairs, etc.).
   - Subway travel estimates: keep speed-based baseline but add rush-transfer penalties.

3. **Component 3 – Penalty Scoring**
   - Define cost/environment/walking penalties and combine them with ETA into an overall score with user-tunable weights.

4. **Streamlit Integration**
   - Wire new components into `streamlit_app.py`, update UI controls (sliders, charts), and surface assumptions in the sidebar.

Tracking individual component plans in separate markdown files:
- `component1_wait_model.md`
- `component2_travel_model.md`
- `component3_penalty_score.md`
