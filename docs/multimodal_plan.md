# Multimodal “Best-Mode” Demo (Quick Build Plan)

Goal: an interactive notebook/Streamlit view that, given an origin–destination (OD) in NYC, recommends Taxi vs Citi Bike vs Subway using simple, explainable heuristics derived from our datasets. Keep scope to ~4 hours of build time.

## Data We Already Have
- **Taxi:** Jan-2024 Yellow trip parquet + zone centroids (`taxi_zone_centroids.csv`).
- **Citi Bike:** Jan-2024 trips + station locations (from CSV lat/lon) and station-level arrival rates.
- **Subway:** Turnstile counts (entry/exit by station) and station locations (in turnstile feed).

## Simple Modeling (lightweight, no routing engine)
For any OD point:
1. **Nearest stop/stand mapping**
   - Citi Bike: nearest start station by haversine on `start_lat/lng` (use station table from trip CSVs).
   - Subway: nearest station from turnstile metadata (station name + lat/lon; derive if needed).
   - Taxi: nearest taxi zone centroid.
2. **Arrival / wait estimates (from counts)**
   - Taxi: use zone pickup rate λ_taxi → expected wait ~ 1/λ_taxi (bounded by a floor/ceiling).
   - Citi Bike: use station start/return rates; proxy bike availability as high if returns > departures; wait ~ 1/λ_return when needing a bike.
   - Subway: use turnstile entry rate by hour as a proxy for headway; expected wait ~ 1/λ_subway_entry (coarse).
3. **In-vehicle travel time (rough heuristics)**
   - Taxi: straight-line distance / avg taxi speed (e.g., 15 km/h off-peak, 10 km/h peak).
   - Citi Bike: straight-line distance / 12 km/h.
   - Subway: straight-line distance / 25 km/h (or 1.5× taxi speed) as a coarse subway speed.
4. **Score with equal weights on (time, convenience, environment, cost)**
   - Time = wait + in-vehicle.
   - Convenience = inverse of number of transfers/walk; approximate with walk distance to access point.
   - Environment = rank Bike best, Subway middle, Taxi worst.
   - Cost = rank Subway best, Bike middle, Taxi worst (or plug flat fares if desired).
   - Normalize each to 0–1, average, pick lowest score.

## Minimal Deliverable
- A notebook or Streamlit app:
  - Inputs: origin lat/lon (or click on a map), destination lat/lon, time-of-day.
  - Outputs: table of Taxi vs Bike vs Subway with:
    - Estimated wait, travel time, simple cost/convenience/env scores.
    - A recommendation (argmin of average score).
  - Plots: small bar chart comparing total times; optional map showing chosen stations/zones.
- Reuse existing helper code for loading taxi/bike/subway counts; add a tiny helper to find nearest station/zone.

## Tasks (short and bounded)
1. Build a small loader to cache:
   - Taxi zone centroids + hourly pickup λ per zone.
   - Citi Bike station lat/lon + hourly start/return λ.
   - Subway station lat/lon + hourly entry λ.
2. Write a nearest-point function (haversine) to map OD to candidate taxi zone / bike station / subway station.
3. Implement heuristic time + wait calculators per mode (as above).
4. Implement scoring (time, convenience, environment, cost) with equal weights → recommendation.
5. Wire into a notebook/Streamlit with map and a comparison table.

## Optional Public Datasets (if time permits)
- **MTA GTFS static** (routes/shapes) for better subway distance; <http://web.mta.info/developers/developer-data-terms.html>.
- **Citi Bike station status** (live availability) via their GBFS feed for real-time bike counts.
- **NYC Street Centerlines** (DoITT) for better bike/taxi distance vs straight-line.

Keep it simple: start with straight-line distances + average speeds and the rates we already estimated; add GTFS/GBFS only if time remains.
