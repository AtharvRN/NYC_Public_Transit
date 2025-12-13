# NYC Public Transit Modeling Docs

## Raw Data Sources
| Dataset | Location | Source URL |
|---------|----------|------------|
| Yellow Taxi (January 2024) | `data/raw/yellow_tripdata_2024-01.parquet` | TLC Trip Record Data (NYC Taxi ") |
| Taxi Zones shapefile | `data/raw/taxi_zones_shp/` | [https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip) |
| Citi Bike monthly trips | `data/raw/202401-citibike-tripdata.zip`, `data/raw/citibike/` | [https://s3.amazonaws.com/tripdata/](https://s3.amazonaws.com/tripdata/) |
| MTA Subway turnstiles | `data/raw/subway/` | [https://www.mta.info/developers/turnstile.html](https://www.mta.info/developers/turnstile.html) |
| Citi Bike station system data | `data/raw/citibike_system_data.html` | Same Citi Bike portal |

## Documentation Layout
- `docs/project_plan.md`: original objectives + milestones.
- `docs/dashboard_roadmap.md`: Streamlit/Dash feature breakdown.
- `docs/poisson_findings.md`: Poisson vs NB findings, tail bounds.
- `docs/data_prep.md`: taxi zone centroid workflow (needs extension for Bike/Subway).
- `docs/report/`: LaTeX write-up + figures (`latexmk -pdf main.tex`).

## Scripts
| Script | Purpose |
|--------|---------|
| `scripts/analyze_manhattan_poisson.py` | Per-zone Poisson vs NB diagnostics (rush/off, weekday/weekend). |
| `scripts/analyze_tail_bounds.py` | Empirical tail probability vs Markov/Chebyshev/Chernoff/Hoeffding bounds. |
| `scripts/build_taxi_zone_centroids.py` | Generate `taxi_zone_centroids.csv` from TLC shapefile. |
| `scripts/analyze_manhattan_hier_nb.py` | Reserved for hierarchical NB (Empirical Bayes) experimentation. |

## Report Assets
- Histograms/plots for the report live under `docs/report/figures/`.
- Re-run scripts before compiling to refresh data + visuals.
