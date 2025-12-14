from collections.abc import Sequence

import polars as pl

def bucket_by_time(trips: pl.DataFrame,
                   group_col: Sequence[str] | str = 'start_station',
                   freq: str = '15m',
                   condition: str = 'arrival')-> pl.DataFrame:
    if isinstance(group_col, str):
        group_col = [group_col]
    if condition == 'arrival':
        grouped = (trips.group_by(
                    *[pl.col(c) for c in group_col],
                    pl.col("ended_at")
                      .dt.truncate(freq)
                      .alias("event_time_bucket"),
                )
                .agg(
                    pl.len().alias("arrival")
                )
            )
    elif condition == 'departure':
        grouped = (trips.group_by(
                    *[pl.col(c) for c in group_col],
                    pl.col("started_at")
                      .dt.truncate(freq)
                      .alias("event_time_bucket"),
                )
                .agg(
                    pl.len().alias("departure")
                )
            )
    else:
        raise ValueError('condition must be "arrival" or "departure" but received {}'.format(condition))

    group_arg = group_col[0] if  len(group_col) == 1 else group_col
    pivot = grouped.pivot(
        values = condition,
        index = 'event_time_bucket',
        columns = group_arg,
        aggregate_function = 'sum'
    )
    return pivot

def get_borough(lat, lon):
    """Approximate borough detection based on lat/lon ranges"""
    # Handle None/NaN values
    if lat is None or lon is None:
        return "Unknown"

    # Rough boundaries
    if 40.70 <= lat <= 40.88 and -74.02 <= lon <= -73.91:
        return "Manhattan"
    elif 40.57 <= lat <= 40.74 and -74.06 <= lon <= -73.83:
        return "Brooklyn"
    elif 40.74 <= lat <= 40.92 and -73.93 <= lon <= -73.70:
        return "Bronx"
    elif 40.50 <= lat <= 40.65 and -74.26 <= lon <= -74.05:
        return "Staten Island"
    elif 40.53 <= lat <= 40.80 and -73.96 <= lon <= -73.70:
        return "Queens"
    else:
        return "Unknown"