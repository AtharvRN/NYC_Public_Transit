from collections.abc import Sequence
from scipy.stats import norm,poisson,nbinom,chi2
from math import lgamma

import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

def group_hour(trips: pl.DataFrame):

    d_hour = trips.with_columns(
        pl.col("event_time_bucket").dt.hour().alias("hour")
    )
    grouped_hour = (d_hour.group_by("hour").agg(pl.all().exclude('event_time_bucket','hour').sum()))
    print(grouped_hour)
    return grouped_hour

def group_day(trips: pl.DataFrame):

    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    d_day = trips.with_columns(pl.col("event_time_bucket")
                               .dt.weekday()
                               .map_elements( lambda i: weekday_names[int(i-1)] if i-1 is not None and 0 <= int(i-1) < 7 else None,return_dtype=pl.Utf8,)
                               .alias("day"))
    grouped_day = (d_day.group_by('day').agg(pl.all().exclude('event_time_bucket','day').sum()))
    return grouped_day

def group_hour_a_day(zone_data: pl.DataFrame):
    if 'event_time_bucket' not in zone_data.columns:
        raise ValueError("trips must contain 'event_time_bucket' column")
    name_col = zone_data.columns[0]
    d_hour = zone_data.with_columns([pl.col("event_time_bucket").dt.date().alias("date"),
                                     pl.col("event_time_bucket").dt.hour().alias("hour")])
    d_hour = d_hour.with_columns( pl.sum_horizontal(pl.all()
                                                    .exclude(["event_time_bucket", "date", "hour"])).alias("total_count"))
    grouped_hour = (d_hour.group_by(["date", "hour"])
                    .agg(pl.col("total_count").sum()
                         .alias("hourly_count")))
    return grouped_hour


def rush_off_bound(trips: pl.DataFrame,
                 threshold:float = 0.95,
                    condition:str = 'hour',
                   check_var: str = 'rush',
                   bound:str = 'markov',
                   dist_var: str = 'gaussian'):
    if 'event_time_bucket' not in trips.columns:
        raise ValueError("trips must contain 'event_time_bucket' column")
    if condition == 'hour':
        data_group = group_hour(trips)
    elif condition == 'day':
        data_group = group_day(trips)
    else:
        raise ValueError('condition must be "hour" or "day" but received {}'.format(condition))
    num_cols = data_group.columns
    index_col = num_cols[0]
    num_cols = num_cols[1:]
    # print(num_cols)
    check_dictionary = {}
    #Using method of moments(MoM) and using
    if dist_var =='gaussian':
        means = data_group.select([pl.col(c).mean().alias(c) for c in num_cols])
        stds = data_group.select([(pl.col(c) ** 2).mean().alias(c) for c in num_cols])
    elif dist_var =='poisson':
        means = data_group.select([pl.col(c).mean().alias(c) for c in num_cols])
        stds = means
    elif dist_var == 'none':
        for col in num_cols:
            if check_var == 'rush':
                percentile_threshold = data_group.select(pl.col(col).quantile(threshold))
                # print(percentile_threshold)
                filtered_col = data_group.select(pl.col(index_col),
                                                 (pl.col(col)-percentile_threshold.item()).alias(col))
            elif check_var == 'off':
                percentile_threshold = data_group.select(pl.col(col).quantile(threshold))
                filtered_col = data_group.select(pl.col(index_col),
                                                 (percentile_threshold.item()-pl.col(col)).alias(col))
            else:
                raise ValueError('check_car must be "rush" or "off" but received {}'.format(condition))
            # print(f'filtered_col:{filtered_col}')
            filtered_rows = filtered_col.filter(pl.col(col) > 0).select(pl.first()).head(100000)
            # print(f'filtered_rows:{filtered_rows}')
            # check_dictionary[col] = filtered_rows
            check_dictionary[col] = filtered_rows.to_series().to_list()
        return check_dictionary
    else:
        raise ValueError('dist_var provided is not listed {}'.format(dist_var))


    if bound == 'cantelli':
        bound_var = stds.with_columns([((pl.col(c) * threshold) / (1 - threshold))
                                     .sqrt().alias(c)for c in num_cols])
    elif bound == 'markov':
        bound_var = means.with_columns([(pl.col(c) /(1-threshold))
                                     .sqrt().alias(c)for c in num_cols])
    else:
        raise ValueError('bound must be "cantelli" or "markov" but received {}'.format(bound))

    for col in num_cols:
        if bound == 'cantelli':
            if check_var == 'rush':
                filtered_col = data_group.select(pl.col(index_col),
                                                 (pl.col(col) - means[col].item()-bound_var[col].item()).alias(col))
            elif check_var == 'off':
                filtered_col = data_group.select(pl.col(index_col),
                                                 (means[col].item() - pl.col(col) -bound_var[col].item
                                                 ()).alias(col))
            else:
                raise ValueError('check_car must be "rush" or "off" but received {}'.format(condition))
        elif bound == 'markov':
            if check_var == 'rush':
                filtered_col = data_group.select([(pl.col(col) -bound_var[col].item()).alias(col)])
            elif check_var == 'off':
                raise ValueError('Markov cannot give lower bound')
            else:
                raise ValueError('check_car must be "rush" or "off" but received {}'.format(condition))
        else:
            raise ValueError('bound must be "cantelli" or "markov" but received {}'.format(bound))

        filtered_rows = filtered_col.filter(pl.col(col) > 0).select(pl.first()).head(100000)
        check_dictionary[col] = filtered_rows.to_series().to_list()

    return check_dictionary
