#!/usr/bin/env python3
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
try:
    from streamlit_folium import st_folium
    import folium
except Exception:
    st_folium = None
    folium = None

# Ensure we can import local modules
ROOT = Path(__file__).resolve().parent
for candidate in [ROOT, *ROOT.parents]:
    src_dir = candidate / 'src'
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
        break
from modeling.wait_times import load_wait_stats
from modeling.travel_times import estimate_travel_minutes, legacy_speed_fallback

# ------------------ helpers ------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def nearest_point(df, lat, lon, id_col):
    dists = df.apply(lambda r: haversine_km(lat, lon, r['lat'], r['lon']), axis=1)
    idx = dists.idxmin()
    return df.loc[idx, id_col], df.loc[idx, 'lat'], df.loc[idx, 'lon'], dists.min()

def get_lambda(rate_df, key_col, key_val, hour_col='hour', hour=12):
    subset = rate_df[(rate_df[key_col] == key_val) & (rate_df[hour_col] == hour)]
    if subset.empty:
        return np.nan
    return subset['lambda_per_min'].iloc[0]

def build_wait_lookup(df):
    if df is None or df.empty:
        return {}
    lookup = {}
    for row in df.itertuples(index=False):
        key = (
            row.location_id,
            int(row.hour),
            bool(getattr(row, "is_weekend", False)),
            bool(getattr(row, "is_rush", False)),
        )
        lookup[key] = {
            'mean_wait': float(row.mean_wait) if pd.notna(row.mean_wait) else np.nan,
            'poisson_wait': float(row.poisson_wait) if pd.notna(row.poisson_wait) else np.nan,
            'wait_count': int(row.wait_count) if pd.notna(row.wait_count) else 0,
        }
    return lookup

def build_coord_lookup(df, *, id_candidates, lat_candidates=('lat', 'latitude'), lon_candidates=('lon', 'longitude', 'lng')):
    if df is None:
        return {}
    id_col = next((c for c in id_candidates if c in df.columns), None)
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    if not id_col or not lat_col or not lon_col:
        return {}
    valid = df.dropna(subset=[id_col, lat_col, lon_col])
    coord = {}
    for row in valid.itertuples(index=False):
        record = row._asdict()
        coord[record[id_col]] = (float(record[lat_col]), float(record[lon_col]))
    return coord

def _nearest_wait_minutes(lookup, coord_lookup, location_id, hour, is_weekend, is_rush, min_samples):
    target = coord_lookup.get(location_id)
    if not target:
        return np.nan
    lat0, lon0 = target
    best_wait = np.nan
    best_dist = float('inf')
    for (loc, h, wknd, rush), stats in lookup.items():
        if h != int(hour) or wknd != bool(is_weekend) or rush != bool(is_rush):
            continue
        if stats.get('wait_count', 0) < min_samples or not np.isfinite(stats.get('mean_wait')):
            continue
        coords = coord_lookup.get(loc)
        if not coords:
            continue
        lat1, lon1 = coords
        dist = haversine_km(lat0, lon0, lat1, lon1)
        if dist < best_dist:
            best_dist = dist
            best_wait = stats['mean_wait']
    return best_wait

def pick_wait_minutes(
    lookup,
    location_id,
    hour,
    *,
    is_weekend: bool,
    is_rush: bool,
    fallback=np.nan,
    min_samples=5,
    coord_lookup=None,
):
    stats = lookup.get((location_id, int(hour), bool(is_weekend), bool(is_rush)))
    if stats:
        if np.isfinite(stats.get('mean_wait')) and stats.get('wait_count', 0) >= min_samples:
            return stats['mean_wait']
        if np.isfinite(stats.get('poisson_wait')):
            return stats['poisson_wait']
    if coord_lookup:
        neighbor = _nearest_wait_minutes(
            lookup,
            coord_lookup,
            location_id,
            hour,
            is_weekend,
            is_rush,
            min_samples,
        )
        if np.isfinite(neighbor):
            return neighbor
    return fallback

def is_rush_hour(hour):
    return (7 <= hour < 10) or (16 <= hour < 19)

# ------------------ load data ------------------
st.set_page_config(page_title="NYC Best Mode: Taxi vs Bike", layout="wide")

st.title("🚕 NYC Best Mode: Taxi vs Citi Bike 🚲")
st.write("Compare travel times between taxi and Citi Bike using Jan 2024 data.")

root = ROOT

def load_taxi():
    derived = root / 'data' / 'derived'
    rates_path = derived / 'taxi_rates.parquet'
    centroids_path = derived / 'taxi_centroids.parquet'
    if not rates_path.exists() or not centroids_path.exists():
        raise FileNotFoundError("Missing taxi derived files. Run scripts/build_travel_stats.py or preprocessing scripts to generate data/derived/taxi_*.parquet.")
    rates = pd.read_parquet(rates_path)
    centroids = pd.read_parquet(centroids_path)
    return rates, centroids

def load_bike():
    derived = root / 'data' / 'derived'
    rates_path = derived / 'citibike_rates.parquet'
    stations_path = derived / 'citibike_stations.parquet'
    if not rates_path.exists() or not stations_path.exists():
        st.warning("Missing Citi Bike derived files. Run preprocessing to generate data/derived/citibike_*.parquet.")
        return None, None
    rates = pd.read_parquet(rates_path)
    stations = pd.read_parquet(stations_path)
    return rates, stations

@st.cache_data(show_spinner=False)
def load_all():
    taxi_rates, centroids = load_taxi()
    bike_rates, stations = load_bike()
    return taxi_rates, centroids, bike_rates, stations

try:
    taxi_rates, centroids, bike_rates, stations = load_all()
except Exception as exc:
    st.error(f"Failed to load data: {exc}")
    st.stop()

def _safe_wait_stats(mode):
    try:
        return load_wait_stats(mode)
    except FileNotFoundError as exc:
        st.warning(f"{mode.title()} wait cache unavailable: {exc}")
    except Exception as exc:
        st.warning(f"Failed to load {mode} wait stats: {exc}")
    return None

with st.spinner("Loading wait-time summaries..."):
    taxi_wait_df = _safe_wait_stats('taxi')
    bike_wait_df = _safe_wait_stats('bike')
taxi_wait_lookup = build_wait_lookup(taxi_wait_df)
bike_wait_lookup = build_wait_lookup(bike_wait_df)
taxi_coord_lookup = build_coord_lookup(centroids, id_candidates=('LocationID', 'location_id'))
bike_coord_lookup = build_coord_lookup(stations, id_candidates=('station_id', 'StationID'))

def safe_travel_estimate(mode, distance_km, is_rush, is_weekend, fallback_speed):
    try:
        return estimate_travel_minutes(
            mode,
            distance_km,
            is_rush=is_rush,
            is_weekend=is_weekend,
            speed_fallback_kmh=fallback_speed,
        )
    except FileNotFoundError:
        if not st.session_state.get('travel_stats_missing_warned'):
            st.warning(f"Travel stats missing. Using speed fallback.")
            st.session_state['travel_stats_missing_warned'] = True
    except Exception:
        if not st.session_state.get('travel_stats_generic_warned'):
            st.warning(f"Failed to compute travel stats. Using fallback.")
            st.session_state['travel_stats_generic_warned'] = True
    return legacy_speed_fallback(distance_km, fallback_speed)

# ------------------ Initialize session state ------------------
if 'origin_lat' not in st.session_state:
    st.session_state['origin_lat'] = 40.7580  # Times Square
if 'origin_lon' not in st.session_state:
    st.session_state['origin_lon'] = -73.9855
if 'dest_lat' not in st.session_state:
    st.session_state['dest_lat'] = 40.8075  # Columbia University
if 'dest_lon' not in st.session_state:
    st.session_state['dest_lon'] = -73.9626
if 'picking_mode' not in st.session_state:
    st.session_state['picking_mode'] = 'origin'
if 'last_click' not in st.session_state:
    st.session_state['last_click'] = None

# ------------------ Sidebar ------------------
st.sidebar.header("⚙️ Settings")
hour = st.sidebar.slider('Hour of day', 0, 23, 9)
day_type = st.sidebar.radio('Day type', ['Weekday', 'Weekend'], horizontal=True, index=0)
is_weekend_choice = day_type == 'Weekend'

# Constants
WALK_SPEED_KMH = 5.0
TAXI_SPEED_KMH = 15.0
BIKE_SPEED_KMH = 12.0

# ------------------ Main interface ------------------
st.subheader("📍 Select Origin and Destination")

# Two column layout for mode selection
col1, col2 = st.columns(2)
with col1:
    if st.button(
        '🟢 SET ORIGIN',
        type='primary' if st.session_state['picking_mode'] == 'origin' else 'secondary',
        use_container_width=True
    ):
        st.session_state['picking_mode'] = 'origin'

with col2:
    if st.button(
        '🔴 SET DESTINATION',
        type='primary' if st.session_state['picking_mode'] == 'destination' else 'secondary',
        use_container_width=True
    ):
        st.session_state['picking_mode'] = 'destination'

# Show current mode
if st.session_state['picking_mode'] == 'origin':
    st.info('👆 Click anywhere on the map below to set the **ORIGIN** (green marker)')
else:
    st.info('👆 Click anywhere on the map below to set the **DESTINATION** (red marker)')

# Display current coordinates
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**🟢 Origin:** `{st.session_state['origin_lat']:.6f}, {st.session_state['origin_lon']:.6f}`")
with col2:
    st.markdown(f"**🔴 Destination:** `{st.session_state['dest_lat']:.6f}, {st.session_state['dest_lon']:.6f}`")

# ------------------ Map ------------------
if st_folium is None or folium is None:
    st.error('Please install: pip install streamlit-folium folium')
    st.stop()

# Pre-compute nearest bike stations for visualization/calcs
bike_start = None
bike_end = None
if stations is not None and not stations.empty:
    start_id, start_lat, start_lon, start_dist = nearest_point(
        stations, st.session_state["origin_lat"], st.session_state["origin_lon"], "start_station_id"
    )
    end_id, end_lat, end_lon, end_dist = nearest_point(
        stations, st.session_state["dest_lat"], st.session_state["dest_lon"], "start_station_id"
    )
    bike_start = {
        "id": start_id,
        "lat": start_lat,
        "lon": start_lon,
        "distance_km": start_dist,
    }
    bike_end = {
        "id": end_id,
        "lat": end_lat,
        "lon": end_lon,
        "distance_km": end_dist,
    }

# Build map
center_lat = (st.session_state['origin_lat'] + st.session_state['dest_lat']) / 2
center_lon = (st.session_state['origin_lon'] + st.session_state['dest_lon']) / 2
fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='cartodbpositron')

# Add origin marker (green)
folium.Marker(
    location=[st.session_state['origin_lat'], st.session_state['origin_lon']],
    popup=f"Origin<br>{st.session_state['origin_lat']:.6f}, {st.session_state['origin_lon']:.6f}",
    icon=folium.Icon(color='green', icon='play', prefix='fa'),
    tooltip='Origin'
).add_to(fmap)

# Add destination marker (red)
folium.Marker(
    location=[st.session_state['dest_lat'], st.session_state['dest_lon']],
    popup=f"Destination<br>{st.session_state['dest_lat']:.6f}, {st.session_state['dest_lon']:.6f}",
    icon=folium.Icon(color='red', icon='stop', prefix='fa'),
    tooltip='Destination'
).add_to(fmap)

# Add connecting line
folium.PolyLine(
    locations=[[st.session_state['origin_lat'], st.session_state['origin_lon']],
               [st.session_state['dest_lat'], st.session_state['dest_lon']]],
    color='blue',
    weight=3,
    opacity=0.7
).add_to(fmap)

# Highlight nearest bike stations and segments
if bike_start:
    folium.Marker(
        location=[bike_start["lat"], bike_start["lon"]],
        popup=f"Bike start station {bike_start['id']}",
        icon=folium.Icon(color='blue', icon='bicycle', prefix='fa')
    ).add_to(fmap)
    folium.PolyLine(
        locations=[[st.session_state['origin_lat'], st.session_state['origin_lon']],
                   [bike_start["lat"], bike_start["lon"]]],
        color='gray',
        weight=2,
        opacity=0.8,
        dash_array='5,5'
    ).add_to(fmap)

if bike_end:
    folium.Marker(
        location=[bike_end["lat"], bike_end["lon"]],
        popup=f"Bike end station {bike_end['id']}",
        icon=folium.Icon(color='purple', icon='flag-checkered', prefix='fa')
    ).add_to(fmap)
    folium.PolyLine(
        locations=[[bike_end["lat"], bike_end["lon"]],
                   [st.session_state['dest_lat'], st.session_state['dest_lon']]],
        color='gray',
        weight=2,
        opacity=0.8,
        dash_array='5,5'
    ).add_to(fmap)

if bike_start and bike_end:
    folium.PolyLine(
        locations=[[bike_start["lat"], bike_start["lon"]],
                   [bike_end["lat"], bike_end["lon"]]],
        color='dodgerblue',
        weight=3,
        opacity=0.8
    ).add_to(fmap)

folium.LayerControl().add_to(fmap)

# Render map and capture clicks
map_output = st_folium(
    fmap,
    height=500,
    width='100%',
    key='main_map',
    returned_objects=['last_clicked']
)

# Handle map clicks
if map_output and map_output.get('last_clicked'):
    click = map_output['last_clicked']
    click_lat = round(float(click['lat']), 6)
    click_lon = round(float(click['lng']), 6)
    
    # Check if this is a new click (avoid duplicate processing)
    new_click = (click_lat, click_lon, st.session_state['picking_mode'])
    if new_click != st.session_state['last_click']:
        st.session_state['last_click'] = new_click
        
        # Update the appropriate location
        if st.session_state['picking_mode'] == 'origin':
            st.session_state['origin_lat'] = click_lat
            st.session_state['origin_lon'] = click_lon
            st.success(f'✅ Origin set to: {click_lat:.6f}, {click_lon:.6f}')
            # Auto-switch to destination
            st.session_state['picking_mode'] = 'destination'
            st.rerun()
        else:
            st.session_state['dest_lat'] = click_lat
            st.session_state['dest_lon'] = click_lon
            st.success(f'✅ Destination set to: {click_lat:.6f}, {click_lon:.6f}')
            st.rerun()

# ------------------ Analysis ------------------
st.markdown("---")
st.subheader("📊 Travel Time Comparison")

origin = (st.session_state['origin_lat'], st.session_state['origin_lon'])
dest = (st.session_state['dest_lat'], st.session_state['dest_lon'])
direct_km = haversine_km(*origin, *dest)
rush_flag = is_rush_hour(hour)

st.info(f"📏 Direct distance: **{direct_km:.2f} km**")

# Calculate for both modes
results = []
walk_to_start_min = None
walk_to_dest_min = None
ride_km = None
wait_bike = None

# TAXI
tz_id, tz_lat, tz_lon, _ = nearest_point(centroids, *origin, 'PULocationID')
lam_taxi = get_lambda(taxi_rates, 'PULocationID', tz_id, hour=hour)
fallback_taxi = np.nan if not np.isfinite(lam_taxi) or lam_taxi <= 0 else 1 / lam_taxi
wait_taxi = pick_wait_minutes(
    taxi_wait_lookup,
    tz_id,
    hour,
    is_weekend=is_weekend_choice,
    is_rush=rush_flag,
    fallback=fallback_taxi,
    coord_lookup=taxi_coord_lookup,
)
travel_taxi_est = safe_travel_estimate('taxi', direct_km, rush_flag, is_weekend_choice, TAXI_SPEED_KMH)
travel_taxi_min = travel_taxi_est.minutes
total_taxi = wait_taxi + travel_taxi_min if np.isfinite(wait_taxi) else np.nan

results.append({
    'Mode': '🚕 Taxi',
    'Wait (min)': f"{wait_taxi:.1f}" if np.isfinite(wait_taxi) else "N/A",
    'Walk to station (min)': "0.0",
    'Ride/Bike (min)': f"{travel_taxi_min:.1f}",
    'Walk to destination (min)': "0.0",
    'Total (min)': f"{total_taxi:.1f}" if np.isfinite(total_taxi) else "N/A",
    'total_numeric': total_taxi
})

# BIKE
if bike_rates is not None and bike_start and bike_end:
    lam_bike = get_lambda(bike_rates, 'start_station_id', bike_start['id'], hour=hour)
    fallback_bike = np.nan if not np.isfinite(lam_bike) or lam_bike <= 0 else 1 / lam_bike
    wait_bike = pick_wait_minutes(
        bike_wait_lookup,
        bike_start['id'],
        hour,
        is_weekend=is_weekend_choice,
        is_rush=rush_flag,
        fallback=fallback_bike,
        coord_lookup=bike_coord_lookup,
    )
    walk_to_start_min = (bike_start['distance_km'] / WALK_SPEED_KMH) * 60
    walk_to_dest_min = (bike_end['distance_km'] / WALK_SPEED_KMH) * 60
    ride_km = haversine_km(bike_start['lat'], bike_start['lon'], bike_end['lat'], bike_end['lon'])
    ride_km = max(0.1, ride_km)
    travel_bike_est = safe_travel_estimate('bike', ride_km, rush_flag, is_weekend_choice, BIKE_SPEED_KMH)
    travel_bike_min = travel_bike_est.minutes
    total_bike = wait_bike + walk_to_start_min + travel_bike_min + walk_to_dest_min if np.isfinite(wait_bike) else np.nan
    
    results.append({
        'Mode': '🚲 Citi Bike',
        'Wait (min)': f"{wait_bike:.1f}" if np.isfinite(wait_bike) else "N/A",
        'Walk to station (min)': f"{walk_to_start_min:.1f}",
        'Ride/Bike (min)': f"{travel_bike_min:.1f}",
        'Walk to destination (min)': f"{walk_to_dest_min:.1f}",
        'Total (min)': f"{total_bike:.1f}" if np.isfinite(total_bike) else "N/A",
        'total_numeric': total_bike
    })

# Display results
result_df = pd.DataFrame(results)
cols_to_show = [
    'Mode',
    'Wait (min)',
    'Walk to station (min)',
    'Ride/Bike (min)',
    'Walk to destination (min)',
    'Total (min)',
]
st.dataframe(result_df[cols_to_show], width='stretch', hide_index=True)

# Bar chart
valid_results = result_df[result_df['total_numeric'].notna()]
if not valid_results.empty:
    fig = go.Figure(go.Bar(
        x=valid_results['Mode'],
        y=valid_results['total_numeric'],
        text=valid_results['total_numeric'].round(1),
        textposition='outside',
        marker_color=['#FFD700', '#1E90FF']
    ))
    fig.update_layout(
        yaxis_title='Total Time (minutes)',
        title='Total Travel Time Comparison',
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig, width='stretch')
    
    # Winner announcement
    winner_idx = valid_results['total_numeric'].idxmin()
    winner = valid_results.loc[winner_idx, 'Mode']
    winner_time = valid_results.loc[winner_idx, 'total_numeric']
    st.success(f"🏆 **Fastest mode: {winner}** at {winner_time:.1f} minutes total!")
else:
    st.warning("Not enough data to compare modes for this route.")

# Additional info
with st.expander("ℹ️ Calculation Details"):
    taxi_md = f"""
    **Taxi (Zone {tz_id}):**
    - Wait for taxi: {wait_taxi:.1f} min
    - Door-to-door ride distance: {direct_km:.2f} km
    """
    st.markdown(taxi_md)

    if bike_start and bike_end and wait_bike is not None:
        bike_md = f"""
        **Citi Bike:**
        - Walk to start station {bike_start['id']}: {bike_start['distance_km']:.3f} km ({walk_to_start_min:.1f} min)
        - Wait for bike: {wait_bike:.1f} min
        - Bike ride between stations: {ride_km:.2f} km
        - Walk from station {bike_end['id']} to destination: {bike_end['distance_km']:.3f} km ({walk_to_dest_min:.1f} min)
        """
        st.markdown(bike_md)
    else:
        st.markdown("**Citi Bike:** station data unavailable for this location.")

    st.caption(f"*Wait times estimated using historical data at {hour}:00*")

st.caption("💡 Click the green/red buttons above to change origin/destination on the map. Analysis updates automatically.")
