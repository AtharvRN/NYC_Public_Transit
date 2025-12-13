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
        key = (row.location_id, int(row.hour))
        lookup[key] = {
            'mean_wait': float(row.mean_wait) if pd.notna(row.mean_wait) else np.nan,
            'poisson_wait': float(row.poisson_wait) if pd.notna(row.poisson_wait) else np.nan,
            'wait_count': int(row.wait_count) if pd.notna(row.wait_count) else 0,
        }
    return lookup

def pick_wait_minutes(lookup, location_id, hour, fallback=np.nan, min_samples=5):
    stats = lookup.get((location_id, int(hour)))
    if stats:
        if np.isfinite(stats.get('mean_wait')) and stats.get('wait_count', 0) >= min_samples:
            return stats['mean_wait']
        if np.isfinite(stats.get('poisson_wait')):
            return stats['poisson_wait']
    return fallback

def is_rush_hour(hour):
    return (7 <= hour < 10) or (16 <= hour < 19)

# ------------------ load data ------------------
st.set_page_config(page_title="NYC Best Mode: Taxi vs Bike", layout="wide")

st.title("🚕 NYC Best Mode: Taxi vs Citi Bike 🚲")
st.write("Compare travel times between taxi and Citi Bike using Jan 2024 data.")

root = ROOT

def load_taxi(max_rows=4000_000):
    from modeling.poisson_zone import load_taxi_pickups
    taxi_path = root / 'data/raw/yellow_tripdata_2024-01.parquet'
    taxi = load_taxi_pickups(taxi_path, max_rows=max_rows)
    taxi['hour'] = taxi['event_time'].dt.hour
    rates = taxi.groupby(['PULocationID', 'hour']).size().rename('rides').reset_index()
    rates['lambda_per_min'] = rates['rides'] / 60.0
    centroids = pd.read_csv(root / 'data/raw/taxi_zone_centroids.csv')[['LocationID', 'lon', 'lat']]
    centroids = centroids.rename(columns={'LocationID': 'PULocationID'})
    return rates, centroids

def load_bike(max_rows=500_000):
    files = sorted((root / 'data/raw/citibike').glob('202401-citibike-tripdata_*.csv'))
    if not files:
        st.warning('No Citi Bike CSVs found.')
        return None, None
    frames = []
    for i, f in enumerate(files):
        frames.append(pd.read_csv(
            f,
            nrows=max_rows if i == 0 else None,
            dtype={'start_station_id': str, 'end_station_id': str},
            low_memory=False,
        ))
    bike = pd.concat(frames, ignore_index=True)
    bike['started_at'] = pd.to_datetime(bike['started_at'])
    bike['hour'] = bike['started_at'].dt.hour
    bike = bike.dropna(subset=['start_station_id', 'start_lat', 'start_lng'])
    bike['start_station_id'] = bike['start_station_id'].astype(str)
    rates = bike.groupby(['start_station_id', 'hour']).size().rename('rides').reset_index()
    rates['lambda_per_min'] = rates['rides'] / 60.0
    stations = bike[['start_station_id', 'start_lat', 'start_lng']].drop_duplicates()
    stations = stations.rename(columns={'start_lat': 'lat', 'start_lng': 'lon'})
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

# Add taxi zones (light background)
taxi_group = folium.FeatureGroup(name='Taxi Zones')
for _, r in centroids.sample(min(100, len(centroids)), random_state=0).iterrows():
    folium.CircleMarker(
        location=[r.lat, r.lon],
        radius=2,
        color='orange',
        fill=True,
        fill_opacity=0.2
    ).add_to(taxi_group)
taxi_group.add_to(fmap)

# Add bike stations (light background)
if stations is not None and not stations.empty:
    bike_group = folium.FeatureGroup(name='Bike Stations')
    for _, r in stations.sample(min(100, len(stations)), random_state=0).iterrows():
        folium.CircleMarker(
            location=[r.lat, r.lon],
            radius=2,
            color='blue',
            fill=True,
            fill_opacity=0.2
        ).add_to(bike_group)
    bike_group.add_to(fmap)

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

# TAXI
tz_id, tz_lat, tz_lon, tz_walk_km = nearest_point(centroids, *origin, 'PULocationID')
lam_taxi = get_lambda(taxi_rates, 'PULocationID', tz_id, hour=hour)
fallback_taxi = np.nan if not np.isfinite(lam_taxi) or lam_taxi <= 0 else 1 / lam_taxi
wait_taxi = pick_wait_minutes(taxi_wait_lookup, tz_id, hour, fallback=fallback_taxi)
walk_taxi_min = (tz_walk_km / WALK_SPEED_KMH) * 60
travel_taxi_est = safe_travel_estimate('taxi', direct_km, rush_flag, is_weekend_choice, TAXI_SPEED_KMH)
travel_taxi_min = travel_taxi_est.minutes
total_taxi = wait_taxi + walk_taxi_min + travel_taxi_min if np.isfinite(wait_taxi) else np.nan

results.append({
    'Mode': '🚕 Taxi',
    'Wait (min)': f"{wait_taxi:.1f}" if np.isfinite(wait_taxi) else "N/A",
    'Walk (min)': f"{walk_taxi_min:.1f}",
    'Travel (min)': f"{travel_taxi_min:.1f}",
    'Total (min)': f"{total_taxi:.1f}" if np.isfinite(total_taxi) else "N/A",
    'total_numeric': total_taxi
})

# BIKE
if bike_rates is not None and stations is not None and not stations.empty:
    st_id, st_lat, st_lon, st_walk_km = nearest_point(stations, *origin, 'start_station_id')
    lam_bike = get_lambda(bike_rates, 'start_station_id', st_id, hour=hour)
    fallback_bike = np.nan if not np.isfinite(lam_bike) or lam_bike <= 0 else 1 / lam_bike
    wait_bike = pick_wait_minutes(bike_wait_lookup, st_id, hour, fallback=fallback_bike)
    walk_bike_min = (st_walk_km / WALK_SPEED_KMH) * 60
    travel_bike_est = safe_travel_estimate('bike', direct_km, rush_flag, is_weekend_choice, BIKE_SPEED_KMH)
    travel_bike_min = travel_bike_est.minutes
    total_bike = wait_bike + walk_bike_min + travel_bike_min if np.isfinite(wait_bike) else np.nan
    
    results.append({
        'Mode': '🚲 Citi Bike',
        'Wait (min)': f"{wait_bike:.1f}" if np.isfinite(wait_bike) else "N/A",
        'Walk (min)': f"{walk_bike_min:.1f}",
        'Travel (min)': f"{travel_bike_min:.1f}",
        'Total (min)': f"{total_bike:.1f}" if np.isfinite(total_bike) else "N/A",
        'total_numeric': total_bike
    })

# Display results
result_df = pd.DataFrame(results)
st.dataframe(result_df[['Mode', 'Wait (min)', 'Walk (min)', 'Travel (min)', 'Total (min)']], 
             width='stretch', hide_index=True)

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
    st.markdown(f"""
    **Taxi (Zone {tz_id}):**
    - Walk to nearest taxi zone: {tz_walk_km:.3f} km ({walk_taxi_min:.1f} min)
    - Wait for taxi: {wait_taxi:.1f} min
    - Travel distance: {direct_km:.2f} km
    
    **Citi Bike (Station {st_id if 'st_id' in locals() else 'N/A'}):**
    - Walk to nearest bike station: {(st_walk_km if 'st_walk_km' in locals() else 0):.3f} km ({(walk_bike_min if 'walk_bike_min' in locals() else 0):.1f} min)
    - Wait for bike: {(wait_bike if 'wait_bike' in locals() else 0):.1f} min
    - Travel distance: {direct_km:.2f} km
    
    *Wait times estimated from Jan 2024 historical data at hour {hour}:00*
    """)

st.caption("💡 Click the green/red buttons above to change origin/destination on the map. Analysis updates automatically.")
