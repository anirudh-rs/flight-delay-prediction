# src/05_fetch_weather.py
# Fetches historical hourly weather from Open-Meteo API
# One API call per airport covering the full date range
# Joins weather to flights on origin airport + date

import pandas as pd
import numpy as np
import os
import sys
import time
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_DATA_DIR

def log(msg):
    print(f"  >>> {msg}")

# ── CONFIG ────────────────────────────────────────────────────────────────────

TOP_N_AIRPORTS  = 150       # Fetch weather for top N busiest airports
API_PAUSE       = 1.5       # Seconds between API calls — be polite to free API
START_DATE      = "2019-01-01"
END_DATE        = "2023-08-31"

# Weather variables to fetch from Open-Meteo
WEATHER_VARS = [
    "temperature_2m",
    "precipitation",
    "windspeed_10m",
    "cloudcover",
    "weathercode"
]

# ── LOAD AIRPORT COORDINATES ──────────────────────────────────────────────────

def get_top_airports(n):
    log("Loading features to find top airports ...")
    df = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'features.csv'),
        usecols=['origin']
    )
    top_airports = df['origin'].value_counts().head(n).index.tolist()

    log("Loading airport coordinates ...")
    airports = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'airports_clean.csv')
    )
    airports = airports[airports['airport_code'].isin(top_airports)].copy()
    airports = airports[['airport_code', 'latitude', 'longitude']].dropna()

    log(f"Found coordinates for {len(airports)} of top {n} airports")
    return airports

# ── FETCH WEATHER FOR ONE AIRPORT ─────────────────────────────────────────────

def fetch_weather_for_airport(airport_code, lat, lon, retries=3):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      START_DATE,
        "end_date":        END_DATE,
        "hourly":          ",".join(WEATHER_VARS),
        "timezone":        "America/New_York",
        "wind_speed_unit": "mph"
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=30)

            # If rate limited, wait longer and retry
            if response.status_code == 429:
                wait = 60 * (attempt + 1)  # 60s, 120s, 180s
                log(f"  Rate limited on {airport_code} — waiting {wait}s before retry {attempt+1}/{retries} ...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            hourly = data.get('hourly', {})
            if not hourly:
                return None

            weather_df = pd.DataFrame({
                'datetime':      hourly['time'],
                'temperature':   hourly['temperature_2m'],
                'precipitation': hourly['precipitation'],
                'windspeed':     hourly['windspeed_10m'],
                'cloudcover':    hourly['cloudcover'],
                'weathercode':   hourly['weathercode']
            })

            weather_df['datetime']     = pd.to_datetime(weather_df['datetime'])
            weather_df['fl_date']      = weather_df['datetime'].dt.date.astype(str)
            weather_df['dep_hour']     = weather_df['datetime'].dt.hour
            weather_df['airport_code'] = airport_code
            weather_df = weather_df.drop(columns=['datetime'])

            return weather_df

        except Exception as e:
            if attempt < retries - 1:
                log(f"  ERROR on {airport_code} attempt {attempt+1} — {e} — retrying ...")
                time.sleep(30)
            else:
                log(f"  WARNING: Failed for {airport_code} after {retries} attempts — {e}")
                return None
# ── FETCH ALL AIRPORTS ────────────────────────────────────────────────────────

def fetch_all_weather(airports_df):
    all_weather = []
    total       = len(airports_df)
    failed      = []

    log(f"Starting weather fetch for {total} airports ...")
    log(f"Estimated time: {total * API_PAUSE / 60:.1f}–{total * 0.8 / 60:.1f} minutes")
    print()

    for i, row in airports_df.iterrows():
        code = row['airport_code']
        lat  = row['latitude']
        lon  = row['longitude']

        print(f"  Fetching {code} ({airports_df.index.get_loc(i)+1}/{total}) ...", end='\r')

        weather = fetch_weather_for_airport(code, lat, lon)

        if weather is not None:
            all_weather.append(weather)
        else:
            failed.append(code)

        time.sleep(API_PAUSE)

    print()  # newline after progress
    log(f"Successfully fetched: {len(all_weather)} airports")

    if failed:
        log(f"Failed airports ({len(failed)}): {failed}")

    if not all_weather:
        log("ERROR: No weather data fetched at all.")
        return None

    combined = pd.concat(all_weather, ignore_index=True)
    log(f"Combined weather rows: {len(combined):,}")
    return combined

# ── AGGREGATE TO DAILY DEPARTURE HOUR ────────────────────────────────────────
# Each flight has a departure hour — we match weather at that exact hour

def aggregate_weather(weather_df):
    log("Aggregating weather by airport + date + hour ...")

    # Weather code — take the worst (highest) code per hour window
    # WMO weather codes: higher = worse (95+ = thunderstorm, 71+ = snow etc.)
    agg = weather_df.groupby(['airport_code', 'fl_date', 'dep_hour']).agg(
        temperature   = ('temperature',   'mean'),
        precipitation = ('precipitation', 'sum'),
        windspeed     = ('windspeed',     'max'),
        cloudcover    = ('cloudcover',    'mean'),
        weathercode   = ('weathercode',   'max')
    ).reset_index()

    # Add a severe weather flag — WMO codes 51+ indicate significant weather
    # 51-67: Rain, 71-77: Snow, 80-82: Showers, 85-86: Snow showers, 95+: Thunderstorm
    agg['is_severe_weather'] = (agg['weathercode'] >= 51).astype(int)

    log(f"Aggregated weather rows: {len(agg):,}")
    return agg

# ── JOIN WEATHER TO FEATURES ──────────────────────────────────────────────────

def join_weather_to_features(weather_agg):
    log("Loading feature matrix ...")
    features = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'features.csv'),
        low_memory=False
    )
    features['fl_date'] = pd.to_datetime(features['fl_date']).dt.date.astype(str)

    log("Joining weather to features on origin + date + hour ...")
    merged = features.merge(
        weather_agg.rename(columns={'airport_code': 'origin'}),
        on=['origin', 'fl_date', 'dep_hour'],
        how='left'
    )

    # Fill missing weather with sensible defaults (clear conditions)
    weather_cols = ['temperature', 'precipitation', 'windspeed',
                    'cloudcover', 'weathercode', 'is_severe_weather']
    defaults = {
        'temperature':    15.0,   # mild temperature
        'precipitation':  0.0,    # no rain
        'windspeed':      8.0,    # light wind
        'cloudcover':     25.0,   # mostly clear
        'weathercode':    0.0,    # clear sky
        'is_severe_weather': 0    # no severe weather
    }
    for col in weather_cols:
        merged[col] = merged[col].fillna(defaults[col])

    covered     = merged['precipitation'].notna().sum()
    total       = len(merged)
    fill_rate   = (merged['weathercode'] != 0.0).sum() / total * 100

    log(f"Weather join complete")
    log(f"Rows with real weather data: {fill_rate:.1f}%")
    log(f"Total rows: {total:,}")

    return merged

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n========================================")
    print(" WEATHER DATA FETCH (RESUME MODE)")
    print("========================================")

    # Step 1 — get top airports with coordinates
    airports_df = get_top_airports(TOP_N_AIRPORTS)

    # Step 2 — check which airports already have data
    raw_path = os.path.join(PROCESSED_DATA_DIR, 'weather_raw.csv')

    if os.path.exists(raw_path):
        log("Found existing weather_raw.csv — loading already fetched airports ...")
        existing = pd.read_csv(raw_path)
        already_fetched = existing['airport_code'].unique().tolist()
        log(f"Already have data for {len(already_fetched)} airports: {already_fetched}")

        # Only fetch airports we don't have yet
        airports_to_fetch = airports_df[
            ~airports_df['airport_code'].isin(already_fetched)
        ].copy()
        log(f"Remaining to fetch: {len(airports_to_fetch)} airports")
    else:
        existing = None
        airports_to_fetch = airports_df
        log(f"No existing data — fetching all {len(airports_to_fetch)} airports")

    # Step 3 — fetch missing airports
    if len(airports_to_fetch) > 0:
        weather_new = fetch_all_weather(airports_to_fetch)

        if weather_new is not None:
            # Combine with existing data if we have any
            if existing is not None:
                weather_raw = pd.concat([existing, weather_new], ignore_index=True)
                log(f"Combined existing + new weather rows: {len(weather_raw):,}")
            else:
                weather_raw = weather_new

            # Save combined raw weather
            weather_raw.to_csv(raw_path, index=False)
            log(f"Raw weather saved to {raw_path}")
        else:
            log("No new weather fetched — using existing data only")
            weather_raw = existing
    else:
        log("All airports already fetched — using existing data")
        weather_raw = existing

    # Step 4 — aggregate to hour level
    weather_agg = aggregate_weather(weather_raw)

    # Step 5 — join to feature matrix
    features_with_weather = join_weather_to_features(weather_agg)

    # Step 6 — save enriched feature matrix
    out_path = os.path.join(PROCESSED_DATA_DIR, 'features.csv')
    features_with_weather.to_csv(out_path, index=False)
    log(f"Enriched feature matrix saved to {out_path}")

    print("\n========================================")
    print(" WEATHER FETCH COMPLETE")
    print("========================================")
    print(f"\nNew feature matrix shape: {features_with_weather.shape}")
    print(f"Airports with real weather: {weather_raw['airport_code'].nunique()}")
    print("\nReady to retrain model with weather features.")