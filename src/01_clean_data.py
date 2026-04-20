# src/01_clean_data.py
# Phase 2 - Data Cleaning
# Cleans raw flights and airports CSVs and saves processed versions
# ready for loading into PostgreSQL in Phase 3

import pandas as pd
import numpy as np
import os
import sys

# Add project root to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# ── HELPER ───────────────────────────────────────────────────────────────────

def log(msg):
    print(f"  >>> {msg}")

# ── PART 1: CLEAN AIRPORTS ────────────────────────────────────────────────────

def clean_airports():
    log("Loading airports.csv ...")
    airports = pd.read_csv(
        os.path.join(RAW_DATA_DIR, 'airports.csv'),
        low_memory=False
    )
    log(f"Raw shape: {airports.shape}")

    # Keep US airports only
    airports = airports[airports['iso_country'] == 'US'].copy()
    log(f"After US filter: {airports.shape}")

    # Keep only real airports (drop heliports, seaplane bases, closed airports)
    valid_types = ['large_airport', 'medium_airport', 'small_airport']
    airports = airports[airports['type'].isin(valid_types)].copy()
    log(f"After type filter: {airports.shape}")

    # Drop rows with no IATA code (can't join to flights without it)
    airports = airports[airports['iata_code'].notna()].copy()
    airports = airports[airports['iata_code'].str.strip() != ''].copy()
    log(f"After IATA filter: {airports.shape}")

    # Extract state from iso_region (format is 'US-TX', we want 'TX')
    airports['state'] = airports['iso_region'].str.split('-').str[1]

    # Select and rename only the columns we need
    airports = airports[[
        'iata_code', 'name', 'municipality', 'state',
        'latitude_deg', 'longitude_deg', 'elevation_ft', 'type'
    ]].rename(columns={
        'iata_code':      'airport_code',
        'name':           'airport_name',
        'municipality':   'city',
        'latitude_deg':   'latitude',
        'longitude_deg':  'longitude',
        'elevation_ft':   'elevation_ft',
        'type':           'airport_type'
    })

    # Drop any duplicate airport codes (keep first)
    airports = airports.drop_duplicates(subset='airport_code').copy()
    log(f"Final airports shape: {airports.shape}")

    # Save
    out_path = os.path.join(PROCESSED_DATA_DIR, 'airports_clean.csv')
    airports.to_csv(out_path, index=False)
    log(f"Saved to {out_path}")
    return airports


# ── PART 2: CLEAN FLIGHTS ─────────────────────────────────────────────────────

def clean_flights():
    log("Loading flights_sample_3m.csv (this may take 30-60 seconds) ...")
    flights = pd.read_csv(
        os.path.join(RAW_DATA_DIR, 'flights_sample_3m.csv'),
        low_memory=False
    )
    log(f"Raw shape: {flights.shape}")

    # ── 1. Drop cancelled and diverted flights ──────────────────────────────
    # We're predicting delays on flights that actually operated
    before = len(flights)
    flights = flights[
        (flights['CANCELLED'] == 0) &
        (flights['DIVERTED'] == 0)
    ].copy()
    log(f"Dropped {before - len(flights)} cancelled/diverted rows. Remaining: {len(flights)}")

    # ── 2. Parse date and extract time features ─────────────────────────────
    flights['FL_DATE'] = pd.to_datetime(flights['FL_DATE'])
    flights['year']        = flights['FL_DATE'].dt.year
    flights['month']       = flights['FL_DATE'].dt.month
    flights['day_of_week'] = flights['FL_DATE'].dt.dayofweek  # 0=Monday, 6=Sunday
    flights['day_of_month']= flights['FL_DATE'].dt.day

    # Season: 1=Winter, 2=Spring, 3=Summer, 4=Fall
    flights['season'] = flights['month'].map({
        12: 1, 1: 1, 2: 1,
        3: 2,  4: 2, 5: 2,
        6: 3,  7: 3, 8: 3,
        9: 4, 10: 4, 11: 4
    })

    # ── 3. Clean departure time into hour of day ────────────────────────────
    # CRS_DEP_TIME is in HHMM format (e.g. 800 = 8:00am, 1435 = 2:35pm)
    flights['CRS_DEP_TIME'] = pd.to_numeric(flights['CRS_DEP_TIME'], errors='coerce')
    flights['dep_hour'] = (flights['CRS_DEP_TIME'] // 100).astype('Int64')

    # ── 4. Clean airline names ──────────────────────────────────────────────
    # Remove trailing punctuation and whitespace (e.g. "United Air Lines Inc.")
    flights['AIRLINE'] = flights['AIRLINE'].str.replace(r'\.$', '', regex=True).str.strip()

    # ── 5. Handle delay columns ─────────────────────────────────────────────
    # NaN in delay reason columns means no delay from that cause — fill with 0
    delay_cols = [
        'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
        'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT'
    ]
    flights[delay_cols] = flights[delay_cols].fillna(0)

    # Fill DEP_DELAY and ARR_DELAY NaNs with 0 (on-time)
    flights['DEP_DELAY'] = pd.to_numeric(flights['DEP_DELAY'], errors='coerce').fillna(0)
    flights['ARR_DELAY'] = pd.to_numeric(flights['ARR_DELAY'], errors='coerce').fillna(0)

    # ── 6. Create target columns for modelling ──────────────────────────────
    # Binary classification target: 1 = delayed by 15+ minutes, 0 = on time
    flights['IS_DELAYED'] = (flights['DEP_DELAY'] >= 15).astype(int)

    # Regression target: how many minutes delayed (floor at 0 — ignore early departures)
    flights['DELAY_MINUTES'] = flights['DEP_DELAY'].clip(lower=0)

    # ── 7. Drop rows missing critical fields ────────────────────────────────
    critical_cols = ['ORIGIN', 'DEST', 'AIRLINE_CODE', 'DEP_DELAY', 'FL_DATE']
    before = len(flights)
    flights = flights.dropna(subset=critical_cols).copy()
    log(f"Dropped {before - len(flights)} rows missing critical fields. Remaining: {len(flights)}")

    # ── 8. Drop columns we no longer need ───────────────────────────────────
    drop_cols = [
        'CANCELLED', 'DIVERTED', 'CANCELLATION_CODE',
        'WHEELS_OFF', 'WHEELS_ON', 'TAXI_OUT', 'TAXI_IN'
    ]
    flights = flights.drop(columns=drop_cols)

    # ── 9. Rename columns to lowercase for PostgreSQL compatibility ─────────
    flights.columns = flights.columns.str.lower()

    log(f"Final flights shape: {flights.shape}")
    log(f"Delayed flights: {flights['is_delayed'].sum():,} ({flights['is_delayed'].mean()*100:.1f}%)")
    log(f"Date range: {flights['fl_date'].min()} to {flights['fl_date'].max()}")
    log(f"Airlines: {flights['airline'].nunique()}")
    log(f"Airports: {flights['origin'].nunique()} unique origins")

    # Save
    out_path = os.path.join(PROCESSED_DATA_DIR, 'flights_clean.csv')
    flights.to_csv(out_path, index=False)
    log(f"Saved to {out_path}")
    return flights


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n========================================")
    print(" CLEANING AIRPORTS")
    print("========================================")
    airports = clean_airports()

    print("\n========================================")
    print(" CLEANING FLIGHTS")
    print("========================================")
    flights = clean_flights()

    print("\n========================================")
    print(" ALL DONE")
    print("========================================")
    print(f"\nAirports clean: {len(airports):,} rows")
    print(f"Flights clean:  {len(flights):,} rows")
    print("\nProcessed files saved to data/processed/")