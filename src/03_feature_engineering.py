# src/03_feature_engineering.py
# Phase 4 - Feature Engineering
# Queries PostgreSQL to build ML-ready features
# Saves a feature matrix as a CSV for model training

import pandas as pd
import numpy as np
import os
import sys
from sqlalchemy import create_engine, text
import holidays

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DB_URL, PROCESSED_DATA_DIR

def log(msg):
    print(f"  >>> {msg}")

# ── LOAD BASE DATA ────────────────────────────────────────────────────────────

def load_base_data(engine):
    log("Loading flights from PostgreSQL ...")
    query = """
        SELECT
            fl_date, airline, airline_code, origin, dest,
            dep_hour, day_of_week, month, year, season,
            day_of_month, distance, crs_elapsed_time,
            delay_due_carrier, delay_due_weather, delay_due_nas,
            delay_due_security, delay_due_late_aircraft,
            dep_delay, arr_delay,
            is_delayed, delay_minutes
        FROM flights
    """
    df = pd.read_sql(query, engine)
    log(f"Loaded {len(df):,} rows")
    return df

# ── FEATURE 1: AIRLINE DELAY RATE ────────────────────────────────────────────
# What percentage of this airline's flights are typically delayed?

def add_airline_delay_rate(df):
    log("Computing airline delay rates ...")
    rate = df.groupby('airline_code')['is_delayed'].mean().reset_index()
    rate.columns = ['airline_code', 'airline_delay_rate']
    df = df.merge(rate, on='airline_code', how='left')
    return df

# ── FEATURE 2: ROUTE DELAY RATE ───────────────────────────────────────────────
# What percentage of flights on this specific route are typically delayed?

def add_route_delay_rate(df):
    log("Computing route delay rates ...")
    rate = df.groupby(['origin', 'dest'])['is_delayed'].mean().reset_index()
    rate.columns = ['origin', 'dest', 'route_delay_rate']
    df = df.merge(rate, on=['origin', 'dest'], how='left')
    # Fill unknown routes with global mean
    df['route_delay_rate'] = df['route_delay_rate'].fillna(df['is_delayed'].mean())
    return df

# ── FEATURE 3: ORIGIN AIRPORT CONGESTION SCORE ───────────────────────────────
# How busy is the departure airport at this hour?
# Measured as average number of departures per hour at this airport

def add_congestion_score(df):
    log("Computing airport congestion scores ...")
    congestion = df.groupby(['origin', 'dep_hour']).size().reset_index()
    congestion.columns = ['origin', 'dep_hour', 'congestion_score']
    # Normalise to 0-1 range
    congestion['congestion_score'] = (
        congestion['congestion_score'] - congestion['congestion_score'].min()
    ) / (congestion['congestion_score'].max() - congestion['congestion_score'].min())
    df = df.merge(congestion, on=['origin', 'dep_hour'], how='left')
    df['congestion_score'] = df['congestion_score'].fillna(0)
    return df

# ── FEATURE 4: HOLIDAY FLAG ───────────────────────────────────────────────────
# Is this flight on or within 2 days of a US public holiday?

def add_holiday_flag(df):
    log("Adding holiday flags ...")
    years = df['year'].unique().tolist()
    us_holidays = set()
    for year in years:
        for date in holidays.US(years=year).keys():
            # Flag the holiday itself plus 2 days either side (travel surge)
            for delta in range(-2, 3):
                us_holidays.add(date + pd.Timedelta(days=delta))

    df['fl_date'] = pd.to_datetime(df['fl_date'])
    df['is_holiday'] = df['fl_date'].dt.date.apply(
        lambda d: 1 if d in us_holidays else 0
    )
    return df

# ── FEATURE 5: TIME OF DAY BUCKET ─────────────────────────────────────────────
# Categorical bucket for time of day — more interpretable than raw hour

def add_time_of_day(df):
    log("Adding time-of-day buckets ...")
    def bucket(hour):
        if 0 <= hour <= 5:   return 0  # Red-eye
        elif 6 <= hour <= 11: return 1  # Morning
        elif 12 <= hour <= 17: return 2  # Afternoon
        elif 18 <= hour <= 21: return 3  # Evening
        else:                  return 4  # Night
    df['time_of_day'] = df['dep_hour'].apply(bucket)
    return df

# ── FEATURE 6: ROUTE FREQUENCY ────────────────────────────────────────────────
# How many times does this route appear in the dataset?
# High frequency = major route = more data = more reliable prediction

def add_route_frequency(df):
    log("Computing route frequencies ...")
    freq = df.groupby(['origin', 'dest']).size().reset_index()
    freq.columns = ['origin', 'dest', 'route_frequency']
    df = df.merge(freq, on=['origin', 'dest'], how='left')
    return df

# ── FEATURE 7: DROP 2020-2021 ─────────────────────────────────────────────────
# Remove pandemic years — distorted delay patterns

def drop_pandemic_years(df):
    before = len(df)
    df = df[~df['year'].isin([2020, 2021])].copy()
    log(f"Dropped pandemic years 2020-2021: removed {before - len(df):,} rows. Remaining: {len(df):,}")
    return df

# ── ENCODE CATEGORICALS ───────────────────────────────────────────────────────
# XGBoost needs numbers — encode airline and airport codes as integers

def encode_categoricals(df):
    log("Encoding categorical columns ...")
    for col in ['airline_code', 'origin', 'dest']:
        df[col + '_enc'] = df[col].astype('category').cat.codes
    return df

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n========================================")
    print(" FEATURE ENGINEERING")
    print("========================================")

    engine = create_engine(DB_URL)

    df = load_base_data(engine)
    df = drop_pandemic_years(df)
    df = add_airline_delay_rate(df)
    df = add_route_delay_rate(df)
    df = add_congestion_score(df)
    df = add_holiday_flag(df)
    df = add_time_of_day(df)
    df = add_route_frequency(df)
    df = encode_categoricals(df)

    print("\n  ── Feature Matrix Summary ─────────────────")
    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {len(df.columns)}")
    print(f"  Delayed      : {df['is_delayed'].sum():,} ({df['is_delayed'].mean()*100:.1f}%)")
    print(f"  Years        : {sorted(df['year'].unique().tolist())}")
    print(f"  Holiday rows : {df['is_holiday'].sum():,}")
    print("  ───────────────────────────────────────────")

    # Save feature matrix
    out_path = os.path.join(PROCESSED_DATA_DIR, 'features.csv')
    df.to_csv(out_path, index=False)
    log(f"Feature matrix saved to {out_path}")

    print("\n========================================")
    print(" FEATURE ENGINEERING COMPLETE")
    print("========================================")
    print("\nReady for model training.")