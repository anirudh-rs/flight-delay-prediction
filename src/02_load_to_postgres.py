# src/02_load_to_postgres.py
# Phase 3 - Load cleaned CSVs into PostgreSQL
# Loads airports and flights into their respective tables
# Uses chunked loading for the large flights file

import pandas as pd
import os
import sys
from sqlalchemy import create_engine, text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DB_URL, PROCESSED_DATA_DIR

# ── HELPER ────────────────────────────────────────────────────────────────────

def log(msg):
    print(f"  >>> {msg}")

# ── LOAD AIRPORTS ─────────────────────────────────────────────────────────────

def load_airports(engine):
    log("Loading airports_clean.csv ...")
    airports = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'airports_clean.csv'))
    log(f"Rows to load: {len(airports):,}")

    # Clear existing data and load fresh
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE airports CASCADE"))
        conn.commit()

    airports.to_sql(
        'airports',
        engine,
        if_exists='append',
        index=False
    )
    log(f"Airports loaded successfully ✅")

# ── LOAD FLIGHTS ──────────────────────────────────────────────────────────────

def load_flights(engine):
    log("Loading flights_clean.csv in chunks (this will take 3-8 minutes) ...")

    # Clear existing data first
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE flights CASCADE"))
        conn.commit()
    log("Existing flights table cleared.")

    chunk_size = 100_000
    total_loaded = 0
    chunk_num = 0

    for chunk in pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'flights_clean.csv'),
        chunksize=chunk_size,
        low_memory=False,
        parse_dates=['fl_date']
    ):
        chunk_num += 1

        # Fix integer columns that may have loaded as float due to NaNs
        int_cols = ['year', 'month', 'day_of_week', 'day_of_month',
                    'season', 'is_delayed']
        for col in int_cols:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0).astype(int)

        # dep_hour uses nullable Int64 — convert to regular int for postgres
        if 'dep_hour' in chunk.columns:
            chunk['dep_hour'] = pd.to_numeric(
                chunk['dep_hour'], errors='coerce'
            ).fillna(0).astype(int)

        chunk.to_sql(
            'flights',
            engine,
            if_exists='append',
            index=False,
            method='multi'
        )

        total_loaded += len(chunk)
        print(f"  Chunk {chunk_num}: {total_loaded:,} rows loaded...", end='\r')

    print()  # newline after progress
    log(f"All flights loaded: {total_loaded:,} rows ✅")

# ── VERIFY ────────────────────────────────────────────────────────────────────

def verify(engine):
    log("Verifying row counts in database ...")
    with engine.connect() as conn:
        airports_count = conn.execute(text("SELECT COUNT(*) FROM airports")).scalar()
        flights_count  = conn.execute(text("SELECT COUNT(*) FROM flights")).scalar()
        delayed_count  = conn.execute(
            text("SELECT COUNT(*) FROM flights WHERE is_delayed = 1")
        ).scalar()
        airlines       = conn.execute(
            text("SELECT COUNT(DISTINCT airline_code) FROM flights")
        ).scalar()

    print("\n  ── Database Verification ──────────────────")
    print(f"  Airports : {airports_count:,}")
    print(f"  Flights  : {flights_count:,}")
    print(f"  Delayed  : {delayed_count:,} ({delayed_count/flights_count*100:.1f}%)")
    print(f"  Airlines : {airlines}")
    print("  ───────────────────────────────────────────")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n========================================")
    print(" CONNECTING TO POSTGRESQL")
    print("========================================")

    engine = create_engine(DB_URL)

    # Test connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log("Connection successful ✅")
    except Exception as e:
        print(f"\n  ERROR: Could not connect to database.")
        print(f"  Check your DB_CONFIG in src/config.py")
        print(f"  Details: {e}")
        sys.exit(1)

    print("\n========================================")
    print(" LOADING AIRPORTS")
    print("========================================")
    load_airports(engine)

    print("\n========================================")
    print(" LOADING FLIGHTS")
    print("========================================")
    load_flights(engine)

    print("\n========================================")
    print(" VERIFYING")
    print("========================================")
    verify(engine)

    print("\n========================================")
    print(" PHASE 3 COMPLETE")
    print("========================================")
    print("\nData is now live in PostgreSQL.")
    print("Ready for Phase 4 - Feature Engineering & Modelling.")