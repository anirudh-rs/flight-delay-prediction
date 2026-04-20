# src/06_export_for_tableau.py
# Phase 5 - Tableau Exports
# Generates all CSV files that feed the Tableau dashboard
# Each CSV is purpose-built for a specific dashboard view

import pandas as pd
import numpy as np
import os
import sys
import pickle
from sqlalchemy import create_engine, text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DB_URL, PROCESSED_DATA_DIR, MODELS_DIR, EXPORTS_DIR

def log(msg):
    print(f"  >>> {msg}")

os.makedirs(EXPORTS_DIR, exist_ok=True)

BRACKET_LABELS = {
    0: 'On Time',
    1: 'Short Delay (15-45 min)',
    2: 'Medium Delay (45-120 min)',
    3: 'Long Delay (120+ min)'
}

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_data(engine):
    log("Loading flights from PostgreSQL ...")
    df = pd.read_sql("SELECT * FROM flights", engine)
    log(f"Loaded {len(df):,} rows")

    log("Loading airports from PostgreSQL ...")
    airports = pd.read_sql("SELECT * FROM airports", engine)

    # Drop pandemic years
    df = df[~df['year'].isin([2020, 2021])].copy()
    log(f"After dropping pandemic years: {len(df):,} rows")

    return df, airports

# ── EXPORT 1: DELAYS BY AIRLINE ───────────────────────────────────────────────

def export_delays_by_airline(df):
    log("Exporting delays by airline ...")
    agg = df.groupby(['airline', 'airline_code', 'year', 'month']).agg(
        total_flights   = ('is_delayed', 'count'),
        delayed_flights = ('is_delayed', 'sum'),
        avg_delay_mins  = ('dep_delay',  'mean'),
        avg_arr_delay   = ('arr_delay',  'mean'),
        carrier_delay   = ('delay_due_carrier',       'mean'),
        weather_delay   = ('delay_due_weather',       'mean'),
        nas_delay       = ('delay_due_nas',           'mean'),
        late_ac_delay   = ('delay_due_late_aircraft', 'mean')
    ).reset_index()

    agg['delay_rate'] = agg['delayed_flights'] / agg['total_flights']
    agg['avg_delay_mins'] = agg['avg_delay_mins'].round(2)

    path = os.path.join(EXPORTS_DIR, 'delays_by_airline.csv')
    agg.to_csv(path, index=False)
    log(f"Saved {len(agg):,} rows to {path} ✅")

# ── EXPORT 2: DELAYS BY ROUTE ─────────────────────────────────────────────────

def export_delays_by_route(df, airports):
    log("Exporting delays by route ...")
    agg = df.groupby(['origin', 'dest']).agg(
        total_flights   = ('is_delayed', 'count'),
        delayed_flights = ('is_delayed', 'sum'),
        avg_delay_mins  = ('dep_delay',  'mean'),
        avg_distance    = ('distance',   'mean')
    ).reset_index()

    agg['delay_rate'] = agg['delayed_flights'] / agg['total_flights']

    # Join origin airport coordinates
    agg = agg.merge(
        airports[['airport_code', 'latitude', 'longitude', 'city', 'state']].rename(
            columns={'airport_code': 'origin', 'latitude': 'origin_lat',
                     'longitude': 'origin_lon', 'city': 'origin_city',
                     'state': 'origin_state'}),
        on='origin', how='left'
    )

    # Join destination airport coordinates
    agg = agg.merge(
        airports[['airport_code', 'latitude', 'longitude', 'city', 'state']].rename(
            columns={'airport_code': 'dest', 'latitude': 'dest_lat',
                     'longitude': 'dest_lon', 'city': 'dest_city',
                     'state': 'dest_state'}),
        on='dest', how='left'
    )

    # Keep routes with meaningful volume
    agg = agg[agg['total_flights'] >= 50].copy()
    agg['avg_delay_mins'] = agg['avg_delay_mins'].round(2)

    path = os.path.join(EXPORTS_DIR, 'delays_by_route.csv')
    agg.to_csv(path, index=False)
    log(f"Saved {len(agg):,} rows to {path} ✅")

# ── EXPORT 3: DELAYS BY AIRPORT ───────────────────────────────────────────────

def export_delays_by_airport(df, airports):
    log("Exporting delays by airport ...")
    agg = df.groupby('origin').agg(
        total_flights   = ('is_delayed', 'count'),
        delayed_flights = ('is_delayed', 'sum'),
        avg_delay_mins  = ('dep_delay',  'mean')
    ).reset_index()

    agg['delay_rate'] = agg['delayed_flights'] / agg['total_flights']

    # Join airport metadata
    agg = agg.merge(
        airports[['airport_code', 'airport_name', 'city', 'state',
                  'latitude', 'longitude', 'airport_type']].rename(
            columns={'airport_code': 'origin'}),
        on='origin', how='left'
    )

    agg['avg_delay_mins'] = agg['avg_delay_mins'].round(2)

    path = os.path.join(EXPORTS_DIR, 'delays_by_airport.csv')
    agg.to_csv(path, index=False)
    log(f"Saved {len(agg):,} rows to {path} ✅")

# ── EXPORT 4: DELAYS BY TIME ──────────────────────────────────────────────────

def export_delays_by_time(df):
    log("Exporting delays by time ...")
    agg = df.groupby(['dep_hour', 'day_of_week', 'month', 'season']).agg(
        total_flights   = ('is_delayed', 'count'),
        delayed_flights = ('is_delayed', 'sum'),
        avg_delay_mins  = ('dep_delay',  'mean')
    ).reset_index()

    agg['delay_rate']     = agg['delayed_flights'] / agg['total_flights']
    agg['avg_delay_mins'] = agg['avg_delay_mins'].round(2)

    # Add readable labels
    day_map    = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday',
                  4:'Friday', 5:'Saturday', 6:'Sunday'}
    season_map = {1:'Winter', 2:'Spring', 3:'Summer', 4:'Fall'}
    month_map  = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                  7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

    agg['day_name']    = agg['day_of_week'].map(day_map)
    agg['season_name'] = agg['season'].map(season_map)
    agg['month_name']  = agg['month'].map(month_map)

    path = os.path.join(EXPORTS_DIR, 'delays_by_time.csv')
    agg.to_csv(path, index=False)
    log(f"Saved {len(agg):,} rows to {path} ✅")

# ── EXPORT 5: DELAY REASONS ───────────────────────────────────────────────────

def export_delay_reasons(df):
    log("Exporting delay reasons ...")
    agg = df.groupby(['airline', 'year']).agg(
        carrier_delay   = ('delay_due_carrier',       'sum'),
        weather_delay   = ('delay_due_weather',       'sum'),
        nas_delay       = ('delay_due_nas',           'sum'),
        security_delay  = ('delay_due_security',      'sum'),
        late_ac_delay   = ('delay_due_late_aircraft', 'sum'),
        total_flights   = ('is_delayed',              'count')
    ).reset_index()

    # Convert to average minutes per flight
    for col in ['carrier_delay', 'weather_delay', 'nas_delay',
                'security_delay', 'late_ac_delay']:
        agg[col] = (agg[col] / agg['total_flights']).round(3)

    path = os.path.join(EXPORTS_DIR, 'delay_reasons.csv')
    agg.to_csv(path, index=False)
    log(f"Saved {len(agg):,} rows to {path} ✅")

# ── EXPORT 6: WEATHER IMPACT ──────────────────────────────────────────────────

def export_weather_impact():
    log("Exporting weather impact ...")
    df = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'features_weather.csv'),
        low_memory=False
    )
    df = df[~df['year'].isin([2020, 2021])].copy()

    # Only use rows with real weather data
    df_weather = df[df['weathercode'] > 0].copy()

    # Bin precipitation into buckets
    df_weather['precip_bucket'] = pd.cut(
        df_weather['precipitation'],
        bins=[-0.01, 0, 0.1, 0.5, 1.0, 999],
        labels=['None', 'Trace', 'Light', 'Moderate', 'Heavy']
    )

    # Bin wind speed
    df_weather['wind_bucket'] = pd.cut(
        df_weather['windspeed'],
        bins=[-1, 10, 20, 30, 999],
        labels=['Calm (<10mph)', 'Breezy (10-20mph)',
                'Windy (20-30mph)', 'Very Windy (30+mph)']
    )

    # Aggregate by precipitation
    precip_agg = df_weather.groupby('precip_bucket', observed=True).agg(
        total_flights   = ('is_delayed', 'count'),
        delayed_flights = ('is_delayed', 'sum'),
        avg_delay_mins  = ('delay_minutes', 'mean')
    ).reset_index()
    precip_agg['delay_rate'] = precip_agg['delayed_flights'] / precip_agg['total_flights']
    precip_agg.rename(columns={'precip_bucket': 'weather_category'}, inplace=True)
    precip_agg['weather_type'] = 'Precipitation'

    # Aggregate by wind
    wind_agg = df_weather.groupby('wind_bucket', observed=True).agg(
        total_flights   = ('is_delayed', 'count'),
        delayed_flights = ('is_delayed', 'sum'),
        avg_delay_mins  = ('delay_minutes', 'mean')
    ).reset_index()
    wind_agg['delay_rate'] = wind_agg['delayed_flights'] / wind_agg['total_flights']
    wind_agg.rename(columns={'wind_bucket': 'weather_category'}, inplace=True)
    wind_agg['weather_type'] = 'Wind Speed'

    weather_impact = pd.concat([precip_agg, wind_agg], ignore_index=True)
    weather_impact['avg_delay_mins'] = weather_impact['avg_delay_mins'].round(2)

    path = os.path.join(EXPORTS_DIR, 'weather_impact.csv')
    weather_impact.to_csv(path, index=False)
    log(f"Saved {len(weather_impact):,} rows to {path} ✅")

# ── EXPORT 7: MODEL PREDICTIONS ───────────────────────────────────────────────

def export_model_predictions(df):
    log("Generating model predictions on test sample ...")

    with open(os.path.join(MODELS_DIR, 'classifier.pkl'), 'rb') as f:
        clf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'bracket_classifier.pkl'), 'rb') as f:
        bracket_clf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'feature_columns.pkl'), 'rb') as f:
        features = pickle.load(f)

    # Load weather features for prediction
    df_weather = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'features_weather.csv'),
        low_memory=False
    )
    df_weather = df_weather[~df_weather['year'].isin([2020, 2021])].copy()

    # Sample 50,000 rows for the predictions export
    sample = df_weather.sample(n=50000, random_state=42)
    X      = sample[features]

    # Generate predictions
    sample = sample.copy()
    sample['is_delayed_predicted']  = clf.predict(X)
    sample['delay_prob']            = clf.predict_proba(X)[:, 1].round(4)
    sample['bracket_predicted']     = bracket_clf.predict(X)
    sample['bracket_label']         = sample['bracket_predicted'].map(BRACKET_LABELS)
    sample['bracket_actual']        = sample['delay_minutes'].apply(
        lambda m: 0 if m < 15 else (1 if m < 45 else (2 if m < 120 else 3))
    )
    sample['bracket_actual_label']  = sample['bracket_actual'].map(BRACKET_LABELS)

    # Select output columns
    output = sample[[
        'fl_date', 'airline', 'origin', 'dest',
        'dep_hour', 'month', 'year', 'is_holiday',
        'dep_delay', 'is_delayed', 'delay_minutes',
        'is_delayed_predicted', 'delay_prob',
        'bracket_predicted', 'bracket_label',
        'bracket_actual', 'bracket_actual_label',
        'temperature', 'precipitation', 'windspeed', 'is_severe_weather'
    ]].copy()

    output['dep_delay']    = output['dep_delay'].round(2)
    output['delay_minutes']= output['delay_minutes'].round(2)

    # Save to PostgreSQL
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE flight_predictions"))
        conn.commit()
    output_db = output[[
        'fl_date', 'airline', 'origin', 'dest', 'dep_hour',
        'dep_delay', 'is_delayed', 'is_delayed_predicted',
        'delay_prob', 'bracket_predicted'
    ]].copy()
    output_db = output_db.loc[:, ~output_db.columns.duplicated()]
    output_db.columns = [
        'fl_date', 'airline', 'origin', 'dest', 'dep_hour',
        'actual_delay', 'is_delayed_actual', 'is_delayed_predicted',
        'delay_prob', 'predicted_minutes'
    ]
    output_db.to_sql('flight_predictions', engine, if_exists='append', index=False)
    log("Predictions saved to PostgreSQL ✅")

    # Save full export for Tableau
    path = os.path.join(EXPORTS_DIR, 'model_predictions.csv')
    output.to_csv(path, index=False)
    log(f"Saved {len(output):,} rows to {path} ✅")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n========================================")
    print(" TABLEAU EXPORTS")
    print("========================================")

    engine = create_engine(DB_URL)
    df, airports = load_data(engine)

    export_delays_by_airline(df)
    export_delays_by_route(df, airports)
    export_delays_by_airport(df, airports)
    export_delays_by_time(df)
    export_delay_reasons(df)
    export_weather_impact()
    export_model_predictions(df)

    print("\n========================================")
    print(" ALL EXPORTS COMPLETE")
    print("========================================")
    print("\nFiles saved to exports/:")
    for f in sorted(os.listdir(EXPORTS_DIR)):
        size = os.path.getsize(os.path.join(EXPORTS_DIR, f))
        print(f"  {f:<35} {size/1024:.1f} KB")
    print("\nReady for Phase 6 - Tableau Dashboard.")