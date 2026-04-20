# src/05_shap_analysis.py
# Phase 5 - SHAP Analysis
# Generates feature importance using SHAP values
# Saves results to PostgreSQL and exports/ for Tableau

import pandas as pd
import numpy as np
import os
import sys
import pickle
import shap
from sqlalchemy import create_engine, text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DB_URL, PROCESSED_DATA_DIR, MODELS_DIR, EXPORTS_DIR

def log(msg):
    print(f"  >>> {msg}")

FEATURE_LABELS = {
    'dep_hour':            'Departure Hour',
    'day_of_week':         'Day of Week',
    'month':               'Month',
    'season':              'Season',
    'day_of_month':        'Day of Month',
    'distance':            'Flight Distance',
    'crs_elapsed_time':    'Scheduled Flight Duration',
    'airline_code_enc':    'Airline',
    'origin_enc':          'Origin Airport',
    'dest_enc':            'Destination Airport',
    'airline_delay_rate':  'Airline Delay Rate',
    'route_delay_rate':    'Route Delay Rate',
    'congestion_score':    'Airport Congestion',
    'is_holiday':          'Holiday Period',
    'time_of_day':         'Time of Day',
    'route_frequency':     'Route Frequency',
    'year':                'Year',
    'temperature':         'Temperature',
    'precipitation':       'Precipitation',
    'windspeed':           'Wind Speed',
    'cloudcover':          'Cloud Cover',
    'weathercode':         'Weather Condition',
    'is_severe_weather':   'Severe Weather Flag'
}

# ── LOAD MODEL AND DATA ───────────────────────────────────────────────────────

def load_model_and_data():
    log("Loading binary classifier ...")
    with open(os.path.join(MODELS_DIR, 'classifier.pkl'), 'rb') as f:
        clf = pickle.load(f)

    log("Loading feature columns ...")
    with open(os.path.join(MODELS_DIR, 'feature_columns.pkl'), 'rb') as f:
        features = pickle.load(f)

    log("Loading feature matrix (sampling 50,000 rows for SHAP) ...")
    df = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'features_weather.csv'),
        low_memory=False
    )

    # Sample for SHAP — full dataset would take too long
    sample = df[features].sample(n=50000, random_state=42)
    log(f"Sample shape: {sample.shape}")

    return clf, features, sample

# ── COMPUTE SHAP VALUES ───────────────────────────────────────────────────────

def compute_shap(clf, sample):
    log("Computing SHAP values (this may take 3-5 minutes) ...")
    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(sample)
    log("SHAP values computed ✅")
    return shap_values

# ── BUILD IMPORTANCE TABLE ────────────────────────────────────────────────────

def build_importance_table(shap_values, features):
    log("Building feature importance table ...")

    # Mean absolute SHAP value per feature = average impact on prediction
    mean_shap = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame({
        'feature_name':  [FEATURE_LABELS.get(f, f) for f in features],
        'feature_code':  features,
        'shap_value':    mean_shap
    })

    importance = importance.sort_values('shap_value', ascending=False).reset_index(drop=True)
    importance['rank'] = importance.index + 1

    print("\n  ── Top 10 Features Driving Delays ─────────")
    for _, row in importance.head(10).iterrows():
        bar = '█' * int(row['shap_value'] * 200)
        print(f"  {row['rank']:2}. {row['feature_name']:<30} {bar} {row['shap_value']:.4f}")
    print("  ───────────────────────────────────────────")

    return importance

# ── SAVE TO POSTGRESQL ────────────────────────────────────────────────────────

def save_to_postgres(importance, engine):
    log("Saving feature importance to PostgreSQL ...")
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE feature_importance"))
        conn.commit()

    importance[['feature_name', 'shap_value', 'rank']].to_sql(
        'feature_importance',
        engine,
        if_exists='append',
        index=False
    )
    log("Saved to PostgreSQL ✅")

# ── SAVE EXPORT FOR TABLEAU ───────────────────────────────────────────────────

def save_export(importance):
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    path = os.path.join(EXPORTS_DIR, 'shap_importance.csv')
    importance.to_csv(path, index=False)
    log(f"Saved to {path} ✅")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n========================================")
    print(" SHAP ANALYSIS")
    print("========================================")

    engine = create_engine(DB_URL)
    clf, features, sample = load_model_and_data()
    shap_values           = compute_shap(clf, sample)
    importance            = build_importance_table(shap_values, features)

    save_to_postgres(importance, engine)
    save_export(importance)

    print("\n========================================")
    print(" SHAP ANALYSIS COMPLETE")
    print("========================================")
    print("\nFeature importance saved to:")
    print("  - PostgreSQL: feature_importance table")
    print("  - exports/shap_importance.csv")
    print("\nReady for Tableau export script.")