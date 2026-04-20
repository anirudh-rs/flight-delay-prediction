# src/04_train_model.py
# Phase 4 - Model Training (Updated)
# Trains two models:
#   1. Binary Classifier  — will the flight be delayed 15+ mins? (yes/no)
#   2. Bracket Classifier — which delay bracket? (on time/short/medium/long)
# Both models saved to models/ as .pkl files

import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

def log(msg):
    print(f"  >>> {msg}")

# ── FEATURES ──────────────────────────────────────────────────────────────────

FEATURES = [
    'dep_hour',
    'day_of_week',
    'month',
    'season',
    'day_of_month',
    'distance',
    'crs_elapsed_time',
    'airline_code_enc',
    'origin_enc',
    'dest_enc',
    'airline_delay_rate',
    'route_delay_rate',
    'congestion_score',
    'is_holiday',
    'time_of_day',
    'route_frequency',
    'year',
    'temperature',
    'precipitation',
    'windspeed',
    'cloudcover',
    'weathercode',
    'is_severe_weather'
]

CLF_TARGET      = 'is_delayed'       # Binary: 0 or 1
BRACKET_TARGET  = 'delay_bracket'    # Multi-class: 0, 1, 2, 3

BRACKET_LABELS  = {
    0: 'On Time (<15 mins)',
    1: 'Short Delay (15-45 mins)',
    2: 'Medium Delay (45-120 mins)',
    3: 'Long Delay (120+ mins)'
}

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_features():
    log("Loading feature matrix ...")
    df = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'features_weather.csv'),
        low_memory=False
    )
    log(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

# ── TRAIN BINARY CLASSIFIER ───────────────────────────────────────────────────

def train_binary_classifier(X_train, X_test, y_train, y_test):
    print("\n  ── Binary Classifier (Delayed Yes/No) ────")
    log("Fitting XGBoost binary classifier ...")

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print(f"\n  Binary Classifier Results:")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")

    return clf

# ── TRAIN BRACKET CLASSIFIER ──────────────────────────────────────────────────

def train_bracket_classifier(X_train, X_test, y_train, y_test):
    print("\n  ── Bracket Classifier (Delay Severity) ───")
    log("Fitting XGBoost bracket classifier ...")

    # Compute class weights to handle imbalance
    class_counts = y_train.value_counts().sort_index()
    total        = len(y_train)
    class_weights = {cls: total / (len(class_counts) * count)
                     for cls, count in class_counts.items()}
    sample_weights = y_train.map(class_weights)

    bracket_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        num_class=4,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        verbosity=0
    )

    bracket_clf.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = bracket_clf.predict(X_test)

    print(f"\n  Bracket Classifier Results:")
    print(f"  Overall Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Weighted F1      : {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"  Macro F1         : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"\n  Per-Bracket Breakdown:")
    print(classification_report(
        y_test, y_pred,
        target_names=[BRACKET_LABELS[i] for i in range(4)],
        digits=3
    ))

    return bracket_clf

# ── SAVE MODELS ───────────────────────────────────────────────────────────────

def save_model(model, filename):
    path = os.path.join(MODELS_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    log(f"Saved {filename} ✅")

def save_feature_list():
    path = os.path.join(MODELS_DIR, 'feature_columns.pkl')
    with open(path, 'wb') as f:
        pickle.dump(FEATURES, f)
    log(f"Saved feature_columns.pkl ✅")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n========================================")
    print(" MODEL TRAINING")
    print("========================================")

    df = load_features()

    # Verify all feature columns exist
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"\n  ERROR: Missing columns: {missing}")
        sys.exit(1)

    X       = df[FEATURES]
    y_clf   = df[CLF_TARGET]
    y_brk   = df[BRACKET_TARGET]

    log("Splitting data 80/20 train/test ...")
    X_train, X_test, y_clf_train, y_clf_test = train_test_split(
        X, y_clf,
        test_size=0.2,
        random_state=42,
        stratify=y_clf
    )

    # Align bracket targets to same split
    y_brk_train = y_brk.loc[y_clf_train.index]
    y_brk_test  = y_brk.loc[y_clf_test.index]

    print(f"\n  Train set : {len(X_train):,} rows")
    print(f"  Test set  : {len(X_test):,} rows")

    # Train both models
    clf     = train_binary_classifier(X_train, X_test, y_clf_train, y_clf_test)
    bracket = train_bracket_classifier(X_train, X_test, y_brk_train, y_brk_test)

    # Save everything
    print("\n========================================")
    print(" SAVING MODELS")
    print("========================================")
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_model(clf,     'classifier.pkl')
    save_model(bracket, 'bracket_classifier.pkl')
    save_feature_list()

    print("\n========================================")
    print(" TRAINING COMPLETE")
    print("========================================")
    print("\nModels saved to models/")
    print("Ready for Phase 5 - SHAP Analysis & Exports.")