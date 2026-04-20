# src/config.py
from urllib.parse import quote_plus

# Your raw password here — @ and other special characters are safe in a plain string
_password = "flightdelay123"  # ← keep your actual password here

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "flight_delays",
    "user": "postgres",
    "password": _password
}

# quote_plus encodes @ as %40 — makes the URL safe
DB_URL = f"postgresql+psycopg2://postgres:{quote_plus(_password)}@localhost:5432/flight_delays"

# Paths — used by all src/ scripts
RAW_DATA_DIR       = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
EXPORTS_DIR        = "exports"
MODELS_DIR         = "models"