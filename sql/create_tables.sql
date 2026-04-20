-- sql/create_tables.sql
-- Defines all tables for the flight delay prediction project
-- Run this once to set up the schema before loading data

-- Drop tables if they already exist (safe to re-run)
DROP TABLE IF EXISTS flights CASCADE;
DROP TABLE IF EXISTS airports CASCADE;
DROP TABLE IF EXISTS flight_predictions CASCADE;
DROP TABLE IF EXISTS feature_importance CASCADE;

-- ── AIRPORTS TABLE ────────────────────────────────────────────────────────────
-- Static reference table for airport metadata
-- Joined to flights on origin/dest = airport_code

CREATE TABLE airports (
    airport_code        VARCHAR(10) PRIMARY KEY,
    airport_name        VARCHAR(255),
    city                VARCHAR(100),
    state               VARCHAR(10),
    latitude            FLOAT,
    longitude           FLOAT,
    elevation_ft        FLOAT,
    airport_type        VARCHAR(50)
);

-- ── FLIGHTS TABLE ─────────────────────────────────────────────────────────────
-- Core table — cleaned flight records from BTS/Kaggle
-- One row per flight that operated (cancelled/diverted excluded)

CREATE TABLE flights (
    fl_date             DATE,
    airline             VARCHAR(100),
    airline_dot         VARCHAR(100),
    airline_code        VARCHAR(10),
    dot_code            INTEGER,
    fl_number           INTEGER,
    origin              VARCHAR(10),
    origin_city         VARCHAR(100),
    dest                VARCHAR(10),
    dest_city           VARCHAR(100),
    crs_dep_time        FLOAT,
    dep_time            FLOAT,
    dep_delay           FLOAT,
    crs_arr_time        FLOAT,
    arr_time            FLOAT,
    arr_delay           FLOAT,
    crs_elapsed_time    FLOAT,
    elapsed_time        FLOAT,
    air_time            FLOAT,
    distance            FLOAT,
    delay_due_carrier   FLOAT,
    delay_due_weather   FLOAT,
    delay_due_nas       FLOAT,
    delay_due_security  FLOAT,
    delay_due_late_aircraft FLOAT,
    year                INTEGER,
    month               INTEGER,
    day_of_week         INTEGER,
    day_of_month        INTEGER,
    season              INTEGER,
    dep_hour            INTEGER,
    is_delayed          INTEGER,
    delay_minutes       FLOAT
);

-- ── FLIGHT PREDICTIONS TABLE ──────────────────────────────────────────────────
-- Stores model predictions written back from Python after training
-- Used by Tableau to compare predicted vs actual

CREATE TABLE flight_predictions (
    fl_date             DATE,
    airline             VARCHAR(100),
    origin              VARCHAR(10),
    dest                VARCHAR(10),
    dep_hour            INTEGER,
    actual_delay        FLOAT,
    is_delayed_actual   INTEGER,
    is_delayed_predicted INTEGER,
    delay_prob          FLOAT,
    predicted_minutes   FLOAT
);

-- ── FEATURE IMPORTANCE TABLE ──────────────────────────────────────────────────
-- Stores SHAP values exported from Python after model training
-- Used by Tableau for the feature importance chart

CREATE TABLE feature_importance (
    feature_name        VARCHAR(100),
    shap_value          FLOAT,
    rank                INTEGER
);

-- ── INDEXES ───────────────────────────────────────────────────────────────────
-- Speed up common query patterns we'll use in feature engineering

CREATE INDEX idx_flights_date        ON flights(fl_date);
CREATE INDEX idx_flights_airline     ON flights(airline_code);
CREATE INDEX idx_flights_origin      ON flights(origin);
CREATE INDEX idx_flights_dest        ON flights(dest);
CREATE INDEX idx_flights_is_delayed  ON flights(is_delayed);
CREATE INDEX idx_flights_year_month  ON flights(year, month);