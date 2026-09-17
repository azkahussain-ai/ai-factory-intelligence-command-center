"""Focused tests for Stage I: generation, cleaning, leakage safety, and split integrity.

Run with: pytest tests/test_stage1.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import config
from src.data.generate_synthetic import generate_sensor_timeseries, generate_production_data
from src.data.validate_schema import SchemaValidationError, validate_schema, SENSOR_SCHEMA
from src.preprocessing.clean_data import clean_sensor_data, clean_production_data
from src.preprocessing.integrate_data import integrate_datasets
from src.features.build_features import build_feature_set
from src.features.split_data import time_based_split


@pytest.fixture(scope="module")
def raw_sensor_and_episodes():
    return generate_sensor_timeseries()


@pytest.fixture(scope="module")
def raw_production(raw_sensor_and_episodes):
    _, episodes = raw_sensor_and_episodes
    return generate_production_data(episodes)


def test_sensor_generation_has_expected_machines(raw_sensor_and_episodes):
    sensor_df, _ = raw_sensor_and_episodes
    assert set(sensor_df["machine_id"].unique()) >= set(m["machine_id"] for m in config.MACHINE_FLEET)


def test_sensor_generation_reproducible():
    df_a, _ = generate_sensor_timeseries()
    df_b, _ = generate_sensor_timeseries()
    pd.testing.assert_frame_equal(df_a, df_b)


def test_raw_sensor_has_injected_quality_issues(raw_sensor_and_episodes):
    sensor_df, _ = raw_sensor_and_episodes
    assert sensor_df.duplicated().sum() > 0
    assert (sensor_df["timestamp"] == "unknown").sum() > 0
    assert sensor_df["machine_type"].nunique() > 3  # inconsistent casing present


def test_schema_validation_rejects_missing_columns():
    bad_df = pd.DataFrame({"machine_id": ["M-01"]})
    with pytest.raises(SchemaValidationError):
        validate_schema(bad_df, SENSOR_SCHEMA, "bad")


def test_clean_sensor_data_removes_duplicates_and_malformed_timestamps(raw_sensor_and_episodes):
    sensor_df, _ = raw_sensor_and_episodes
    cleaned, summary = clean_sensor_data(sensor_df)
    assert cleaned.duplicated().sum() == 0
    assert cleaned["timestamp"].isna().sum() == 0
    assert summary["rows_dropped_malformed_timestamp"] > 0
    assert summary["rows_dropped_exact_duplicate"] > 0


def test_clean_sensor_data_removes_physically_impossible_values(raw_sensor_and_episodes):
    sensor_df, _ = raw_sensor_and_episodes
    cleaned, _ = clean_sensor_data(sensor_df)
    assert cleaned["rotational_speed_rpm"].max() < 5000
    assert (cleaned["vibration_mm_s"].dropna() >= 0).all()
    assert (cleaned["temperature_k"].dropna() > 250).all()


def test_clean_sensor_preserves_down_state_missingness(raw_sensor_and_episodes):
    sensor_df, _ = raw_sensor_and_episodes
    cleaned, summary = clean_sensor_data(sensor_df)
    down_rows = cleaned[cleaned["operating_status"] == "DOWN"]
    assert down_rows["vibration_mm_s"].isna().sum() > 0
    assert summary["down_state_missing_preserved"] > 0


def test_clean_production_fixes_invalid_counts(raw_sensor_and_episodes, raw_production):
    cleaned, summary = clean_production_data(raw_production)
    assert (cleaned["production_count"] >= 0).all()
    assert summary["invalid_production_count_fixed"] > 0


def test_integration_does_not_multiply_rows(raw_sensor_and_episodes, raw_production):
    sensor_df, _ = raw_sensor_and_episodes
    sensor_clean, _ = clean_sensor_data(sensor_df)
    production_clean, _ = clean_production_data(raw_production)
    from src.preprocessing.clean_data import clean_maintenance_data
    from src.data.generate_synthetic import generate_maintenance_notes
    _, episodes = generate_sensor_timeseries()
    maintenance_raw = generate_maintenance_notes(episodes)
    maintenance_clean, _ = clean_maintenance_data(maintenance_raw)

    integrated, summary = integrate_datasets(production_clean, sensor_clean, maintenance_clean)
    # one row per (machine, date, shift) - same cardinality as production data
    assert len(integrated) == len(production_clean)


def test_lag_features_do_not_leak_future_information():
    df = pd.DataFrame({
        "machine_id": ["M-01"] * 5,
        "date": pd.date_range("2026-05-01", periods=5),
        "shift": ["Morning"] * 5,
        "quality_score": [90.0, 80.0, 70.0, 60.0, 50.0],
        "production_rate": [10.0, 10.0, 10.0, 10.0, 10.0],
        "production_count": [80, 80, 80, 80, 80],
        "downtime_minutes": [0.0] * 5,
        "downtime_ratio": [0.0] * 5,
        "production_efficiency": [1.0] * 5,
        "failure_occurred_this_shift": [0, 0, 0, 0, 1],
        "maintenance_count_past_7d": [0] * 5,
        "temperature_k_mean": [300.0] * 5,
        "process_temperature_k_mean": [310.0] * 5,
        "vibration_mm_s_mean": [1.5] * 5,
        "pressure_bar_mean": [5.0] * 5,
        "torque_nm_mean": [40.0] * 5,
        "rotational_speed_rpm_mean": [1500.0] * 5,
        "load_pct_mean": [50.0] * 5,
    })
    feature_df, _ = build_feature_set(df)
    # lag1 quality_score at row t must equal quality_score at row t-1, never t or later.
    row = feature_df[feature_df["quality_score"] == 70.0].iloc[0]
    assert row["quality_score_lag1"] == 80.0
    # first row of the series must have no lag value (nothing precedes it)
    first_row = feature_df.sort_values("date").iloc[0]
    assert pd.isna(first_row["quality_score_lag1"])


def test_failure_next_shift_is_shifted_forward_correctly():
    df = pd.DataFrame({
        "machine_id": ["M-01"] * 3,
        "date": pd.date_range("2026-05-01", periods=3),
        "shift": ["Morning"] * 3,
        "quality_score": [90.0, 90.0, 90.0],
        "production_rate": [10.0] * 3,
        "production_count": [80, 80, 80],
        "downtime_minutes": [0.0] * 3,
        "downtime_ratio": [0.0] * 3,
        "production_efficiency": [1.0] * 3,
        "failure_occurred_this_shift": [0, 1, 0],
        "maintenance_count_past_7d": [0] * 3,
        "temperature_k_mean": [300.0] * 3,
        "process_temperature_k_mean": [310.0] * 3,
        "vibration_mm_s_mean": [1.5] * 3,
        "pressure_bar_mean": [5.0] * 3,
        "torque_nm_mean": [40.0] * 3,
        "rotational_speed_rpm_mean": [1500.0] * 3,
        "load_pct_mean": [50.0] * 3,
    })
    feature_df, summary = build_feature_set(df)
    # row for day 1 should have failure_next_shift == 1 (day 2 fails)
    day1 = feature_df[feature_df["date"] == pd.Timestamp("2026-05-01")].iloc[0]
    assert day1["failure_next_shift"] == 1
    # the last day is dropped since it has no future shift to know
    assert pd.Timestamp("2026-05-03") not in feature_df["date"].values
    assert summary["rows_dropped_no_future_shift"] == 1


def test_time_based_split_has_no_day_overlap():
    sensor_df, episodes = generate_sensor_timeseries()
    sensor_clean, _ = clean_sensor_data(sensor_df)
    production_df = generate_production_data(episodes)
    production_clean, _ = clean_production_data(production_df)
    from src.data.generate_synthetic import generate_maintenance_notes
    from src.preprocessing.clean_data import clean_maintenance_data
    maintenance_clean, _ = clean_maintenance_data(generate_maintenance_notes(episodes))

    integrated, _ = integrate_datasets(production_clean, sensor_clean, maintenance_clean)
    feature_df, _ = build_feature_set(integrated)
    train, val, test, summary = time_based_split(feature_df)

    assert summary["day_overlap_between_splits"] == []
    assert train["sim_day"].max() < val["sim_day"].min()
    assert val["sim_day"].max() < test["sim_day"].min()
    assert len(train) > 0 and len(val) > 0 and len(test) > 0


def test_split_preserves_chronological_order_per_split():
    sensor_df, episodes = generate_sensor_timeseries()
    sensor_clean, _ = clean_sensor_data(sensor_df)
    production_df = generate_production_data(episodes)
    production_clean, _ = clean_production_data(production_df)
    from src.data.generate_synthetic import generate_maintenance_notes
    from src.preprocessing.clean_data import clean_maintenance_data
    maintenance_clean, _ = clean_maintenance_data(generate_maintenance_notes(episodes))

    integrated, _ = integrate_datasets(production_clean, sensor_clean, maintenance_clean)
    feature_df, _ = build_feature_set(integrated)
    train, _, _, _ = time_based_split(feature_df)

    assert train["sim_day"].min() >= config.TRAIN_DAY_RANGE[0]
    assert train["sim_day"].max() <= config.TRAIN_DAY_RANGE[1]
