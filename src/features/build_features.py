"""Leakage-safe feature engineering on the shift-level integrated dataset.

Every predictor feature uses only information available at or before the
current shift (lag features are shifted by 1 before any rolling window is
applied, so a rolling feature at shift t never includes shift t's own
reading). The prediction target intentionally looks one shift into the
future - that is what a supervised label is - and is kept separate from the
predictor columns for that reason.
"""
import numpy as np
import pandas as pd

SHIFT_ORDER = {"Morning": 0, "Afternoon": 1, "Night": 2}
ROLLING_WINDOWS_SHIFTS = [3, 9]  # ~1 day, ~3 days of shifts

SENSOR_MEAN_COLS = [
    "temperature_k_mean", "process_temperature_k_mean", "vibration_mm_s_mean",
    "pressure_bar_mean", "torque_nm_mean", "rotational_speed_rpm_mean", "load_pct_mean",
]
LAG_SOURCE_COLS = ["quality_score", "production_rate", "downtime_minutes"] + SENSOR_MEAN_COLS


def _shift_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["shift_order"] = df["shift"].map(SHIFT_ORDER)
    df = df.sort_values(["machine_id", "date", "shift_order"]).reset_index(drop=True)
    return df


def add_production_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    shift_hours = 8
    df["downtime_ratio"] = (df["downtime_minutes"] / (shift_hours * 60)).clip(0, 1)
    df["production_efficiency"] = np.where(
        df["downtime_ratio"] < 1,
        df["production_count"] / ((shift_hours - df["downtime_minutes"] / 60).clip(lower=0.1)),
        0.0,
    )
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Per-machine lag (t-1) and past-only rolling features.

    Rolling stats are computed on the already-lagged series
    (`.shift(1).rolling(window)`), so the feature at row t is built strictly
    from shifts before t, never including t itself.
    """
    df = _shift_sort_key(df)
    new_cols = []

    for col in LAG_SOURCE_COLS:
        if col not in df.columns:
            continue
        lag_col = f"{col}_lag1"
        df[lag_col] = df.groupby("machine_id")[col].shift(1)
        new_cols.append(lag_col)

        for window in ROLLING_WINDOWS_SHIFTS:
            roll_col = f"{col}_rollmean_{window}"
            df[roll_col] = (
                df.groupby("machine_id")[col]
                .transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
            )
            new_cols.append(roll_col)

    df = df.drop(columns=["shift_order"])
    return df, new_cols


def add_maintenance_frequency_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Recent maintenance count is already computed past-only in
    integrate_data.py; here we just derive a stable "has_recent_maintenance"
    flag used as a simple categorical feature.
    """
    df = df.copy()
    df["has_recent_maintenance"] = (df["maintenance_count_past_7d"] > 0).astype(int)
    return df


def add_prediction_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target: does the machine fail during the NEXT shift.

    This column looks forward by design (it is the label), and must never
    be used as a predictor feature.
    """
    df = _shift_sort_key(df)
    df["failure_next_shift"] = df.groupby("machine_id")["failure_occurred_this_shift"].shift(-1)
    df = df.drop(columns=["shift_order"])
    return df


def build_feature_set(integrated_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    summary = {"rows_before": len(integrated_df)}

    df = add_production_features(integrated_df)
    df, lag_roll_cols = add_lag_and_rolling_features(df)
    df = add_maintenance_frequency_feature(df)
    df = add_prediction_target(df)

    # Rows with no next shift (the last shift recorded per machine) have no
    # valid target and cannot be used for supervised learning.
    rows_no_target = int(df["failure_next_shift"].isna().sum())
    df = df.dropna(subset=["failure_next_shift"])
    df["failure_next_shift"] = df["failure_next_shift"].astype(int)

    summary["engineered_feature_columns"] = lag_roll_cols
    summary["rows_dropped_no_future_shift"] = rows_no_target
    summary["rows_after"] = len(df)
    summary["failure_next_shift_positive_rate"] = float(df["failure_next_shift"].mean())
    return df, summary
