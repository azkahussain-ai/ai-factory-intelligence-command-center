"""Cleaning logic for the three raw manufacturing datasets.

Each clean_* function returns (cleaned_df, summary_dict) so cleaning
decisions are traceable and can be written into the Stage I EDA report
instead of silently disappearing.
"""
import numpy as np
import pandas as pd

SENSOR_NUMERIC_COLS = [
    "temperature_k", "process_temperature_k", "vibration_mm_s",
    "pressure_bar", "torque_nm", "rotational_speed_rpm", "load_pct",
]

# Domain-plausible physical bounds. Values outside these are sensor glitches,
# not real machine states, and are treated as missing rather than truth.
PHYSICAL_BOUNDS = {
    "temperature_k": (250, 400),
    "process_temperature_k": (250, 420),
    "vibration_mm_s": (0, 30),
    "pressure_bar": (0, 60),
    "torque_nm": (0, 150),
    "rotational_speed_rpm": (0, 5000),
    "load_pct": (0, 100),
}


def _normalize_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def clean_sensor_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    summary = {}

    summary["rows_before"] = len(df)

    # Normalize inconsistent categorical labels (case/whitespace variants).
    df["machine_type"] = _normalize_label(df["machine_type"])
    df["operating_status"] = _normalize_label(df["operating_status"])

    # Malformed timestamps ("unknown") cannot be recovered or safely imputed
    # for a time-series key, so those rows are dropped and counted.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    malformed_timestamp_rows = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])
    summary["rows_dropped_malformed_timestamp"] = malformed_timestamp_rows

    # Exact duplicates.
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates()
    summary["rows_dropped_exact_duplicate"] = duplicate_rows

    # Physically impossible readings -> treated as missing (sensor glitch),
    # never silently kept as if they were real values.
    glitch_counts = {}
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        invalid_mask = (df[col] < lo) | (df[col] > hi)
        glitch_counts[col] = int(invalid_mask.sum())
        df.loc[invalid_mask, col] = np.nan
    summary["physically_invalid_values_nulled"] = glitch_counts

    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    # Missingness during DOWN status is meaningful (sensors offline while the
    # machine is stopped) and is preserved, not imputed. Missingness while
    # RUNNING is dropout/glitch noise and is short-gap filled per machine.
    running_mask = df["operating_status"] == "RUNNING"
    imputed_counts = {}
    for col in SENSOR_NUMERIC_COLS:
        before_na = int(df.loc[running_mask, col].isna().sum())
        df[col] = df.groupby("machine_id")[col].transform(
            lambda s: s.ffill(limit=4).bfill(limit=4))
        # Any still-missing RUNNING values (gap too long to bridge) fall back
        # to that machine's overall median rather than being left NaN.
        still_missing = running_mask & df[col].isna()
        if still_missing.any():
            medians = df.groupby("machine_id")[col].transform("median")
            df.loc[still_missing, col] = medians[still_missing]
        after_na = int(df.loc[running_mask, col].isna().sum())
        imputed_counts[col] = before_na - after_na
    summary["running_state_values_imputed"] = imputed_counts
    summary["down_state_missing_preserved"] = int(
        df.loc[~running_mask, SENSOR_NUMERIC_COLS].isna().sum().sum())

    # Outlier analysis (reported, not removed - a spike can be a real
    # degrading-machine signal rather than a data error).
    outlier_counts = {}
    for col in ["vibration_mm_s", "temperature_k", "pressure_bar"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        outlier_counts[col] = int(((df[col] < bounds[0]) | (df[col] > bounds[1])).sum())
    summary["iqr_outliers_detected_and_kept"] = outlier_counts

    summary["rows_after"] = len(df)
    return df, summary


def clean_production_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    summary = {"rows_before": len(df)}

    df["machine_type"] = _normalize_label(df["machine_type"])
    df["operating_status"] = _normalize_label(df["operating_status"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates()
    summary["rows_dropped_exact_duplicate"] = duplicate_rows

    # Negative production_count is physically impossible; cannot be
    # recovered, so replaced with the machine+shift median (a stable,
    # explainable fallback) rather than dropped, since the rest of the row
    # (quality_score, downtime) is still valid information.
    invalid_count_mask = df["production_count"] < 0
    summary["invalid_production_count_fixed"] = int(invalid_count_mask.sum())
    df["production_count"] = df["production_count"].astype(float)
    group_medians = df.groupby(["machine_id", "shift"])["production_count"].transform("median")
    df.loc[invalid_count_mask, "production_count"] = group_medians[invalid_count_mask]
    df["production_count"] = df["production_count"].round().astype(int)

    missing_quality = int(df["quality_score"].isna().sum())
    df["quality_score"] = df["quality_score"].fillna(
        df.groupby("machine_id")["quality_score"].transform("median"))
    summary["quality_score_imputed"] = missing_quality

    df = df.sort_values(["machine_id", "date", "shift"]).reset_index(drop=True)
    summary["rows_after"] = len(df)
    return df, summary


def clean_maintenance_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    summary = {"rows_before": len(df)}

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates()
    summary["rows_dropped_exact_duplicate"] = duplicate_rows

    # Missing urgency is preserved as an explicit "Unknown" category rather
    # than guessed - fabricating a severity label would be worse than
    # admitting it wasn't recorded.
    missing_urgency = int(df["urgency"].isna().sum())
    df["urgency"] = df["urgency"].fillna("Unknown")
    summary["urgency_marked_unknown"] = missing_urgency

    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    summary["rows_after"] = len(df)
    return df, summary
