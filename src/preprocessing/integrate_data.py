"""Integrate sensor, production, and maintenance data into one dataset.

Sensor readings (15-minute) are aggregated up to the shift level
(machine_id, date, shift) so they can be joined with the shift-level
production records on a clean key. Maintenance history is joined as
past-only rolling counts, never future events, to keep the integration
leakage-safe.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import config

SENSOR_AGG_COLS = [
    "temperature_k", "process_temperature_k", "vibration_mm_s",
    "pressure_bar", "torque_nm", "rotational_speed_rpm", "load_pct",
]


def _assign_shift(hour: int) -> str:
    for shift in config.SHIFTS:
        start, end = shift["start_hour"], shift["end_hour"]
        if start < end:
            if start <= hour < end:
                return shift["name"]
        else:  # overnight shift, e.g. 22-6
            if hour >= start or hour < end:
                return shift["name"]
    return "Unknown"


def aggregate_sensors_to_shift(sensor_df: pd.DataFrame) -> pd.DataFrame:
    df = sensor_df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["shift"] = df["hour"].apply(_assign_shift)
    # Night shift starts before midnight; attribute the post-midnight hours
    # to the shift's starting calendar date so it matches the production
    # table's one-row-per-shift convention.
    df["shift_date"] = df["timestamp"].dt.date
    night_after_midnight = (df["shift"] == "Night") & (df["hour"] < 6)
    df.loc[night_after_midnight, "shift_date"] = (
        df.loc[night_after_midnight, "timestamp"] - pd.Timedelta(days=1)).dt.date

    agg_dict = {col: ["mean", "std", "max"] for col in SENSOR_AGG_COLS}
    agg_dict["failure_flag"] = "max"
    grouped = df.groupby(["machine_id", "shift_date", "shift"]).agg(agg_dict)
    grouped.columns = ["_".join(c) for c in grouped.columns]
    grouped = grouped.rename(columns={"failure_flag_max": "failure_occurred_this_shift"})
    grouped = grouped.reset_index().rename(columns={"shift_date": "date"})
    grouped["date"] = pd.to_datetime(grouped["date"])
    return grouped


def _maintenance_features(maintenance_df: pd.DataFrame, machine_id: str, as_of: pd.Timestamp) -> dict:
    """Past-only maintenance history for one machine as of a shift start."""
    history = maintenance_df[
        (maintenance_df["machine_id"] == machine_id) & (maintenance_df["timestamp"] < as_of)
    ]
    if history.empty:
        return {"days_since_last_maintenance": np.nan, "maintenance_count_past_7d": 0}
    days_since = (as_of - history["timestamp"].max()).total_seconds() / 86400
    recent = history[history["timestamp"] >= as_of - pd.Timedelta(days=7)]
    return {
        "days_since_last_maintenance": round(days_since, 2),
        "maintenance_count_past_7d": len(recent),
    }


def integrate_datasets(
    production_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    maintenance_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    summary = {}
    sensor_shift_df = aggregate_sensors_to_shift(sensor_df)

    production_df = production_df.copy()
    production_df["date"] = pd.to_datetime(production_df["date"])

    rows_before = len(production_df)
    merged = production_df.merge(
        sensor_shift_df, on=["machine_id", "date", "shift"], how="left", suffixes=("", "_sensor"))
    summary["rows_before_merge"] = rows_before
    summary["rows_after_merge"] = len(merged)
    summary["unmatched_sensor_rows"] = int(merged[SENSOR_AGG_COLS[0] + "_mean"].isna().sum())

    # Shift start timestamp, used only to compute past-only maintenance features.
    shift_start_hour = {s["name"]: s["start_hour"] for s in config.SHIFTS}
    merged["shift_start_ts"] = merged.apply(
        lambda r: r["date"] + pd.Timedelta(hours=shift_start_hour[r["shift"]]), axis=1)

    maint_features = merged.apply(
        lambda r: _maintenance_features(maintenance_df, r["machine_id"], r["shift_start_ts"]),
        axis=1, result_type="expand")
    merged = pd.concat([merged, maint_features], axis=1)
    merged = merged.drop(columns=["shift_start_ts"])

    merged = merged.sort_values(["machine_id", "date", "shift"]).reset_index(drop=True)
    summary["final_columns"] = list(merged.columns)
    summary["final_rows"] = len(merged)
    return merged, summary
