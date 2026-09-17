"""Reusable, portable data loading for the AI Factory pipeline.

Uses pathlib and project-relative paths only - no absolute local paths -
so the pipeline runs unmodified on Windows, GitHub Actions, and Hugging
Face Spaces.
"""
from pathlib import Path

import pandas as pd

from src.data.validate_schema import (
    MAINTENANCE_SCHEMA,
    PRODUCTION_SCHEMA,
    SENSOR_SCHEMA,
    validate_schema,
)


def load_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}. Run src/data/generate_synthetic.py first.")
    return pd.read_csv(path)


def load_raw_sensor_data(raw_dir: Path) -> pd.DataFrame:
    df = load_csv(Path(raw_dir) / "raw_sensor_data.csv")
    validate_schema(df, SENSOR_SCHEMA, "raw_sensor_data")
    return df


def load_raw_production_data(raw_dir: Path) -> pd.DataFrame:
    df = load_csv(Path(raw_dir) / "raw_production_data.csv")
    validate_schema(df, PRODUCTION_SCHEMA, "raw_production_data")
    return df


def load_raw_maintenance_data(raw_dir: Path) -> pd.DataFrame:
    df = load_csv(Path(raw_dir) / "raw_maintenance_data.csv")
    validate_schema(df, MAINTENANCE_SCHEMA, "raw_maintenance_data")
    return df
