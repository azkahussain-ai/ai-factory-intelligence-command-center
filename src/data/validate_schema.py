"""Schema validation for raw manufacturing datasets.

Fails loudly with a specific message rather than letting bad data flow
silently into cleaning/feature code.
"""
import pandas as pd

SENSOR_SCHEMA = {
    "machine_id": "object",
    "machine_type": "object",
    "timestamp": "object",
    "temperature_k": "float",
    "process_temperature_k": "float",
    "vibration_mm_s": "float",
    "pressure_bar": "float",
    "torque_nm": "float",
    "rotational_speed_rpm": "float",
    "load_pct": "float",
    "tool_wear_min": "float",
    "operating_status": "object",
    "failure_flag": "int",
}

PRODUCTION_SCHEMA = {
    "machine_id": "object",
    "machine_type": "object",
    "date": "object",
    "shift": "object",
    "production_rate": "float",
    "production_count": "int",
    "quality_score": "float",
    "downtime_minutes": "float",
    "operating_status": "object",
}

MAINTENANCE_SCHEMA = {
    "machine_id": "object",
    "timestamp": "object",
    "maintenance_type": "object",
    "incident_type": "object",
    "maintenance_note": "object",
    "urgency": "object",
}


class SchemaValidationError(Exception):
    pass


def validate_schema(df: pd.DataFrame, schema: dict, name: str) -> None:
    """Check required columns are present. Raise with a clear message if not.

    Datatype is checked loosely (numeric vs object) since raw CSVs may still
    contain string-coded invalid values (e.g. "unknown" timestamps) that
    cleaning is responsible for resolving.
    """
    missing = [col for col in schema if col not in df.columns]
    if missing:
        raise SchemaValidationError(f"[{name}] missing required columns: {missing}")

    if df.empty:
        raise SchemaValidationError(f"[{name}] dataset is empty")

    for col, expected_kind in schema.items():
        if expected_kind == "float" or expected_kind == "int":
            non_null = pd.to_numeric(df[col], errors="coerce")
            unparseable = non_null.isna().sum() - df[col].isna().sum()
            if unparseable > len(df) * 0.5:
                raise SchemaValidationError(
                    f"[{name}] column '{col}' expected numeric but is mostly unparseable")

    required_ids = set(df["machine_id"].dropna().unique()) if "machine_id" in df.columns else set()
    if not required_ids:
        raise SchemaValidationError(f"[{name}] no valid machine_id values found")
