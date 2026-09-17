"""Central configuration for the AI Factory 2.0 data pipeline.

All paths are relative to the project root so the pipeline runs the same
way on Windows, in CI, and on Hugging Face Spaces.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
SAMPLE_DIR = DATA_DIR / "sample"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42

# --- Synthetic fleet ---------------------------------------------------
# Machine types and their AI4I-style quality-variant mix (L=50%, M=30%, H=20%).
MACHINE_FLEET = [
    {"machine_id": "M-01", "machine_type": "CNC", "quality_variant": "L"},
    {"machine_id": "M-02", "machine_type": "CNC", "quality_variant": "M"},
    {"machine_id": "M-03", "machine_type": "Press", "quality_variant": "L"},
    {"machine_id": "M-04", "machine_type": "Press", "quality_variant": "H"},
    {"machine_id": "M-05", "machine_type": "Injection_Molder", "quality_variant": "M"},
    {"machine_id": "M-06", "machine_type": "Injection_Molder", "quality_variant": "L"},
]

# Baseline sensor ranges per machine type. Grounded loosely in the AI4I 2020
# dataset's documented generation process (air temperature random walk around
# 300K, torque ~N(40, 10) Nm, rotational speed derived from a ~2860W power
# draw) and extended with vibration/pressure baselines typical of the given
# machine class for a manufacturing digital twin.
MACHINE_TYPE_PROFILES = {
    "CNC": {
        "temperature_k_mean": 301.0, "temperature_k_std": 2.0,
        "vibration_mm_s_mean": 1.8, "vibration_mm_s_std": 0.4,
        "pressure_bar_mean": 5.5, "pressure_bar_std": 0.5,
        "torque_nm_mean": 40.0, "torque_nm_std": 10.0,
        "rated_power_w": 2860,
    },
    "Press": {
        "temperature_k_mean": 304.0, "temperature_k_std": 2.5,
        "vibration_mm_s_mean": 2.6, "vibration_mm_s_std": 0.6,
        "pressure_bar_mean": 12.0, "pressure_bar_std": 1.2,
        "torque_nm_mean": 55.0, "torque_nm_std": 12.0,
        "rated_power_w": 4200,
    },
    "Injection_Molder": {
        "temperature_k_mean": 308.0, "temperature_k_std": 3.0,
        "vibration_mm_s_mean": 1.4, "vibration_mm_s_std": 0.3,
        "pressure_bar_mean": 18.0, "pressure_bar_std": 1.8,
        "torque_nm_mean": 30.0, "torque_nm_std": 8.0,
        "rated_power_w": 3500,
    },
}

# --- Simulation window ---------------------------------------------------
SIM_START_DATE = "2026-05-01"
SIM_DAYS = 90
SENSOR_INTERVAL_MINUTES = 15
SHIFTS = [
    {"name": "Morning", "start_hour": 6, "end_hour": 14},
    {"name": "Afternoon", "start_hour": 14, "end_hour": 22},
    {"name": "Night", "start_hour": 22, "end_hour": 6},
]

# --- Time-based train/validation/test split (day-of-simulation boundaries) ---
TRAIN_DAY_RANGE = (1, 50)
VAL_DAY_RANGE = (51, 65)
TEST_DAY_RANGE = (66, 90)

# --- Feature engineering windows (in number of 15-minute sensor readings) ---
ROLLING_WINDOWS = [4, 16]  # 1 hour, 4 hours
LAG_STEPS = [1, 4]  # 15 min, 1 hour

# Prediction horizon for the shift-level failure target: does the machine
# fail during the *next* shift given everything known through the end of
# the current shift.
PREDICTION_HORIZON_SHIFTS = 1

# --- Stage VII: Digital Twin / What-If Simulation -------------------------
# Downtime durations below are EMPIRICAL, computed from the real generated
# fleet data (data/processed/cleaned_production.csv), not invented:
#   - shifts with operating_status == DOWN (unplanned failure, fills the
#     whole shift) average 480 minutes of downtime = 8.0 hours
#   - shifts with operating_status == MAINTENANCE (a planned/partial stop)
#     average 270 minutes = 4.5 hours
# Financial constants (unit margin, hourly maintenance cost, failure repair
# cost) have no data source anywhere in this project (no pricing/financial
# data was ever generated), so they are explicit, clearly-labelled
# assumptions - not measured, and not randomly generated either.
DIGITAL_TWIN_HORIZON_SHIFTS = 6           # ~2 days ahead
SHIFT_DURATION_HOURS = 8
UNPLANNED_FAILURE_DOWNTIME_HOURS = 8.0    # empirical: mean DOWN-status downtime
PREVENTIVE_MAINTENANCE_DOWNTIME_HOURS = 4.5  # empirical: mean MAINTENANCE-status downtime
LOAD_REDUCTION_FRACTION = 0.20            # documented assumption: 20% load/rate cut

UNIT_MARGIN_USD = 12.0                    # documented assumption: profit per unit produced
MAINTENANCE_COST_PER_HOUR_USD = 150.0     # documented assumption: labor/opportunity cost per downtime hour
FAILURE_REPAIR_COST_USD = 2000.0          # documented assumption: extra parts/labor cost of an unplanned failure
                                           # beyond the downtime hours themselves
