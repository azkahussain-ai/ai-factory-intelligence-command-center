"""Generate synthetic manufacturing data for the AI Factory 2.0 fleet.

Sensor generation follows the documented AI4I 2020 Predictive Maintenance
Dataset methodology (Matzka, 2020): temperature as a normalized random walk,
torque as a bounded normal distribution, and rotational speed derived from a
fixed power draw with noise. That dataset is itself synthetic ("a synthetic
dataset that reflects real predictive maintenance data encountered in
industry" - UCI repository page), so we replicate its generation logic
locally rather than claim collected sensor data. Vibration and pressure
channels are not part of AI4I and are added with documented, plausible
baselines per machine type since the hackathon spec requires them.

Realistic data-quality problems (missing values, duplicates, invalid
readings, inconsistent labels, malformed timestamps) are deliberately
injected into the *raw* outputs so Stage I's cleaning code has real work to
do. This is documented in data/synthetic/SOURCE.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import config


def _rng():
    return np.random.RandomState(config.RANDOM_STATE)


def _timestamp_index(start_date: str, days: int, interval_minutes: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date)
    periods = int(days * 24 * 60 / interval_minutes)
    return pd.date_range(start=start, periods=periods, freq=f"{interval_minutes}min")


def _plan_failure_episodes(machine_ids, days, rng):
    """Randomly assign 0-3 degrade-to-failure episodes per machine.

    Each episode has a ramp (3-7 days of worsening sensor trend) ending in a
    failure event, followed by downtime (2-24 hours) before the machine
    returns to a healthy baseline.
    """
    episodes = {m: [] for m in machine_ids}
    for m in machine_ids:
        n_episodes = rng.choice([1, 2, 2, 3, 3, 4], p=[0.1, 0.25, 0.25, 0.2, 0.15, 0.05])
        used_days = []
        for _ in range(n_episodes):
            ramp_days = rng.randint(3, 8)
            # keep episodes spaced apart and inside the simulation window
            attempt = 0
            while attempt < 20:
                failure_day = rng.randint(ramp_days + 2, days - 2)
                if all(abs(failure_day - d) > ramp_days + 3 for d in used_days):
                    used_days.append(failure_day)
                    downtime_hours = rng.randint(2, 25)
                    episodes[m].append({
                        "failure_day": failure_day,
                        "ramp_days": ramp_days,
                        "downtime_hours": downtime_hours,
                    })
                    break
                attempt += 1
    return episodes


def generate_sensor_timeseries() -> pd.DataFrame:
    """Simulate 15-minute sensor readings for the whole fleet, with injected
    degradation-to-failure episodes and realistic data-quality issues.
    """
    rng = _rng()
    machine_ids = [m["machine_id"] for m in config.MACHINE_FLEET]
    timestamps = _timestamp_index(config.SIM_START_DATE, config.SIM_DAYS, config.SENSOR_INTERVAL_MINUTES)
    episodes = _plan_failure_episodes(machine_ids, config.SIM_DAYS, rng)

    rows = []
    for machine in config.MACHINE_FLEET:
        mid = machine["machine_id"]
        profile = config.MACHINE_TYPE_PROFILES[machine["machine_type"]]
        machine_episodes = episodes[mid]

        # AI4I-style random walk for temperature, normalized around the profile mean.
        temp_walk = np.cumsum(rng.normal(0, 0.05, size=len(timestamps)))
        temp_walk = (temp_walk - temp_walk.mean())
        temp_walk = temp_walk / (temp_walk.std() + 1e-9) * profile["temperature_k_std"]

        wear_minutes = 0.0
        for i, ts in enumerate(timestamps):
            day = i * config.SENSOR_INTERVAL_MINUTES / (24 * 60)

            # default: healthy baseline
            degrade_factor = 0.0
            status = "RUNNING"
            in_downtime = False

            for ep in machine_episodes:
                ramp_start = ep["failure_day"] - ep["ramp_days"]
                failure_day = ep["failure_day"]
                downtime_end_day = failure_day + ep["downtime_hours"] / 24
                if ramp_start <= day < failure_day:
                    degrade_factor = max(degrade_factor, (day - ramp_start) / ep["ramp_days"])
                elif failure_day <= day < downtime_end_day:
                    in_downtime = True
                    status = "DOWN"

            air_temp = profile["temperature_k_mean"] + temp_walk[i] + degrade_factor * 4.0
            process_temp = air_temp + rng.normal(10, 1)
            vibration = max(0.0, rng.normal(
                profile["vibration_mm_s_mean"] + degrade_factor * 3.5,
                profile["vibration_mm_s_std"]))
            pressure = max(0.0, rng.normal(
                profile["pressure_bar_mean"] + degrade_factor * 1.5,
                profile["pressure_bar_std"]))
            torque = max(0.5, rng.normal(profile["torque_nm_mean"], profile["torque_nm_std"]))
            rpm = (profile["rated_power_w"] / torque) * 9.5493 + rng.normal(0, 15)
            load_pct = float(np.clip((torque / profile["torque_nm_mean"]) * 70 + rng.normal(0, 5), 0, 100))

            if in_downtime:
                air_temp = process_temp = vibration = pressure = torque = rpm = load_pct = np.nan
            else:
                wear_minutes += config.SENSOR_INTERVAL_MINUTES / 60

            failure_flag = 1 if any(
                abs(day - ep["failure_day"]) < (config.SENSOR_INTERVAL_MINUTES / (24 * 60))
                for ep in machine_episodes
            ) else 0

            rows.append({
                "machine_id": mid,
                "machine_type": machine["machine_type"],
                "timestamp": ts,
                "temperature_k": air_temp,
                "process_temperature_k": process_temp,
                "vibration_mm_s": vibration,
                "pressure_bar": pressure,
                "torque_nm": torque,
                "rotational_speed_rpm": rpm,
                "load_pct": load_pct,
                "tool_wear_min": round(wear_minutes, 2),
                "operating_status": status,
                "failure_flag": failure_flag,
            })

    df = pd.DataFrame(rows)
    df = _inject_sensor_quality_issues(df, rng)
    return df, episodes


def _inject_sensor_quality_issues(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Introduce realistic messiness: sensor dropout, invalid readings,
    inconsistent labels, and a handful of exact-duplicate rows.
    """
    df = df.copy()
    n = len(df)

    # Sensor dropout: ~0.4% of numeric readings go missing at random.
    numeric_cols = ["temperature_k", "process_temperature_k", "vibration_mm_s",
                     "pressure_bar", "torque_nm", "rotational_speed_rpm", "load_pct"]
    for col in numeric_cols:
        dropout_idx = rng.choice(n, size=int(n * 0.004), replace=False)
        df.loc[dropout_idx, col] = np.nan

    # Physically impossible sensor values (sensor glitches).
    glitch_idx = rng.choice(n, size=20, replace=False)
    for i, idx in enumerate(glitch_idx):
        if i % 4 == 0:
            df.loc[idx, "vibration_mm_s"] = -abs(rng.normal(3, 1))  # negative vibration
        elif i % 4 == 1:
            df.loc[idx, "rotational_speed_rpm"] = 999999  # impossible rpm
        elif i % 4 == 2:
            df.loc[idx, "temperature_k"] = -50  # impossible negative Kelvin
        else:
            df.loc[idx, "pressure_bar"] = -abs(rng.normal(2, 1))  # negative pressure

    # Inconsistent categorical labels for machine_type and operating_status.
    label_idx = rng.choice(n, size=15, replace=False)
    messy_type_map = {"CNC": "cnc", "Press": "PRESS ", "Injection_Molder": "injection_molder"}
    for idx in label_idx[:8]:
        original = df.loc[idx, "machine_type"]
        df.loc[idx, "machine_type"] = messy_type_map.get(original, original)
    for idx in label_idx[8:]:
        df.loc[idx, "operating_status"] = df.loc[idx, "operating_status"].lower()

    # Exact duplicate rows.
    dup_rows = df.sample(n=10, random_state=config.RANDOM_STATE)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Malformed timestamps (stored as text so this must survive to the raw CSV).
    df["timestamp"] = df["timestamp"].astype(str)
    malformed_idx = rng.choice(len(df), size=6, replace=False)
    for idx in malformed_idx:
        df.loc[idx, "timestamp"] = "unknown"

    return df


def generate_production_data(episodes) -> pd.DataFrame:
    """Aggregate shift-level production records, reduced by any overlapping
    downtime and degraded quality during ramp-to-failure periods.
    """
    rng = _rng()
    rows = []
    n_days = config.SIM_DAYS
    start = pd.Timestamp(config.SIM_START_DATE)

    for machine in config.MACHINE_FLEET:
        mid = machine["machine_id"]
        mtype = machine["machine_type"]
        profile = config.MACHINE_TYPE_PROFILES[mtype]
        base_rate = {"CNC": 42, "Press": 65, "Injection_Molder": 120}[mtype]
        machine_episodes = episodes[mid]

        for day in range(1, n_days + 1):
            date = start + pd.Timedelta(days=day - 1)
            for shift in config.SHIFTS:
                shift_hours = 8
                degrade_factor = 0.0
                downtime_minutes = 0
                status = "RUNNING"

                for ep in machine_episodes:
                    ramp_start = ep["failure_day"] - ep["ramp_days"]
                    failure_day = ep["failure_day"]
                    downtime_end_day = failure_day + ep["downtime_hours"] / 24
                    if ramp_start <= day < failure_day:
                        degrade_factor = max(degrade_factor, (day - ramp_start) / ep["ramp_days"])
                    if failure_day <= day < downtime_end_day:
                        overlap_hours = min(shift_hours, ep["downtime_hours"])
                        downtime_minutes += overlap_hours * 60
                        status = "DOWN" if overlap_hours >= shift_hours else "MAINTENANCE"

                effective_hours = max(0.0, shift_hours - downtime_minutes / 60)
                production_rate = base_rate * (1 - 0.25 * degrade_factor) * rng.normal(1.0, 0.03)
                production_count = max(0, round(production_rate * effective_hours))
                quality_score = float(np.clip(rng.normal(96 - 15 * degrade_factor, 2), 0, 100))
                shift_downtime = round(downtime_minutes, 1)

                rows.append({
                    "machine_id": mid,
                    "machine_type": mtype,
                    "date": date.date().isoformat(),
                    "shift": shift["name"],
                    "production_rate": round(production_rate, 2),
                    "production_count": production_count,
                    "quality_score": round(quality_score, 2),
                    "downtime_minutes": shift_downtime,
                    "operating_status": status,
                })

    df = pd.DataFrame(rows)
    df = _inject_production_quality_issues(df, rng)
    return df


def _inject_production_quality_issues(df: pd.DataFrame, rng) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    missing_idx = rng.choice(n, size=int(n * 0.01), replace=False)
    df.loc[missing_idx, "quality_score"] = np.nan

    invalid_idx = rng.choice(n, size=6, replace=False)
    df.loc[invalid_idx, "production_count"] = -1

    status_idx = rng.choice(n, size=10, replace=False)
    for idx in status_idx:
        df.loc[idx, "operating_status"] = df.loc[idx, "operating_status"].capitalize()

    dup_rows = df.sample(n=5, random_state=config.RANDOM_STATE)
    df = pd.concat([df, dup_rows], ignore_index=True)
    return df


_INCIDENT_TEMPLATES = {
    "Mechanical": "Operator reported {noise} on {machine}. Vibration levels appeared elevated during inspection.",
    "Electrical": "Intermittent power fluctuation observed on {machine}. Panel indicator flickered during operation.",
    "Quality": "Output from {machine} showed increased defect rate this shift. Quality inspector flagged the batch.",
    "Safety": "Guard interlock on {machine} triggered unexpectedly. Machine stopped as a safety precaution.",
}
_NOISE_PHRASES = ["unusual grinding noise", "a knocking sound", "high-pitched vibration noise", "abnormal rattling"]


def generate_maintenance_notes(episodes) -> pd.DataFrame:
    """Generate corrective notes tied to failure episodes plus routine
    preventive maintenance entries, with realistic missing/urgency gaps.
    """
    rng = _rng()
    rows = []
    start = pd.Timestamp(config.SIM_START_DATE)

    for machine in config.MACHINE_FLEET:
        mid = machine["machine_id"]
        for ep in episodes[mid]:
            ts = start + pd.Timedelta(days=ep["failure_day"] - 1, hours=int(rng.randint(6, 20)))
            category = rng.choice(["Mechanical", "Electrical", "Quality"], p=[0.6, 0.2, 0.2])
            note = _INCIDENT_TEMPLATES[category].format(
                noise=rng.choice(_NOISE_PHRASES), machine=mid)
            rows.append({
                "machine_id": mid,
                "timestamp": ts,
                "maintenance_type": "Corrective",
                "incident_type": category,
                "maintenance_note": note,
                "urgency": rng.choice(["Medium", "High"], p=[0.3, 0.7]),
            })

        n_routine = rng.randint(3, 8)
        for _ in range(n_routine):
            day_offset = rng.randint(0, config.SIM_DAYS)
            ts = start + pd.Timedelta(days=int(day_offset), hours=int(rng.randint(6, 20)))
            rows.append({
                "machine_id": mid,
                "timestamp": ts,
                "maintenance_type": "Preventive",
                "incident_type": "Scheduled",
                "maintenance_note": f"Routine scheduled inspection and lubrication performed on {mid}.",
                "urgency": "Low",
            })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df = _inject_maintenance_quality_issues(df, rng)
    return df


def _inject_maintenance_quality_issues(df: pd.DataFrame, rng) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    if n == 0:
        return df
    missing_idx = rng.choice(n, size=max(1, int(n * 0.08)), replace=False)
    df.loc[missing_idx, "urgency"] = np.nan
    return df


def save_raw_outputs():
    """Run the full generator and persist raw (uncleaned) CSVs."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    sensor_df, episodes = generate_sensor_timeseries()
    production_df = generate_production_data(episodes)
    maintenance_df = generate_maintenance_notes(episodes)

    sensor_df.to_csv(config.RAW_DIR / "raw_sensor_data.csv", index=False)
    production_df.to_csv(config.RAW_DIR / "raw_production_data.csv", index=False)
    maintenance_df.to_csv(config.RAW_DIR / "raw_maintenance_data.csv", index=False)

    return sensor_df, production_df, maintenance_df, episodes


if __name__ == "__main__":
    s, p, m, ep = save_raw_outputs()
    print(f"sensor rows: {len(s)}, production rows: {len(p)}, maintenance rows: {len(m)}")
    total_episodes = sum(len(v) for v in ep.values())
    print(f"failure episodes generated: {total_episodes}")
