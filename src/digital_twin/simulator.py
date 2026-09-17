"""Stage VII — Digital Twin / What-If Simulation.

Every scenario's failure-risk number comes from a real forward pass through
Stage II's trained GRU (`models/gru_model.npz`) - never a random or
hand-picked probability. "What-if" scenarios that change the machine's
state (preventive maintenance completed, load reduced) are built by
perturbing the GRU's real input window with either the machine's own real
current readings (load-reduction scenario) or fleet-wide empirical
healthy-state averages computed from the real training data (post-
maintenance scenario) - documented counterfactuals, not invented values.

Downtime-duration constants are empirical (computed from
`data/processed/cleaned_production.csv`); financial constants (unit
margin, hourly cost, repair cost) are explicit documented assumptions, since
no pricing data exists anywhere in this project - see `config/config.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.agents.predictive_maintenance_agent import (
    _load_gru_artifacts,
    build_gru_input,
)
from src.agents.schemas import unavailable_result, error_result

_HEALTHY_BASELINE_CACHE: dict | None = None


def _fleet_healthy_baseline(contract: dict) -> dict:
    """Mean value of each GRU feature, computed once from TRAINING rows
    where operating_status == RUNNING - a real, data-derived "typical
    healthy machine" reference point, used only to construct the
    post-maintenance what-if scenario below (never used for training or
    for any other stage's predictions).
    """
    global _HEALTHY_BASELINE_CACHE
    if _HEALTHY_BASELINE_CACHE is not None:
        return _HEALTHY_BASELINE_CACHE

    train_path = config.PROCESSED_DIR / "train.csv"
    train = pd.read_csv(train_path)
    healthy = train[train["operating_status"] == "RUNNING"]

    feature_cols = contract["feature_cols"]
    baseline = {}
    for col in feature_cols:
        if col in ("operating_status_DOWN", "operating_status_MAINTENANCE", "operating_status_RUNNING"):
            continue
        baseline[col] = float(healthy[col].mean()) if col in healthy.columns else 0.0
    _HEALTHY_BASELINE_CACHE = baseline
    return baseline


def _predict_from_raw_row(raw_values: dict, contract: dict, model, window: int) -> float:
    """Standardize a single raw feature-value dict using the GRU's train-only
    means/stds, repeat it across the window (representing a stable/just-
    reset state with no historical trend), and get the model's real
    predicted probability.
    """
    feature_cols = contract["feature_cols"]
    means = contract["means"]
    stds = contract["stds"]
    row = np.array([(raw_values[c] - means[c]) / (stds[c] or 1.0) for c in feature_cols])
    X_seq = np.tile(row, (window, 1)).reshape(1, window, len(feature_cols))
    return float(model.predict_proba(X_seq)[0])


def _current_state(machine_id: str, model, contract: dict, sim_day: int | None = None, shift: str | None = None):
    """Real production rate + real GRU failure probability for a machine.

    By default uses the machine's actual most-recent window (live state).
    Pass `sim_day`/`shift` to instead build the state from a specific
    historical window (e.g. a known past failure) - reuses Stage VI's
    `_build_window_as_of` rather than duplicating window-construction logic.
    """
    if sim_day is not None and shift is not None:
        from src.xai.gru_explainer import _build_window_as_of
        X_seq, recent = _build_window_as_of(machine_id, contract, sim_day, shift)
    else:
        X_seq, recent = build_gru_input(machine_id, contract)
    if X_seq is None:
        return None
    current_prob = float(model.predict_proba(X_seq)[0])
    production_rate = float(recent["production_rate"].iloc[-1])
    machine_type = recent["machine_type"].iloc[-1] if "machine_type" in recent.columns else None
    last_raw_row = {c: float(recent[c].iloc[-1]) for c in contract["feature_cols"]
                     if c in recent.columns}
    return {
        "production_rate": production_rate,
        "machine_type": machine_type,
        "current_failure_probability": current_prob,
        "last_raw_row": last_raw_row,
    }


def _compounded_risk(per_shift_prob: float, n_shifts: float) -> float:
    """Cumulative probability of at least one failure over n_shifts,
    assuming the given per-shift probability applies independently each
    shift - a standard, documented statistical compounding assumption
    (1 - (1-p)^n), not an arbitrary number.
    """
    n_shifts = max(0.0, n_shifts)
    return 1.0 - (1.0 - per_shift_prob) ** n_shifts


def _financials(expected_production: float, downtime_hours: float, cumulative_risk: float) -> dict:
    revenue = expected_production * config.UNIT_MARGIN_USD
    downtime_cost = downtime_hours * config.MAINTENANCE_COST_PER_HOUR_USD
    expected_failure_cost = cumulative_risk * config.FAILURE_REPAIR_COST_USD
    net_value = revenue - downtime_cost - expected_failure_cost
    return {
        "expected_revenue_usd": round(revenue, 2),
        "downtime_cost_usd": round(downtime_cost, 2),
        "expected_failure_repair_cost_usd": round(expected_failure_cost, 2),
        "estimated_net_value_usd": round(net_value, 2),
    }


def simulate_continue_operating(machine_id: str, model, contract: dict, state: dict, horizon_shifts: int) -> dict:
    horizon_hours = horizon_shifts * config.SHIFT_DURATION_HOURS
    p = state["current_failure_probability"]
    cumulative_risk = _compounded_risk(p, horizon_shifts)
    expected_lost_hours = cumulative_risk * config.UNPLANNED_FAILURE_DOWNTIME_HOURS
    expected_production = state["production_rate"] * max(0.0, horizon_hours - expected_lost_hours)

    result = {
        "scenario": "continue_operating",
        "description": "Continue normal operation at the current production rate for the full horizon.",
        "failure_probability_source": "GRU model, current real input window (unmodified)",
        "per_shift_failure_probability": round(p, 4),
        "cumulative_failure_risk": round(cumulative_risk, 4),
        "expected_downtime_hours": round(expected_lost_hours, 2),
        "expected_production_units": round(expected_production, 1),
        "production_loss_units": round(state["production_rate"] * horizon_hours - expected_production, 1),
    }
    result.update(_financials(expected_production, expected_lost_hours, cumulative_risk))
    return result


def simulate_stop_for_maintenance(machine_id: str, model, contract: dict, state: dict, horizon_shifts: int) -> dict:
    horizon_hours = horizon_shifts * config.SHIFT_DURATION_HOURS
    downtime_hours = min(config.PREVENTIVE_MAINTENANCE_DOWNTIME_HOURS, horizon_hours)
    remaining_hours = horizon_hours - downtime_hours
    remaining_shifts = remaining_hours / config.SHIFT_DURATION_HOURS

    baseline = _fleet_healthy_baseline(contract)
    post_maintenance_row = dict(baseline)
    post_maintenance_row.update({
        "downtime_minutes": 0.0,
        "downtime_ratio": 0.0,
        "has_recent_maintenance": 1.0,
        "days_since_last_maintenance": 0.0,
        "failure_occurred_this_shift": 0.0,
        "operating_status_DOWN": 0.0,
        "operating_status_MAINTENANCE": 0.0,
        "operating_status_RUNNING": 1.0,
    })
    if "maintenance_count_past_7d" in state["last_raw_row"]:
        post_maintenance_row["maintenance_count_past_7d"] = state["last_raw_row"]["maintenance_count_past_7d"] + 1

    p_post = _predict_from_raw_row(post_maintenance_row, contract, model, contract["window"])
    cumulative_risk = _compounded_risk(p_post, remaining_shifts)
    expected_production = state["production_rate"] * remaining_hours

    result = {
        "scenario": "stop_for_maintenance",
        "description": f"Stop now for preventive maintenance ({downtime_hours:.1f}h, the fleet's empirical "
                        "average planned-maintenance duration), then resume normal operation.",
        "failure_probability_source": "GRU model, counterfactual post-maintenance input "
                                       "(fleet-wide healthy-state training averages + reset maintenance fields)",
        "per_shift_failure_probability": round(p_post, 4),
        "cumulative_failure_risk": round(cumulative_risk, 4),
        "expected_downtime_hours": round(downtime_hours, 2),
        "expected_production_units": round(expected_production, 1),
        "production_loss_units": round(state["production_rate"] * downtime_hours, 1),
    }
    result.update(_financials(expected_production, downtime_hours, cumulative_risk))
    return result


def simulate_reduce_load(machine_id: str, model, contract: dict, state: dict, horizon_shifts: int) -> dict:
    horizon_hours = horizon_shifts * config.SHIFT_DURATION_HOURS
    reduction = config.LOAD_REDUCTION_FRACTION

    reduced_row = dict(state["last_raw_row"])
    for col in ("load_pct_mean", "torque_nm_mean", "rotational_speed_rpm_mean", "production_rate"):
        if col in reduced_row:
            reduced_row[col] = reduced_row[col] * (1 - reduction)

    p_reduced = _predict_from_raw_row(reduced_row, contract, model, contract["window"])
    cumulative_risk = _compounded_risk(p_reduced, horizon_shifts)
    reduced_rate = state["production_rate"] * (1 - reduction)
    expected_lost_hours = cumulative_risk * config.UNPLANNED_FAILURE_DOWNTIME_HOURS
    expected_production = reduced_rate * max(0.0, horizon_hours - expected_lost_hours)

    result = {
        "scenario": "reduce_load",
        "description": f"Reduce production rate/load by {reduction:.0%} for the full horizon.",
        "failure_probability_source": "GRU model, counterfactual input with load/torque/rpm/production_rate "
                                       f"scaled down {reduction:.0%} from the machine's real current readings",
        "per_shift_failure_probability": round(p_reduced, 4),
        "cumulative_failure_risk": round(cumulative_risk, 4),
        "expected_downtime_hours": round(expected_lost_hours, 2),
        "expected_production_units": round(expected_production, 1),
        "production_loss_units": round(state["production_rate"] * horizon_hours - expected_production, 1),
    }
    result.update(_financials(expected_production, expected_lost_hours, cumulative_risk))
    return result


def run_scenarios(machine_id: str, horizon_shifts: int | None = None,
                   sim_day: int | None = None, shift: str | None = None) -> dict:
    """Run all 3 required what-if scenarios for a machine and return a
    structured comparison. Every number traces back to either a real GRU
    forward pass, real historical fleet data, or an explicitly documented
    assumption (never a random value).

    By default simulates from the machine's current (live) state. Pass
    `sim_day`/`shift` to instead simulate starting from a specific
    historical window (e.g. a known real past failure), for
    demonstration/validation purposes.
    """
    model, contract = _load_gru_artifacts()
    if model is None:
        return unavailable_result("Stage II GRU model/feature contract not found for Digital Twin simulation")

    horizon_shifts = horizon_shifts or config.DIGITAL_TWIN_HORIZON_SHIFTS

    try:
        state = _current_state(machine_id, model, contract, sim_day=sim_day, shift=shift)
        if state is None:
            return unavailable_result(
                f"Not enough recorded history to build a state window for machine {machine_id}"
                + (f" as of {sim_day}/{shift}" if sim_day else ""))

        scenarios = [
            simulate_continue_operating(machine_id, model, contract, state, horizon_shifts),
            simulate_stop_for_maintenance(machine_id, model, contract, state, horizon_shifts),
            simulate_reduce_load(machine_id, model, contract, state, horizon_shifts),
        ]
    except Exception as exc:
        return error_result(f"Digital twin simulation failed for {machine_id}: {exc}")

    best = max(scenarios, key=lambda s: s["estimated_net_value_usd"])
    return {
        "status": "ok",
        "machine_id": machine_id,
        "machine_type": state["machine_type"],
        "horizon_shifts": horizon_shifts,
        "horizon_hours": horizon_shifts * config.SHIFT_DURATION_HOURS,
        "current_production_rate": round(state["production_rate"], 2),
        "current_failure_probability": round(state["current_failure_probability"], 4),
        "scenarios": scenarios,
        "recommended_scenario": best["scenario"],
        "recommendation_basis": "highest estimated_net_value_usd across the 3 simulated scenarios",
    }
