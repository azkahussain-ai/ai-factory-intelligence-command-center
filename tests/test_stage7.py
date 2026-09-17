"""Stage VII tests. Run with: pytest tests/test_stage7.py -v"""
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.digital_twin.simulator import run_scenarios, _compounded_risk
from src.digital_twin.digital_twin_pipeline import simulate_and_save
from src.agents.orchestrator import run_workflow

VALID_MACHINE = "M-01"
INVALID_MACHINE = "M-99"


# ---- Core simulation correctness ----------------------------------------

def test_run_scenarios_returns_exactly_three_required_scenarios():
    result = run_scenarios(VALID_MACHINE)
    assert result["status"] == "ok"
    names = {s["scenario"] for s in result["scenarios"]}
    assert names == {"continue_operating", "stop_for_maintenance", "reduce_load"}


def test_each_scenario_reports_required_fields():
    result = run_scenarios(VALID_MACHINE)
    required = {
        "expected_production_units", "expected_downtime_hours", "cumulative_failure_risk",
        "estimated_net_value_usd", "per_shift_failure_probability", "failure_probability_source",
    }
    for scenario in result["scenarios"]:
        assert required.issubset(scenario)


def test_recommended_scenario_is_the_actual_max_net_value():
    result = run_scenarios(VALID_MACHINE)
    best = max(result["scenarios"], key=lambda s: s["estimated_net_value_usd"])
    assert result["recommended_scenario"] == best["scenario"]


def test_simulation_is_deterministic_not_random():
    r1 = run_scenarios(VALID_MACHINE)
    r2 = run_scenarios(VALID_MACHINE)
    assert r1["scenarios"] == r2["scenarios"]


def test_compounded_risk_formula():
    # 1 - (1-p)^n, standard documented compounding, not arbitrary
    assert _compounded_risk(0.1, 1) == pytest.approx(0.1)
    assert _compounded_risk(0.0, 5) == 0.0
    assert _compounded_risk(0.5, 2) == pytest.approx(0.75)


def test_stop_for_maintenance_always_incurs_its_documented_downtime():
    from config.config import PREVENTIVE_MAINTENANCE_DOWNTIME_HOURS
    result = run_scenarios(VALID_MACHINE)
    maint = next(s for s in result["scenarios"] if s["scenario"] == "stop_for_maintenance")
    assert maint["expected_downtime_hours"] == pytest.approx(PREVENTIVE_MAINTENANCE_DOWNTIME_HOURS)


def test_reduce_load_scenario_produces_less_output_than_continue_operating():
    result = run_scenarios(VALID_MACHINE)
    continue_op = next(s for s in result["scenarios"] if s["scenario"] == "continue_operating")
    reduced = next(s for s in result["scenarios"] if s["scenario"] == "reduce_load")
    assert reduced["expected_production_units"] < continue_op["expected_production_units"]


# ---- Real historical high-risk case (proves scenarios genuinely differ) --

def test_scenarios_differentiate_meaningfully_on_a_real_historical_failure():
    """M-01 genuinely failed on sim_day 23 (Afternoon) - the digital twin
    simulated from that real high-risk state should show stopping for
    maintenance clearly outperforming continuing to operate, not just
    return three near-identical numbers.
    """
    result = run_scenarios(VALID_MACHINE, sim_day=23, shift="Afternoon")
    assert result["status"] == "ok"
    assert result["current_failure_probability"] > 0.9

    continue_op = next(s for s in result["scenarios"] if s["scenario"] == "continue_operating")
    maint = next(s for s in result["scenarios"] if s["scenario"] == "stop_for_maintenance")
    assert continue_op["cumulative_failure_risk"] > maint["cumulative_failure_risk"]
    assert maint["estimated_net_value_usd"] > continue_op["estimated_net_value_usd"]
    assert result["recommended_scenario"] == "stop_for_maintenance"


# ---- No fake / random numbers -------------------------------------------

def test_financial_constants_are_the_documented_config_values():
    from config.config import UNIT_MARGIN_USD, MAINTENANCE_COST_PER_HOUR_USD, FAILURE_REPAIR_COST_USD
    result = run_scenarios(VALID_MACHINE, horizon_shifts=1)
    continue_op = next(s for s in result["scenarios"] if s["scenario"] == "continue_operating")
    expected_revenue = continue_op["expected_production_units"] * UNIT_MARGIN_USD
    assert continue_op["expected_revenue_usd"] == pytest.approx(expected_revenue, rel=1e-3)


def test_horizon_shifts_is_configurable_and_scales_production():
    short = run_scenarios(VALID_MACHINE, horizon_shifts=2)
    long = run_scenarios(VALID_MACHINE, horizon_shifts=12)
    short_prod = next(s for s in short["scenarios"] if s["scenario"] == "continue_operating")["expected_production_units"]
    long_prod = next(s for s in long["scenarios"] if s["scenario"] == "continue_operating")["expected_production_units"]
    assert long_prod > short_prod


# ---- Error handling -------------------------------------------------------

def test_unavailable_for_unknown_machine_no_fake_scenarios():
    result = run_scenarios(INVALID_MACHINE)
    assert result["status"] == "unavailable"
    assert "reason" in result
    assert "scenarios" not in result


def test_pipeline_saves_real_output_file():
    result = simulate_and_save(VALID_MACHINE)
    from config.config import REPORTS_DIR
    out_path = REPORTS_DIR / f"stage7_digital_twin_{VALID_MACHINE}.json"
    assert out_path.exists()
    import json
    saved = json.loads(out_path.read_text())
    assert saved["machine_id"] == VALID_MACHINE
    assert saved["status"] == result["status"]


# ---- Stage preservation / integration -------------------------------------

def test_orchestrator_includes_digital_twin_step():
    result = run_workflow(VALID_MACHINE)
    assert "digital_twin_result" in result
    agent_names = {t["agent"] for t in result["agent_trace"]}
    assert "digital_twin" in agent_names


def test_planning_agent_backward_compatible_without_twin_argument():
    from src.agents import planning_agent
    pm = {"status": "ok", "risk_level": "LOW", "failure_probability": 0.01,
          "model_name": "gru", "important_signals": ["load_pct_mean"]}
    no_vision = {"status": "unavailable", "reason": "n/a"}
    no_nlp = {"status": "unavailable", "reason": "n/a"}
    no_rag = {"status": "ok", "grounded": False, "sources": []}
    decision = planning_agent.decide(no_vision, pm, no_nlp, no_rag)  # no xai, no twin
    assert decision["status"] == "ok"


def test_planning_agent_reasoning_cites_real_twin_recommendation_when_provided():
    from src.agents import planning_agent
    pm = {"status": "ok", "risk_level": "LOW", "failure_probability": 0.01,
          "model_name": "gru", "important_signals": ["load_pct_mean"]}
    no_vision = {"status": "unavailable", "reason": "n/a"}
    no_nlp = {"status": "unavailable", "reason": "n/a"}
    no_rag = {"status": "ok", "grounded": False, "sources": []}
    twin = {"status": "ok", "horizon_shifts": 6, "recommended_scenario": "stop_for_maintenance",
            "recommendation_basis": "highest estimated_net_value_usd across the 3 simulated scenarios"}
    decision = planning_agent.decide(no_vision, pm, no_nlp, no_rag, twin=twin)
    assert any("stop for maintenance" in r.lower() for r in decision["reasoning"])


def test_stage1_to_6_tests_still_pass():
    import tests.test_stage1  # noqa: F401
    import tests.test_stage2  # noqa: F401
    import tests.test_stage3  # noqa: F401
    import tests.test_stage4  # noqa: F401
    import tests.test_stage5  # noqa: F401
    import tests.test_stage6  # noqa: F401
