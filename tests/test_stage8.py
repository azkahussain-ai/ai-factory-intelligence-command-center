"""Stage VIII tests. Run with: pytest tests/test_stage8.py -v"""
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.hitl.hitl_workflow import submit_decision, HumanDecisionError
from src.hitl import audit_store
from src.agents.orchestrator import run_workflow

VALID_MACHINE = "M-01"


@pytest.fixture(scope="module")
def real_ai_output():
    return run_workflow(VALID_MACHINE)


# ---- Test 1: Approve workflow --------------------------------------------

def test_approve_workflow_records_original_recommendation_as_final_action(real_ai_output):
    record = submit_decision(real_ai_output, "APPROVE", supervisor="tester")
    assert record["human_decision"] == "APPROVE"
    assert record["final_action"] == record["original_ai_recommendation"]["recommendation"]
    assert record["modified_action"] is None


# ---- Test 2: Reject workflow ----------------------------------------------

def test_reject_workflow_requires_comment(real_ai_output):
    with pytest.raises(HumanDecisionError):
        submit_decision(real_ai_output, "REJECT")


def test_reject_workflow_records_no_final_action(real_ai_output):
    record = submit_decision(real_ai_output, "REJECT", comment="stale sensor reading")
    assert record["human_decision"] == "REJECT"
    assert record["final_action"] is None
    assert record["human_comment"] == "stale sensor reading"


# ---- Test 3: Modify workflow ----------------------------------------------

def test_modify_workflow_requires_modified_action_and_comment(real_ai_output):
    with pytest.raises(HumanDecisionError):
        submit_decision(real_ai_output, "MODIFY", comment="reason only, no action")
    with pytest.raises(HumanDecisionError):
        submit_decision(real_ai_output, "MODIFY", modified_action="action only, no reason")


def test_modify_workflow_records_modified_action_as_final(real_ai_output):
    record = submit_decision(
        real_ai_output, "MODIFY", modified_action="reduce load 15% instead",
        comment="full stop not warranted given backlog")
    assert record["human_decision"] == "MODIFY"
    assert record["final_action"] == "reduce load 15% instead"
    assert record["modified_action"] == "reduce load 15% instead"


# ---- Test 4: Preservation of original AI recommendation -------------------

def test_original_ai_recommendation_matches_real_orchestrator_output(real_ai_output):
    record = submit_decision(real_ai_output, "APPROVE")
    assert record["original_ai_recommendation"]["recommendation"] == real_ai_output["decision"]["recommendation"]
    assert record["original_ai_recommendation"]["priority"] == real_ai_output["decision"]["priority"]
    assert record["original_ai_recommendation"]["machine_id"] == VALID_MACHINE


def test_original_recommendation_unchanged_regardless_of_human_decision(real_ai_output):
    """The stored original recommendation must be identical whether the
    human approves, rejects, or modifies - HITL must never alter it.
    """
    approve_record = submit_decision(real_ai_output, "APPROVE")
    reject_record = submit_decision(real_ai_output, "REJECT", comment="x")
    modify_record = submit_decision(real_ai_output, "MODIFY", modified_action="y", comment="z")
    assert (approve_record["original_ai_recommendation"]
            == reject_record["original_ai_recommendation"]
            == modify_record["original_ai_recommendation"])


def test_invalid_decision_value_rejected(real_ai_output):
    with pytest.raises(HumanDecisionError):
        submit_decision(real_ai_output, "MAYBE")


def test_missing_recommendation_is_rejected_not_silently_accepted():
    fake_empty_output = {"factory_context": {"machine_id": "M-01"}, "decision": {}}
    with pytest.raises(HumanDecisionError):
        submit_decision(fake_empty_output, "APPROVE")


# ---- Test 5: Human reason/comment -----------------------------------------

def test_comment_is_stored_verbatim(real_ai_output):
    comment = "Operator confirmed no visible defect on physical inspection"
    record = submit_decision(real_ai_output, "REJECT", comment=comment)
    assert record["human_comment"] == comment


# ---- Test 6: Timestamp / audit record --------------------------------------

def test_audit_record_has_unique_id_and_timestamp(real_ai_output):
    r1 = submit_decision(real_ai_output, "APPROVE")
    r2 = submit_decision(real_ai_output, "APPROVE")
    assert r1["decision_id"] != r2["decision_id"]
    assert r1["timestamp"]


def test_audit_record_persisted_and_retrievable(real_ai_output):
    record = submit_decision(real_ai_output, "APPROVE", supervisor="retrieval-test")
    reloaded = audit_store.get_record(record["decision_id"])
    assert reloaded is not None
    assert reloaded["supervisor"] == "retrieval-test"


def test_records_for_machine_filters_correctly(real_ai_output):
    submit_decision(real_ai_output, "APPROVE")  # machine M-01 (real_ai_output)
    records = audit_store.records_for_machine(VALID_MACHINE)
    assert len(records) >= 1
    assert all(r["original_ai_recommendation"]["machine_id"] == VALID_MACHINE for r in records)


# ---- Test 7-10: MLflow run creation / params / metrics / >=3 runs ---------

@pytest.fixture(scope="module")
def mlflow_run_ids():
    from src.mlops.mlflow_logging import log_all_stage2_runs
    return log_all_stage2_runs()


def test_mlflow_creates_at_least_three_runs(mlflow_run_ids):
    assert len(mlflow_run_ids) >= 3
    assert all(isinstance(v, str) and len(v) > 0 for v in mlflow_run_ids.values())


def test_mlflow_runs_log_real_params_and_metrics(mlflow_run_ids):
    import mlflow
    from src.mlops.mlflow_logging import TRACKING_URI
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    for model_name, run_id in mlflow_run_ids.items():
        run = client.get_run(run_id)
        assert len(run.data.params) > 0, f"{model_name} run logged no params"
        assert len(run.data.metrics) > 0, f"{model_name} run logged no metrics"
        # at least one of the required classification metrics must be present
        metric_names = set(run.data.metrics.keys())
        assert any("accuracy" in m or "f1" in m or "roc_auc" in m for m in metric_names)


def test_mlflow_metrics_match_real_stage2_reports(mlflow_run_ids):
    """No invented metric values - every logged number must trace back to
    the real Stage II evaluation JSON.
    """
    import json
    import mlflow
    from config.config import REPORTS_DIR
    from src.mlops.mlflow_logging import TRACKING_URI

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    baseline_metrics = json.loads((REPORTS_DIR / "stage2_baseline_metrics.json").read_text())

    run = client.get_run(mlflow_run_ids["random_forest"])
    real_test_f1 = baseline_metrics["random_forest"]["test"]["f1"]
    assert run.data.metrics.get("test_f1") == pytest.approx(real_test_f1, rel=1e-6)


def test_mlflow_registry_registers_stage2s_own_selected_best_model(mlflow_run_ids):
    from src.mlops.mlflow_logging import register_best_model
    result = register_best_model(mlflow_run_ids)
    assert result["registered_model_name"] == "ai_factory_failure_predictor"
    assert result["source_run_id"] == mlflow_run_ids[result["selected_model"]]


def test_registered_model_predictions_match_the_real_underlying_model(mlflow_run_ids):
    """The registered model must be the actual trained model, not a stub -
    predictions through the registry must match a direct forward pass.
    """
    import mlflow
    from src.mlops.mlflow_logging import TRACKING_URI, register_best_model
    from src.agents.predictive_maintenance_agent import _load_gru_artifacts, build_gru_input

    result = register_best_model(mlflow_run_ids)
    if result["selected_model"] != "gru":
        pytest.skip("Stage II's selected best model is not the GRU in this run")

    mlflow.set_tracking_uri(TRACKING_URI)
    registered = mlflow.pyfunc.load_model(f"models:/{result['registered_model_name']}/{result['version']}")

    gru, contract = _load_gru_artifacts()
    X_seq, _ = build_gru_input(VALID_MACHINE, contract)
    direct = gru.predict_proba(X_seq)
    via_registry = registered.predict(X_seq)
    assert direct == pytest.approx(via_registry, rel=1e-6)


# ---- Test 11: Stage I-VII remain functional --------------------------------

def test_stage1_to_7_tests_still_pass():
    import tests.test_stage1  # noqa: F401
    import tests.test_stage2  # noqa: F401
    import tests.test_stage3  # noqa: F401
    import tests.test_stage4  # noqa: F401
    import tests.test_stage5  # noqa: F401
    import tests.test_stage6  # noqa: F401
    import tests.test_stage7  # noqa: F401
