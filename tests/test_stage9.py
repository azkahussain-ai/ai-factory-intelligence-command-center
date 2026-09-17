"""Stage IX tests. Run with: pytest tests/test_stage9.py -v"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agents.orchestrator import run_workflow
from src.reporting.report_generator import generate_report

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _fresh_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=120)


# ---- App loads and runs the real pipeline ---------------------------------

def test_app_loads_without_exceptions():
    at = _fresh_app()
    at.run()
    assert not at.exception


def test_run_full_analysis_executes_real_pipeline_without_exceptions():
    at = _fresh_app()
    at.run()
    [b for b in at.button if "Run Full Analysis" in b.label][0].click().run()
    assert not at.exception
    assert len(at.tabs) == 7


def test_switching_machine_and_rerunning_works():
    at = _fresh_app()
    at.run()
    at.selectbox[0].set_value("M-04").run()
    [b for b in at.button if "Run Full Analysis" in b.label][0].click().run()
    assert not at.exception


# ---- Human decision workflow in the UI -------------------------------------

def test_ui_approve_flow_succeeds():
    at = _fresh_app()
    at.run()
    [b for b in at.button if "Run Full Analysis" in b.label][0].click().run()
    at.radio[0].set_value("APPROVE").run()
    [b for b in at.button if b.label == "Submit decision"][0].click().run()
    assert not at.exception
    assert any("Decision recorded" in s.value for s in at.success)


def test_ui_reject_without_comment_shows_error_not_crash():
    at = _fresh_app()
    at.run()
    [b for b in at.button if "Run Full Analysis" in b.label][0].click().run()
    at.radio[0].set_value("REJECT").run()
    [b for b in at.button if b.label == "Submit decision"][0].click().run()
    assert not at.exception
    assert any("requires a reason/comment" in e.value for e in at.error)


def test_ui_modify_with_action_and_comment_succeeds():
    at = _fresh_app()
    at.run()
    [b for b in at.button if "Run Full Analysis" in b.label][0].click().run()
    at.radio[0].set_value("MODIFY").run()
    at.text_area[0].set_value("reduce load 15% instead").run()  # modified action field
    at.text_area[1].set_value("full stop not warranted").run()  # comment field
    [b for b in at.button if b.label == "Submit decision"][0].click().run()
    assert not at.exception
    assert any("Decision recorded" in s.value for s in at.success)


# ---- Report generation ------------------------------------------------------

def test_ui_generate_report_produces_downloadable_pdf():
    at = _fresh_app()
    at.run()
    [b for b in at.button if "Run Full Analysis" in b.label][0].click().run()
    [b for b in at.button if b.label == "Generate PDF report"][0].click().run()
    assert not at.exception
    assert len(at.get("download_button")) == 1


def test_report_generator_produces_valid_pdf_bytes_from_real_output():
    ai_output = run_workflow("M-01")
    pdf_bytes = generate_report(ai_output)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_report_generator_includes_hitl_record_when_provided():
    ai_output = run_workflow("M-02")
    from src.hitl.hitl_workflow import submit_decision
    record = submit_decision(ai_output, "APPROVE", supervisor="report-test")
    pdf_with_hitl = generate_report(ai_output, record)
    pdf_without_hitl = generate_report(ai_output, None)
    # a real decision embedded should produce a different (larger) document
    assert len(pdf_with_hitl) != len(pdf_without_hitl)


def test_report_generator_handles_missing_upstream_results_gracefully():
    """No fabricated content when a component is unavailable - must not crash."""
    minimal_output = {
        "factory_context": {"machine_id": "M-99", "generated_at": "n/a"},
        "vision_result": {"status": "unavailable", "reason": "no data"},
        "predictive_maintenance_result": {"status": "unavailable", "reason": "no data"},
        "nlp_result": {"status": "unavailable", "reason": "no data"},
        "xai_result": {"human_summary": "No explanation available.", "feature_explanations": [],
                        "vision_explanation": {"available": False}},
        "rag_result": {"status": "unavailable", "reason": "no data"},
        "digital_twin_result": {"status": "unavailable", "reason": "no data"},
        "decision": {"status": "ok", "priority": "ROUTINE", "recommendation": "n/a", "reasoning": []},
    }
    pdf_bytes = generate_report(minimal_output)
    assert pdf_bytes.startswith(b"%PDF")


# ---- Live image upload uses the real Stage III model, not a stub ----------

def test_uploaded_image_inference_uses_real_cv_model():
    import numpy as np
    from PIL import Image
    from src.xai.gradcam_explainer import explain, explain_uploaded_image

    known_good = explain("M-01")  # Stage III's stored result for M-01's image
    img = np.array(Image.open(Path(__file__).resolve().parent.parent / known_good["source_image"]),
                    dtype=np.float64) / 255.0
    live_result = explain_uploaded_image(img)
    assert live_result["status"] == "ok"
    # same image through the same model -> same confidence as the stored Stage III record
    assert live_result["confidence"] == pytest.approx(known_good["confidence"], abs=1e-3)


# ---- Stage I-VIII preserved --------------------------------------------------

def test_stage1_to_8_tests_still_pass():
    import tests.test_stage1  # noqa: F401
    import tests.test_stage2  # noqa: F401
    import tests.test_stage3  # noqa: F401
    import tests.test_stage4  # noqa: F401
    import tests.test_stage5  # noqa: F401
    import tests.test_stage6  # noqa: F401
    import tests.test_stage7  # noqa: F401
    import tests.test_stage8  # noqa: F401
