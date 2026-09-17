"""Stage V — Vision Agent.

Consumes Stage III's existing CV result from
`reports/stage3_structured_output.json`. Does NOT run new inference or
retrain the CNN - it only reads and structures the result Stage III already
produced, per the Stage V brief.

A small NLP info-reader lives here too (not one of the four required
agents - the brief lists NLP as information that feeds the Planning Agent
alongside the four agents, not a fifth required agent) for the same reason:
it reads Stage III's existing NLP result rather than rerunning the
classifier.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config import REPORTS_DIR
from src.agents.schemas import unavailable_result

STAGE3_OUTPUT_PATH = REPORTS_DIR / "stage3_structured_output.json"


def _load_stage3_record(machine_id: str) -> dict | None:
    if not STAGE3_OUTPUT_PATH.exists():
        return None
    try:
        records = json.loads(STAGE3_OUTPUT_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    for record in records:
        if record.get("machine_id") == machine_id:
            return record
    return None


def get_vision_result(machine_id: str) -> dict:
    """Structured Vision Agent result for one machine.

    `defect_type` and `severity` are reported as unavailable rather than
    invented: Stage III's CV model is a binary defect/normal classifier
    only, it never produced a defect-type or severity label.
    """
    record = _load_stage3_record(machine_id)
    if record is None or "cv_prediction" not in record:
        return unavailable_result(
            f"No Stage III CV result found for machine {machine_id} in {STAGE3_OUTPUT_PATH.name}")

    return {
        "status": "ok",
        "machine_id": machine_id,
        "defect_detected": record["cv_prediction"] == "defect",
        "defect_type": None,  # not classified by the Stage III model (binary defect/normal only)
        "severity": None,     # not classified by the Stage III model
        "confidence": round(float(record.get("cv_confidence", 0.0)), 4),
        "source_image": record.get("cv_source_image"),
        "source": "stage3_structured_output.json (existing Stage III CV result, not re-inferred)",
    }


def get_nlp_result(machine_id: str) -> dict:
    """Structured reader for Stage III's existing NLP classification of the
    machine's most recent maintenance note.
    """
    record = _load_stage3_record(machine_id)
    if record is None or "nlp_prediction" not in record:
        return unavailable_result(
            f"No Stage III NLP result found for machine {machine_id} in {STAGE3_OUTPUT_PATH.name}")

    return {
        "status": "ok",
        "machine_id": machine_id,
        "incident_type": record["nlp_prediction"],
        "confidence": round(float(record.get("nlp_confidence", 0.0)), 4),
        "source_timestamp": record.get("nlp_source_timestamp"),
        "source": "stage3_structured_output.json (existing Stage III NLP result, not re-inferred)",
    }
