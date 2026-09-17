"""Stage IV — optional integration with Stage II/III outputs.

Reads (never modifies) reports/stage3_structured_output.json so a
machine-specific RAG question can be answered with awareness of that
machine's current CV/NLP findings, per the Stage IV brief: "The RAG layer
should be capable of receiving relevant outputs from the completed earlier
stages... Do not modify the predictive models from Stage II or the CV/NLP
models from Stage III."
"""
from __future__ import annotations

import json
from pathlib import Path

from config.config import REPORTS_DIR


def load_machine_context(machine_id: str, reports_dir: Path = REPORTS_DIR) -> dict | None:
    """Return the Stage III structured-output record for a machine, or None
    if the file is missing or the machine_id is not found (never raises —
    this is optional context, not required for RAG to function)."""
    path = reports_dir / "stage3_structured_output.json"
    if not path.exists():
        return None
    try:
        records = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    for record in records:
        if record.get("machine_id") == machine_id:
            return record
    return None


def format_machine_context(record: dict | None) -> str | None:
    if record is None:
        return None
    details = []
    if "cv_prediction" in record:
        details.append(
            f"vision inspection = {record['cv_prediction']} "
            f"(confidence {record.get('cv_confidence', 0):.2f})"
        )
    if "nlp_prediction" in record:
        details.append(
            f"latest maintenance-note classification = {record['nlp_prediction']} "
            f"(confidence {record.get('nlp_confidence', 0):.2f})"
        )
    if not details:
        return f"Machine {record.get('machine_id')}: no Stage III findings available."
    return f"Machine {record.get('machine_id')}: " + ", ".join(details)
