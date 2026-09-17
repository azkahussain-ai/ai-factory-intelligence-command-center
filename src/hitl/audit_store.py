"""Stage VIII — append-only audit storage for human decisions.

One JSON object per line (JSONL) in `reports/hitl_audit_log.jsonl` - simple,
inspectable, consistent with this project's existing lightweight-storage
choices (plain CSV/JSON everywhere else; MLflow's own SQLite is the only
database in the project, used only for MLflow itself). Records are only
ever appended, never rewritten or deleted, so the original AI
recommendation embedded in each record can never be altered after the fact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config import REPORTS_DIR

AUDIT_LOG_PATH = REPORTS_DIR / "hitl_audit_log.jsonl"


def append_record(record: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_all_records() -> list[dict]:
    if not AUDIT_LOG_PATH.exists():
        return []
    records = []
    with open(AUDIT_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_record(decision_id: str) -> dict | None:
    for record in load_all_records():
        if record.get("decision_id") == decision_id:
            return record
    return None


def records_for_machine(machine_id: str) -> list[dict]:
    return [r for r in load_all_records()
            if r.get("original_ai_recommendation", {}).get("machine_id") == machine_id]
