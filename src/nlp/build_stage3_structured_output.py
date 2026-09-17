"""Stage III - combine CV and NLP structured predictions into one
per-machine record, for later stages (multi-agent, digital twin, etc.) to
consume without needing to know about both source files.

Only includes fields actually supported by the existing data: machine_id,
latest CV prediction/confidence for that machine (from the synthetic image
set), and latest NLP prediction/confidence for that machine (from the
maintenance-note classifier). If a machine has no image or no maintenance
record, that side is simply omitted (null), not fabricated.

Run (after train_cv.py and train_nlp.py):
    python -m src.nlp.build_stage3_structured_output
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    with open(REPORTS_DIR / "stage3_cv_predictions.json") as f:
        cv_preds = pd.DataFrame(json.load(f))
    with open(REPORTS_DIR / "stage3_nlp_predictions.json") as f:
        nlp_preds = pd.DataFrame(json.load(f))

    machines = sorted(set(cv_preds["machine_id"]) | set(nlp_preds["machine_id"]))
    records = []
    for machine_id in machines:
        cv_rows = cv_preds[cv_preds["machine_id"] == machine_id]
        nlp_rows = nlp_preds[nlp_preds["machine_id"] == machine_id]

        record = {"machine_id": machine_id}
        if len(cv_rows):
            latest_cv = cv_rows.iloc[-1]
            record["cv_prediction"] = latest_cv["cv_prediction"]
            record["cv_confidence"] = latest_cv["cv_confidence"]
            record["cv_source_image"] = latest_cv["image_path"]
        else:
            record["cv_prediction"] = None
            record["cv_confidence"] = None
            record["cv_source_image"] = None

        if len(nlp_rows):
            nlp_rows_sorted = nlp_rows.sort_values("timestamp")
            latest_nlp = nlp_rows_sorted.iloc[-1]
            record["nlp_prediction"] = latest_nlp["nlp_prediction"]
            record["nlp_confidence"] = latest_nlp["nlp_confidence"]
            record["nlp_source_timestamp"] = latest_nlp["timestamp"]
        else:
            record["nlp_prediction"] = None
            record["nlp_confidence"] = None
            record["nlp_source_timestamp"] = None

        records.append(record)

    out_path = REPORTS_DIR / "stage3_structured_output.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Combined {len(records)} per-machine records -> {out_path}")


if __name__ == "__main__":
    main()
