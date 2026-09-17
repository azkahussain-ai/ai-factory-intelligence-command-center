"""Stage VI — top-level XAI entrypoint.

`explain_machine(machine_id)` runs the predictive-maintenance explainer
(GRU permutation importance - the model Stage V's live agent actually
uses) and the vision explainer (Grad-CAM) for a machine, then assembles
them into the project's structured XAI schema with a dynamically generated
human-readable summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config import REPORTS_DIR
from src.xai import gru_explainer, gradcam_explainer
from src.xai.explanation_formatter import build_xai_result


def explain_machine(machine_id: str, sim_day: int | None = None, shift: str | None = None) -> dict:
    """Explain a machine's current (or, if `sim_day`/`shift` given, a
    specific historical) predictive-maintenance prediction and its most
    recent vision inspection.
    """
    pm_explanation = gru_explainer.explain(machine_id, sim_day=sim_day, shift=shift)
    vision_explanation = gradcam_explainer.explain(machine_id)
    return build_xai_result(machine_id, pm_explanation, vision_explanation)


def explain_and_save(machine_id: str, sim_day: int | None = None, shift: str | None = None) -> dict:
    result = explain_machine(machine_id, sim_day=sim_day, shift=shift)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"stage6_xai_{machine_id}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-id", default="M-01")
    parser.add_argument("--sim-day", type=int, default=None)
    parser.add_argument("--shift", default=None)
    args = parser.parse_args()
    result = explain_and_save(args.machine_id, sim_day=args.sim_day, shift=args.shift)
    print(json.dumps(result, indent=2, default=str))
