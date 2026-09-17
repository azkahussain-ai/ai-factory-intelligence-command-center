"""Stage VIII — Human-in-the-Loop pipeline entrypoint.

Runs the actual Stage V multi-agent workflow (with Stage VI/VII evidence
already folded in) for a machine, then records a human supervisor's
decision on it.

Run:
    python -m src.hitl.hitl_pipeline --machine-id M-01 --decision APPROVE
    python -m src.hitl.hitl_pipeline --machine-id M-01 --decision REJECT --comment "..."
    python -m src.hitl.hitl_pipeline --machine-id M-01 --decision MODIFY \
        --modified-action "..." --comment "..."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.agents.orchestrator import run_workflow
from src.hitl.hitl_workflow import submit_decision


def run_and_record(machine_id: str, decision: str, comment: str | None = None,
                    modified_action: str | None = None, supervisor: str | None = None) -> dict:
    """End-to-end: real prediction -> real XAI -> real multi-agent reasoning
    -> real digital-twin analysis -> AI recommendation -> human decision +
    audit record. Every upstream step is the actual existing Stage V-VII
    pipeline; nothing here recomputes or second-guesses their outputs.
    """
    ai_output = run_workflow(machine_id)
    record = submit_decision(
        ai_output, decision, comment=comment, modified_action=modified_action, supervisor=supervisor)
    return {"ai_output": ai_output, "audit_record": record}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-id", default="M-01")
    parser.add_argument("--decision", required=True, choices=["APPROVE", "REJECT", "MODIFY"])
    parser.add_argument("--comment", default=None)
    parser.add_argument("--modified-action", default=None)
    parser.add_argument("--supervisor", default=None)
    args = parser.parse_args()

    result = run_and_record(
        args.machine_id, args.decision, comment=args.comment,
        modified_action=args.modified_action, supervisor=args.supervisor)
    print(json.dumps(result["audit_record"], indent=2, default=str))
