"""Stage VII — Digital Twin pipeline entrypoint.

Run:
    python -m src.digital_twin.digital_twin_pipeline --machine-id M-01
    python -m src.digital_twin.digital_twin_pipeline --machine-id M-01 --sim-day 23 --shift Afternoon
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config import REPORTS_DIR
from src.digital_twin.simulator import run_scenarios


def simulate_and_save(machine_id: str, horizon_shifts: int | None = None,
                       sim_day: int | None = None, shift: str | None = None) -> dict:
    result = run_scenarios(machine_id, horizon_shifts=horizon_shifts, sim_day=sim_day, shift=shift)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"stage7_digital_twin_{machine_id}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-id", default="M-01")
    parser.add_argument("--horizon-shifts", type=int, default=None)
    parser.add_argument("--sim-day", type=int, default=None)
    parser.add_argument("--shift", default=None)
    args = parser.parse_args()
    result = simulate_and_save(args.machine_id, horizon_shifts=args.horizon_shifts,
                                sim_day=args.sim_day, shift=args.shift)
    print(json.dumps(result, indent=2, default=str))
