"""Stage V — shared structured schemas for agent communication.

Every agent returns a plain dict matching one of these shapes (documented
here rather than enforced with a heavy schema library, consistent with the
project's existing dependency-minimal style). `status` is always present:
"ok" when the agent produced a real result, "unavailable" when upstream
data was missing, "error" when the agent itself failed - never silently
replaced with a fabricated value.
"""
from __future__ import annotations

import time


def make_trace_entry(agent_name: str, status: str, input_summary: str,
                      output_summary: str, error: str | None = None) -> dict:
    return {
        "agent": agent_name,
        "status": status,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "error": error,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def unavailable_result(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def error_result(reason: str) -> dict:
    return {"status": "error", "reason": reason}
