"""
observability.py — structured debug tracing for the support agent.

Traces are emitted as JSON-lines to stderr when DEBUG=true.

Security: sensitive fields are scrubbed before logging.
Never logs: email, address, risk scores, warehouse notes, API keys.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from config import DEBUG

# Fields that must never appear in logs
_FORBIDDEN_KEYS = frozenset(
    {
        "email",
        "shipping_address",
        "risk_score",
        "warehouse_note",
        "support_tags",
        "internal",
        "customer",        # entire customer sub-object
        "OPENAI_API_KEY",
    }
)


def _scrub(obj: Any, depth: int = 0) -> Any:
    """
    Recursively remove forbidden keys from dicts.
    Stops at depth 10 to prevent pathological recursion.
    """
    if depth > 10:
        return obj
    if isinstance(obj, dict):
        return {
            k: _scrub(v, depth + 1)
            for k, v in obj.items()
            if k not in _FORBIDDEN_KEYS
        }
    if isinstance(obj, list):
        return [_scrub(i, depth + 1) for i in obj]
    return obj


def trace(event: str, data: dict[str, Any]) -> None:
    """
    Emit one structured log line to stderr when debug mode is active.

    Args:
        event: short label for the trace event (e.g. "retrieval", "tool_call")
        data:  arbitrary dict — will be scrubbed of sensitive fields
    """
    if not DEBUG:
        return
    payload = {
        "ts": round(time.time(), 3),
        "event": event,
        "data": _scrub(data),
    }
    try:
        print(json.dumps(payload, default=str), file=sys.stderr, flush=True)
    except Exception:
        pass  # never crash the agent due to a trace failure
