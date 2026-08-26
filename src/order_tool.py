"""
order_tool.py — deterministic order-status lookup tool.

Design principles
─────────────────
• The LLM never receives the raw orders database.
• Only a sanitized, customer-safe subset of fields is returned.
• Status takes precedence over stale operational fields.
• Internal fields (email, address, risk score, warehouse notes) are
  never exposed to the model or the customer.
• Tool results are UNTRUSTED DATA — warehouse-note injection attempts
  are ignored because the agent's system prompt handles that boundary.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import ORDERS_FILE

# ---------------------------------------------------------------------------
# Field whitelists
# ---------------------------------------------------------------------------

# Fields the model is allowed to see from the top-level order record
_SAFE_TOP_FIELDS: frozenset[str] = frozenset(
    {
        "order_id",
        "membership_tier",
        "items",
        "placed_at",
        "status",
        "status_updated_at",
        "shipped_at",
        "delivered_at",
        "carrier",
        "tracking_number",
        "estimated_delivery",
        "customer_safe_message",
    }
)

# Fields the model is allowed to see inside each item
_SAFE_ITEM_FIELDS: frozenset[str] = frozenset({"name", "quantity", "final_sale"})

# Statuses where carrier / tracking / ETA fields are stale and must be hidden
_STALE_STATUSES: frozenset[str] = frozenset({"cancelled", "returned"})

# ---------------------------------------------------------------------------
# Loader (lazy, cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_orders() -> dict[str, dict[str, Any]]:
    """Load orders.json and index by normalised uppercase order_id."""
    raw = json.loads(Path(ORDERS_FILE).read_text(encoding="utf-8"))
    return {o["order_id"].upper(): o for o in raw.get("orders", [])}


# ---------------------------------------------------------------------------
# Normalisation & validation
# ---------------------------------------------------------------------------


def normalize_order_id(raw: str) -> str:
    """
    Normalise harmless input differences:
      ord-1007  →  ORD-1007
      ORD 1007  →  ORD-1007
      ORD1007   →  ORD-1007
      '  ORD-1007  '  →  ORD-1007

    Does NOT guess substantially different IDs.
    """
    s = raw.strip().upper()
    m = re.match(r"^ORD[-\s]?(\d+)$", s)
    if m:
        return f"ORD-{m.group(1)}"
    return s  # return stripped+uppercased; validation will reject if still invalid


def _is_valid_format(order_id: str) -> bool:
    return bool(re.match(r"^ORD-\d+$", order_id))


# ---------------------------------------------------------------------------
# Sanitisation
# ---------------------------------------------------------------------------


def _sanitize(order: dict[str, Any]) -> dict[str, Any]:
    """
    Return a customer-safe copy of an order record.

    Rules applied:
    1. Only whitelisted top-level fields are included.
    2. item objects are filtered to whitelisted sub-fields.
    3. For cancelled/returned orders, stale operational fields are removed.
    4. Internal sub-object is never included.
    """
    status = str(order.get("status", "")).lower()

    result: dict[str, Any] = {}

    for key in _SAFE_TOP_FIELDS:
        if key not in order:
            continue

        if key == "items":
            result["items"] = [
                {k: v for k, v in item.items() if k in _SAFE_ITEM_FIELDS}
                for item in order.get("items", [])
            ]
        else:
            result[key] = order[key]

    # Status-aware suppression of stale fields
    if status in _STALE_STATUSES:
        result.pop("carrier", None)
        result.pop("tracking_number", None)
        result.pop("estimated_delivery", None)

    return result


# ---------------------------------------------------------------------------
# Public lookup function
# ---------------------------------------------------------------------------


def lookup_order(order_id: str) -> dict[str, Any]:
    """
    Deterministic order lookup.

    Returns either:
      {"order_id": "ORD-XXXX", "result": <sanitized order dict>}
    or an error dict:
      {"error": "<code>", "message": "<human-readable>", ...}

    Never raises; errors are surfaced as structured dicts so the LLM
    can produce a helpful response without hallucinating.
    """
    # --- missing ---
    if not order_id or not order_id.strip():
        return {
            "error": "missing_order_id",
            "message": "An order ID is required to look up an order.",
        }

    normalized = normalize_order_id(order_id)

    # --- malformed ---
    if not _is_valid_format(normalized):
        return {
            "error": "malformed_order_id",
            "message": (
                f"'{order_id}' does not look like a valid order ID. "
                "Order IDs follow the format ORD-1234."
            ),
        }

    orders = _load_orders()
    order = orders.get(normalized)

    # --- not found ---
    if order is None:
        return {
            "error": "not_found",
            "message": (
                f"No order with ID {normalized} was found. "
                "Please double-check the order ID, or contact support if you believe this is an error."
            ),
            "order_id": normalized,
        }

    # --- success ---
    return {
        "order_id": normalized,
        "result": _sanitize(order),
    }
