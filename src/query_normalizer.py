"""
query_normalizer.py — lightweight query normalization and intent detection.

Identifies useful structure in the raw user query without transforming
the customer's meaning. The original query is always preserved for
observability.

This module uses only deterministic keyword patterns — no LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Order ID extraction pattern
# ---------------------------------------------------------------------------

_ORDER_ID_RE = re.compile(r"\bORD[-\s]?\d{4}\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Intent keyword patterns
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: dict[str, list[str]] = {
    "order_status": [
        "order status", "where is my order", "track", "tracking",
        "shipped", "shipping status", "delivery status", "when will",
        "arrive", "estimated delivery", "eta", "carrier",
    ],
    "return_policy": [
        "return", "refund", "exchange", "send back", "return window",
        "return policy", "returning", "money back",
    ],
    "shipping": [
        "shipping", "delivery", "ship to", "how long", "transit",
        "international", "domestic", "canada", "free shipping",
    ],
    "warranty": [
        "warranty", "defect", "manufacturing defect", "broken",
        "repair", "warranty claim", "coverage",
    ],
    "product_info": [
        "product", "care", "cleaning", "wash", "dishwasher",
        "material", "breeze tumbler", "ridge daypack", "summit",
        "trailpouch",
    ],
    "price_adjustment": [
        "price adjustment", "price drop", "price match",
        "price difference", "price went down",
    ],
    "membership": [
        "trailplus", "trail plus", "member", "membership",
    ],
    "cancellation": [
        "cancel", "cancellation", "cancel order", "stop order",
        "change order", "modify order", "address change",
    ],
    "damaged_item": [
        "damaged", "wrong item", "defective", "broken",
        "incorrect item", "missing item",
    ],
    "gift_card": [
        "gift card", "gift certificate", "store credit",
    ],
    "final_sale": [
        "final sale", "clearance", "non-returnable",
    ],
}


# ---------------------------------------------------------------------------
# NormalizedQuery result
# ---------------------------------------------------------------------------


@dataclass
class NormalizedQuery:
    """Result of query normalization — preserves original for observability."""
    original: str
    normalized: str
    detected_order_ids: list[str] = field(default_factory=list)
    detected_intents: list[str] = field(default_factory=list)
    is_multi_intent: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_query(raw_query: str) -> NormalizedQuery:
    """
    Analyse a raw user query and extract structured signals.

    Returns a NormalizedQuery with:
      - original: the verbatim user input
      - normalized: cleaned for retrieval (whitespace, casing)
      - detected_order_ids: any ORD-XXXX patterns found
      - detected_intents: list of detected intent categories
      - is_multi_intent: True if 2+ distinct intents detected
    """
    original = raw_query
    cleaned = " ".join(raw_query.split())  # collapse whitespace

    # Extract order IDs
    order_ids = [
        _normalize_order_id_format(m.group())
        for m in _ORDER_ID_RE.finditer(cleaned)
    ]

    # Detect intents
    lower = cleaned.lower()
    intents: list[str] = []
    for intent, keywords in _INTENT_PATTERNS.items():
        if any(kw in lower for kw in keywords):
            intents.append(intent)

    # If an order ID is detected but no specific intent matched, assume order_status
    if order_ids and not intents:
        intents.append("order_status")

    # If nothing matched at all, mark as general
    if not intents:
        intents.append("general")

    return NormalizedQuery(
        original=original,
        normalized=cleaned,
        detected_order_ids=order_ids,
        detected_intents=intents,
        is_multi_intent=len(intents) >= 2,
    )


def _normalize_order_id_format(raw: str) -> str:
    """Normalise an extracted order ID to ORD-XXXX format."""
    s = raw.strip().upper()
    m = re.match(r"^ORD[-\s]?(\d+)$", s)
    if m:
        return f"ORD-{m.group(1)}"
    return s
