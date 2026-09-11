"""
router.py — deterministic intent routing and mixed-intent support.

Routes queries to the appropriate pipeline components (retrieval, tool,
or both) based on signals from the query normalizer. Does NOT use the
LLM for routing — all decisions are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from query_normalizer import NormalizedQuery


# ---------------------------------------------------------------------------
# Routing result
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """Describes which pipeline components to activate for a query."""
    needs_retrieval: bool = True
    needs_order_tool: bool = False
    order_ids: list[str] = field(default_factory=list)
    retrieval_queries: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    is_multi_intent: bool = False
    reasoning: str = ""


# Intents that require knowledge-base retrieval
_RETRIEVAL_INTENTS = frozenset({
    "return_policy", "shipping", "warranty", "product_info",
    "price_adjustment", "membership", "cancellation", "damaged_item",
    "gift_card", "final_sale", "general",
})

# Intents that require order tool
_ORDER_INTENTS = frozenset({
    "order_status",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route_query(
    normalized: NormalizedQuery,
    has_conversation_context: bool = False,
) -> RoutingDecision:
    """
    Determine which pipeline components to activate.

    Rules:
      1. Order ID detected + only order intents → tool only
      2. Policy keywords + no order ID → retrieval only
      3. Both detected → hybrid (tool + retrieval)
      4. No clear signal → full pipeline with retrieval
      5. Follow-up without clear intent → retrieval (use conversation context)
    """
    intents = normalized.detected_intents
    order_ids = normalized.detected_order_ids

    needs_retrieval = False
    needs_tool = False
    reasoning_parts: list[str] = []

    # Check if any retrieval intents are present
    retrieval_intents = [i for i in intents if i in _RETRIEVAL_INTENTS]
    order_intents = [i for i in intents if i in _ORDER_INTENTS]

    if retrieval_intents:
        needs_retrieval = True
        reasoning_parts.append(
            f"retrieval needed for intents: {retrieval_intents}"
        )

    if order_ids:
        needs_tool = True
        reasoning_parts.append(
            f"order tool needed for IDs: {order_ids}"
        )

    if order_intents and order_ids:
        needs_tool = True
        if not retrieval_intents:
            reasoning_parts.append("order-only query, skipping retrieval")

    # If nothing was detected, default to retrieval
    if not needs_retrieval and not needs_tool:
        needs_retrieval = True
        reasoning_parts.append("no clear signal, defaulting to retrieval")

    # For follow-up queries with conversation context, always retrieve
    if has_conversation_context and not needs_tool and "general" in intents:
        needs_retrieval = True
        reasoning_parts.append("follow-up with context, enabling retrieval")

    return RoutingDecision(
        needs_retrieval=needs_retrieval,
        needs_order_tool=needs_tool,
        order_ids=order_ids,
        retrieval_queries=[normalized.normalized],
        intents=intents,
        is_multi_intent=normalized.is_multi_intent,
        reasoning="; ".join(reasoning_parts),
    )
