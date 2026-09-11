"""
test_router.py — Unit tests for deterministic query routing.

Tests verify that queries are correctly routed to retrieval, the order tool,
or both (mixed-intent) based on NormalizedQuery signals without calling an LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from query_normalizer import NormalizedQuery, normalize_query
from router import RoutingDecision, route_query


# ---------------------------------------------------------------------------
# Test Suite: Policy Queries (Retrieval Only)
# ---------------------------------------------------------------------------


class TestRouterPolicyQueries:
    """Queries concerning policies should trigger retrieval only, not order tool."""

    def test_return_policy_query_retrieval_only(self):
        """Test 1: Return policy query -> needs_retrieval=True, needs_order_tool=False."""
        query = NormalizedQuery(
            original="What is the return policy?",
            normalized="What is the return policy?",
            detected_order_ids=[],
            detected_intents=["return_policy"],
            is_multi_intent=False,
        )
        decision = route_query(query)

        assert decision.needs_retrieval is True
        assert decision.needs_order_tool is False
        assert decision.order_ids == []
        assert decision.intents == ["return_policy"]
        assert not decision.is_multi_intent
        assert "retrieval needed for intents: ['return_policy']" in decision.reasoning

    @pytest.mark.parametrize(
        "intent",
        [
            "return_policy",
            "shipping",
            "warranty",
            "product_info",
            "price_adjustment",
            "membership",
            "cancellation",
            "damaged_item",
            "gift_card",
            "final_sale",
        ],
    )
    def test_all_retrieval_intents_route_to_retrieval_only(self, intent: str):
        """All defined retrieval intents without order ID should only require retrieval."""
        query = NormalizedQuery(
            original=f"Question about {intent}",
            normalized=f"question about {intent}",
            detected_order_ids=[],
            detected_intents=[intent],
            is_multi_intent=False,
        )
        decision = route_query(query)

        assert decision.needs_retrieval is True
        assert decision.needs_order_tool is False
        assert decision.order_ids == []


# ---------------------------------------------------------------------------
# Test Suite: Order Queries (Order Tool Only)
# ---------------------------------------------------------------------------


class TestRouterOrderQueries:
    """Queries with order ID and order-only intents should route to order tool only."""

    def test_order_id_only_tool_only(self):
        """Test 2: Order ID only -> needs_order_tool=True, needs_retrieval=False."""
        query = NormalizedQuery(
            original="Where is ORD-1001?",
            normalized="Where is ORD-1001?",
            detected_order_ids=["ORD-1001"],
            detected_intents=["order_status"],
            is_multi_intent=False,
        )
        decision = route_query(query)

        assert decision.needs_order_tool is True
        assert decision.needs_retrieval is False
        assert decision.order_ids == ["ORD-1001"]
        assert "order-only query, skipping retrieval" in decision.reasoning

    def test_multiple_order_ids_tool_only(self):
        """Multiple order IDs with order_status intent route to tool only."""
        query = NormalizedQuery(
            original="Check status for ORD-1001 and ORD-1002",
            normalized="check status for ORD-1001 and ORD-1002",
            detected_order_ids=["ORD-1001", "ORD-1002"],
            detected_intents=["order_status"],
            is_multi_intent=False,
        )
        decision = route_query(query)

        assert decision.needs_order_tool is True
        assert decision.needs_retrieval is False
        assert decision.order_ids == ["ORD-1001", "ORD-1002"]

    def test_order_status_intent_without_order_id_defaults_to_retrieval(self):
        """If user asks about order status without providing an ID, fallback to retrieval."""
        query = NormalizedQuery(
            original="where is my order?",
            normalized="where is my order?",
            detected_order_ids=[],
            detected_intents=["order_status"],
            is_multi_intent=False,
        )
        decision = route_query(query)

        # Since order_status is not in _RETRIEVAL_INTENTS and no order ID is present,
        # it falls back to default retrieval
        assert decision.needs_retrieval is True
        assert decision.needs_order_tool is False
        assert "no clear signal, defaulting to retrieval" in decision.reasoning


# ---------------------------------------------------------------------------
# Test Suite: Mixed Intent Queries (Hybrid Tool + Retrieval)
# ---------------------------------------------------------------------------


class TestRouterMixedIntent:
    """Queries combining order lookup and policy inquiries require both components."""

    def test_order_id_plus_policy_query_hybrid(self):
        """Test 3: Order ID + policy query -> both True (mixed intent)."""
        query = NormalizedQuery(
            original="Can I return ORD-1005?",
            normalized="can i return ORD-1005?",
            detected_order_ids=["ORD-1005"],
            detected_intents=["return_policy", "order_status"],
            is_multi_intent=True,
        )
        decision = route_query(query)

        assert decision.needs_retrieval is True
        assert decision.needs_order_tool is True
        assert decision.order_ids == ["ORD-1005"]
        assert decision.is_multi_intent is True
        assert "retrieval needed for intents: ['return_policy']" in decision.reasoning
        assert "order tool needed for IDs: ['ORD-1005']" in decision.reasoning

    def test_order_id_with_warranty_policy(self):
        """Order ID with warranty intent activates both retrieval and order tool."""
        query = NormalizedQuery(
            original="Is item in ORD-1010 covered under warranty?",
            normalized="is item in ORD-1010 covered under warranty?",
            detected_order_ids=["ORD-1010"],
            detected_intents=["warranty"],
            is_multi_intent=False,
        )
        decision = route_query(query)

        assert decision.needs_retrieval is True
        assert decision.needs_order_tool is True
        assert decision.order_ids == ["ORD-1010"]


# ---------------------------------------------------------------------------
# Test Suite: General and Ambiguous Queries
# ---------------------------------------------------------------------------


class TestRouterGeneralQueries:
    """General queries or queries without clear intent should default to retrieval."""

    def test_general_query_needs_retrieval(self):
        """Test 4: General query -> needs_retrieval=True, needs_order_tool=False."""
        query = NormalizedQuery(
            original="Hello, I have a general question",
            normalized="hello, i have a general question",
            detected_order_ids=[],
            detected_intents=["general"],
            is_multi_intent=False,
        )
        decision = route_query(query)

        assert decision.needs_retrieval is True
        assert decision.needs_order_tool is False
        assert decision.order_ids == []

    def test_empty_intents_fallback_to_retrieval(self):
        """Empty intents and no order IDs default to retrieval."""
        query = NormalizedQuery(
            original="",
            normalized="",
            detected_order_ids=[],
            detected_intents=[],
            is_multi_intent=False,
        )
        decision = route_query(query)

        assert decision.needs_retrieval is True
        assert decision.needs_order_tool is False
        assert "no clear signal, defaulting to retrieval" in decision.reasoning


# ---------------------------------------------------------------------------
# Test Suite: Follow-up Queries with Context
# ---------------------------------------------------------------------------


class TestRouterFollowUpWithContext:
    """Follow-up queries in existing conversation should retrieve using conversation context."""

    def test_follow_up_with_context_needs_retrieval(self):
        """Test 5: Follow-up with context -> needs_retrieval=True."""
        query = NormalizedQuery(
            original="What about that?",
            normalized="what about that?",
            detected_order_ids=[],
            detected_intents=["general"],
            is_multi_intent=False,
        )
        decision = route_query(query, has_conversation_context=True)

        assert decision.needs_retrieval is True
        assert decision.needs_order_tool is False
        assert "follow-up with context, enabling retrieval" in decision.reasoning

    def test_follow_up_with_context_and_order_id_needs_tool(self):
        """Follow-up with context that contains an order ID activates order tool."""
        query = NormalizedQuery(
            original="and what about ORD-1002?",
            normalized="and what about ORD-1002?",
            detected_order_ids=["ORD-1002"],
            detected_intents=["general"],
            is_multi_intent=False,
        )
        decision = route_query(query, has_conversation_context=True)

        assert decision.needs_order_tool is True
        assert decision.needs_retrieval is True
        assert decision.order_ids == ["ORD-1002"]


# ---------------------------------------------------------------------------
# Test Suite: RoutingDecision Data Structure & Integration
# ---------------------------------------------------------------------------


class TestRouterStructureAndIntegration:
    """Verify RoutingDecision attributes and integration with normalize_query."""

    def test_routing_decision_structure(self):
        """Verify that RoutingDecision preserves normalized query for retrieval."""
        query = NormalizedQuery(
            original="How do I return a damaged mug?",
            normalized="how do i return a damaged mug?",
            detected_order_ids=[],
            detected_intents=["return_policy", "damaged_item"],
            is_multi_intent=True,
        )
        decision = route_query(query)

        assert isinstance(decision, RoutingDecision)
        assert decision.retrieval_queries == ["how do i return a damaged mug?"]
        assert decision.intents == ["return_policy", "damaged_item"]
        assert decision.is_multi_intent is True

    def test_integration_with_normalize_query_policy(self):
        """End-to-end normalization to routing for policy query."""
        normalized = normalize_query("What is your refund policy?")
        decision = route_query(normalized)

        assert decision.needs_retrieval is True
        assert decision.needs_order_tool is False

    def test_integration_with_normalize_query_order(self):
        """End-to-end normalization to routing for order status query."""
        normalized = normalize_query("Track my package ORD-1003")
        decision = route_query(normalized)

        assert decision.needs_order_tool is True
        assert decision.needs_retrieval is False
        assert "ORD-1003" in decision.order_ids

    def test_integration_with_normalize_query_mixed(self):
        """End-to-end normalization to routing for mixed intent query."""
        normalized = normalize_query("Where is ORD-1004 and how long is the warranty?")
        decision = route_query(normalized)

        assert decision.needs_order_tool is True
        assert decision.needs_retrieval is True
        assert "ORD-1004" in decision.order_ids
