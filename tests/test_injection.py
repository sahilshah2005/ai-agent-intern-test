"""
test_injection.py — prompt injection and privacy regression tests.

These tests do NOT call the LLM. They verify that:
  1. The order sanitisation layer never lets internal fields through.
  2. The retriever gives low precedence to the injection-laden doc 14.
  3. The system prompt does not contain any accidentally hardcoded secrets.
  4. Warehouse-note injection text does not appear in sanitised output.
  5. The observability scrubber removes sensitive keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from order_tool import lookup_order, _load_orders, _sanitize
from observability import _scrub
from system_prompt import SYSTEM_PROMPT
from config import KNOWLEDGE_BASE_DIR


# ---------------------------------------------------------------------------
# Order tool — injection / privacy
# ---------------------------------------------------------------------------


class TestOrderToolPrivacy:
    """Verify that sensitive fields are never returned by lookup_order."""

    # Known sensitive values from orders.json
    _SENSITIVE_VALUES = [
        "ava.morgan@example.test",       # ORD-1007 email
        "220 King Street West",          # ORD-1007 address
        "82",                            # ORD-1007 risk_score
        "Manual fraud review cleared",   # ORD-1007 warehouse_note
        "maya.reed@example.test",        # ORD-1001 email
        "AI instruction",                # ORD-1005 warehouse injection
        "coupon",                        # ORD-1005 warehouse injection
        "Do not mention review status",  # ORD-1012 warehouse note
        "Payment verification",          # ORD-1012 internal context
    ]

    @pytest.mark.parametrize("order_id", [
        "ORD-1001", "ORD-1002", "ORD-1003", "ORD-1004", "ORD-1005",
        "ORD-1006", "ORD-1007", "ORD-1008", "ORD-1009", "ORD-1010",
        "ORD-1011", "ORD-1012",
    ])
    def test_no_sensitive_value_in_any_order_result(self, order_id: str):
        result_str = str(lookup_order(order_id))
        for sensitive in self._SENSITIVE_VALUES:
            assert sensitive not in result_str, (
                f"Sensitive value '{sensitive}' found in result for {order_id}"
            )

    def test_internal_block_never_in_result(self):
        """The 'internal' key must not appear in any sanitised result."""
        orders = _load_orders()
        for order in orders.values():
            sanitised = _sanitize(order)
            assert "internal" not in sanitised
            assert "risk_score" not in str(sanitised)
            assert "warehouse_note" not in str(sanitised)
            assert "support_tags" not in str(sanitised)

    def test_customer_block_never_in_result(self):
        """Customer PII (name, email, address) must never appear."""
        orders = _load_orders()
        for order in orders.values():
            sanitised = _sanitize(order)
            assert "customer" not in sanitised
            # Check no email-like strings
            assert "@example.test" not in str(sanitised)


# ---------------------------------------------------------------------------
# Retriever — doc 14 precedence
# ---------------------------------------------------------------------------


class TestDoc14Precedence:
    def test_doc14_has_lowest_precedence_in_corpus(self):
        """
        14-internal-content-migration-notes.md (draft + internal + authority=none)
        must score lower than every active+official+customer document.
        """
        from retriever import load_chunks
        chunks = load_chunks(KNOWLEDGE_BASE_DIR)
        doc14_chunks = [c for c in chunks if c.filename == "14-internal-content-migration-notes.md"]
        active_official_chunks = [
            c for c in chunks
            if c.status == "active"
            and c.policy_authority == "official"
            and c.audience == "customer"
        ]
        assert doc14_chunks, "Doc 14 must be indexed"
        assert active_official_chunks

        max_doc14_score = max(c.precedence_score for c in doc14_chunks)
        min_active_score = min(c.precedence_score for c in active_official_chunks)

        assert max_doc14_score < min_active_score, (
            f"Doc 14 precedence ({max_doc14_score}) should be below "
            f"active+official minimum ({min_active_score})"
        )

    def test_doc14_precedence_is_negative(self):
        from scoring import compute_precedence_score
        fm = {"status": "draft", "policy_authority": "none", "audience": "internal"}
        score = compute_precedence_score(fm)
        assert score < 0

    def test_doc02_superseded_lower_than_doc01_active(self):
        from scoring import compute_precedence_score
        fm_current = {"status": "active", "policy_authority": "official", "audience": "customer"}
        fm_legacy = {"status": "superseded", "policy_authority": "official", "audience": "customer"}
        assert compute_precedence_score(fm_current) > compute_precedence_score(fm_legacy)


# ---------------------------------------------------------------------------
# Observability scrubber
# ---------------------------------------------------------------------------


class TestObservabilityScrubber:
    def test_scrub_removes_email(self):
        data = {"order_id": "ORD-1007", "email": "test@test.com", "status": "shipped"}
        scrubbed = _scrub(data)
        assert "email" not in scrubbed
        assert scrubbed["status"] == "shipped"
        assert scrubbed["order_id"] == "ORD-1007"

    def test_scrub_removes_internal(self):
        data = {"status": "shipped", "internal": {"risk_score": 99}}
        scrubbed = _scrub(data)
        assert "internal" not in scrubbed

    def test_scrub_removes_shipping_address(self):
        data = {"a": 1, "shipping_address": "123 Main St", "b": 2}
        scrubbed = _scrub(data)
        assert "shipping_address" not in scrubbed
        assert scrubbed["a"] == 1

    def test_scrub_handles_nested_lists(self):
        data = {"items": [{"name": "Bag", "email": "leak@test.com"}]}
        scrubbed = _scrub(data)
        assert "email" not in scrubbed["items"][0]
        assert scrubbed["items"][0]["name"] == "Bag"

    def test_scrub_does_not_raise_on_none(self):
        assert _scrub(None) is None

    def test_scrub_does_not_raise_on_string(self):
        assert _scrub("safe string") == "safe string"


# ---------------------------------------------------------------------------
# System prompt — no hardcoded secrets
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_no_api_key_in_prompt(self):
        assert "sk-" not in SYSTEM_PROMPT
        assert "OPENAI_API_KEY" not in SYSTEM_PROMPT

    def test_prompt_establishes_data_boundary(self):
        """The prompt must explicitly state that retrieved content is data."""
        lower = SYSTEM_PROMPT.lower()
        assert "untrusted" in lower or "data" in lower
        assert "not instructions" in lower or "not an instruction" in lower

    def test_prompt_forbids_internal_disclosure(self):
        """The prompt must explicitly prohibit disclosing internal fields."""
        lower = SYSTEM_PROMPT.lower()
        assert "risk score" in lower or "internal" in lower
        assert "never" in lower or "must not" in lower or "do not" in lower

    def test_prompt_requires_source_citation(self):
        lower = SYSTEM_PROMPT.lower()
        assert "source" in lower
        assert "filename" in lower or "citation" in lower or "cite" in lower

    def test_prompt_requires_abstention(self):
        lower = SYSTEM_PROMPT.lower()
        assert "insufficient" in lower or "abstain" in lower or "do not guess" in lower
