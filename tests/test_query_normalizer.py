"""
test_query_normalizer.py — unit tests for query normalization and intent detection.

All tests are deterministic and self-contained with no LLM or network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from query_normalizer import (
    NormalizedQuery,
    _normalize_order_id_format,
    normalize_query,
)


# ---------------------------------------------------------------------------
# 1. Empty & Whitespace Queries -> general intent
# ---------------------------------------------------------------------------


class TestEmptyAndGeneralQueries:
    """Test empty queries, whitespace queries, and unmatched queries defaulting to 'general'."""

    def test_empty_string_yields_general_intent(self):
        result = normalize_query("")
        assert isinstance(result, NormalizedQuery)
        assert result.original == ""
        assert result.normalized == ""
        assert result.detected_order_ids == []
        assert result.detected_intents == ["general"]
        assert result.is_multi_intent is False

    def test_whitespace_only_yields_general_intent(self):
        raw = "    \t  \n  \r\n  "
        result = normalize_query(raw)
        assert result.original == raw
        assert result.normalized == ""
        assert result.detected_order_ids == []
        assert result.detected_intents == ["general"]
        assert result.is_multi_intent is False

    def test_unmatched_query_yields_general_intent(self):
        query = "Hello, I just wanted to say thank you for your service!"
        result = normalize_query(query)
        assert result.detected_intents == ["general"]
        assert result.detected_order_ids == []
        assert result.is_multi_intent is False

    def test_special_characters_only_yields_general_intent(self):
        query = "??? !!! @#$% ^&*() _+"
        result = normalize_query(query)
        assert result.detected_intents == ["general"]
        assert result.detected_order_ids == []
        assert result.is_multi_intent is False


# ---------------------------------------------------------------------------
# 2. Return Policy Queries -> 'return_policy' intent
# ---------------------------------------------------------------------------


class TestReturnPolicyIntent:
    """Test detection of return policy intent across various phrasing."""

    def test_return_policy_direct_query(self):
        result = normalize_query("What is your return policy?")
        assert "return_policy" in result.detected_intents
        assert result.detected_intents == ["return_policy"]
        assert result.is_multi_intent is False

    def test_refund_query(self):
        result = normalize_query("I would like to request a refund")
        assert "return_policy" in result.detected_intents
        assert result.is_multi_intent is False

    def test_exchange_query(self):
        result = normalize_query("How do I exchange my item for a different size?")
        assert "return_policy" in result.detected_intents

    def test_send_back_query(self):
        result = normalize_query("How can I send back a package?")
        assert "return_policy" in result.detected_intents

    def test_return_window_query(self):
        result = normalize_query("What is the return window?")
        assert "return_policy" in result.detected_intents

    def test_money_back_query(self):
        result = normalize_query("Do you offer a money back guarantee?")
        assert "return_policy" in result.detected_intents


# ---------------------------------------------------------------------------
# 3. Shipping Queries -> 'shipping' intent
# ---------------------------------------------------------------------------


class TestShippingIntent:
    """Test detection of shipping intent across various phrasing."""

    def test_international_shipping_query(self):
        result = normalize_query("Do you offer international shipping?")
        assert result.detected_intents == ["shipping"]
        assert result.is_multi_intent is False

    def test_canada_delivery_query(self):
        result = normalize_query("Do you ship to Canada?")
        assert "shipping" in result.detected_intents
        assert result.is_multi_intent is False

    def test_free_shipping_query(self):
        result = normalize_query("What is the minimum order for free shipping?")
        assert "shipping" in result.detected_intents

    def test_transit_time_query(self):
        result = normalize_query("What is the transit time for domestic delivery?")
        assert "shipping" in result.detected_intents


# ---------------------------------------------------------------------------
# 4. Order ID Queries -> extracts ORD-XXXX, 'order_status' intent
# ---------------------------------------------------------------------------


class TestOrderIdExtractionAndOrderStatus:
    """Test extracting ORD-XXXX and defaulting to order_status intent when no explicit intent."""

    def test_bare_order_id(self):
        result = normalize_query("ORD-1007")
        assert result.detected_order_ids == ["ORD-1007"]
        assert result.detected_intents == ["order_status"]
        assert result.is_multi_intent is False

    def test_order_id_in_neutral_query(self):
        result = normalize_query("Can you check ORD-1007 for me?")
        assert result.detected_order_ids == ["ORD-1007"]
        assert result.detected_intents == ["order_status"]
        assert result.is_multi_intent is False

    def test_order_id_with_explicit_order_status_keyword(self):
        result = normalize_query("Where is my order ORD-1007?")
        assert result.detected_order_ids == ["ORD-1007"]
        assert result.detected_intents == ["order_status"]
        assert result.is_multi_intent is False

    def test_order_id_with_tracking_keyword(self):
        result = normalize_query("Please track ORD-1002")
        assert result.detected_order_ids == ["ORD-1002"]
        assert result.detected_intents == ["order_status"]
        assert result.is_multi_intent is False

    def test_order_id_with_estimated_delivery_keyword(self):
        result = normalize_query("What is the estimated delivery for ORD-1003?")
        # Matches order_status keywords ('estimated delivery', 'delivery' under shipping)
        assert result.detected_order_ids == ["ORD-1003"]
        assert "order_status" in result.detected_intents

    def test_multiple_order_ids_extracted(self):
        result = normalize_query("Please check ORD-1001 and ORD-1002")
        assert result.detected_order_ids == ["ORD-1001", "ORD-1002"]
        assert result.detected_intents == ["order_status"]


# ---------------------------------------------------------------------------
# 5. Malformed Order IDs -> still extracts and normalizes
# ---------------------------------------------------------------------------


class TestMalformedOrderIdExtraction:
    """Test extracting and normalizing malformed order ID variants (ord 1007, ord-1007, ORD1007)."""

    def test_lowercase_with_dash(self):
        result = normalize_query("Check ord-1007 please")
        assert result.detected_order_ids == ["ORD-1007"]
        assert result.detected_intents == ["order_status"]

    def test_lowercase_with_space(self):
        result = normalize_query("What happened to ord 1007?")
        assert result.detected_order_ids == ["ORD-1007"]
        assert result.detected_intents == ["order_status"]

    def test_uppercase_with_space(self):
        result = normalize_query("Lookup ORD 1007")
        assert result.detected_order_ids == ["ORD-1007"]
        assert result.detected_intents == ["order_status"]

    def test_no_separator(self):
        result = normalize_query("Can you find ORD1007?")
        assert result.detected_order_ids == ["ORD-1007"]
        assert result.detected_intents == ["order_status"]

    def test_lowercase_no_separator(self):
        result = normalize_query("Status of ord1007")
        assert result.detected_order_ids == ["ORD-1007"]
        assert result.detected_intents == ["order_status"]

    def test_helper_normalize_order_id_format(self):
        assert _normalize_order_id_format("ord 1007") == "ORD-1007"
        assert _normalize_order_id_format("ord-1007") == "ORD-1007"
        assert _normalize_order_id_format("ORD1007") == "ORD-1007"
        assert _normalize_order_id_format("ord1007") == "ORD-1007"
        assert _normalize_order_id_format("ORD-1007") == "ORD-1007"
        assert _normalize_order_id_format("  ORD-1007  ") == "ORD-1007"

    def test_helper_normalize_order_id_format_non_matching(self):
        # Strings that do not match the ORD pattern pass through stripped and uppercased
        assert _normalize_order_id_format("INVOICE-99") == "INVOICE-99"
        assert _normalize_order_id_format("  custom_id  ") == "CUSTOM_ID"


# ---------------------------------------------------------------------------
# 6. Order ID + Return Keywords -> Multi-Intent Detection
# ---------------------------------------------------------------------------


class TestOrderIdWithReturnKeywords:
    """Test queries that combine order status / tracking with return keywords."""

    def test_order_id_with_track_and_return(self):
        result = normalize_query("Track ORD-1007 and I want to return it")
        assert result.detected_order_ids == ["ORD-1007"]
        assert "order_status" in result.detected_intents
        assert "return_policy" in result.detected_intents
        assert result.is_multi_intent is True

    def test_order_id_with_where_is_my_order_and_refund(self):
        result = normalize_query("Where is my order ORD-1007? I want a refund.")
        assert result.detected_order_ids == ["ORD-1007"]
        assert "order_status" in result.detected_intents
        assert "return_policy" in result.detected_intents
        assert result.is_multi_intent is True

    def test_order_id_with_order_status_and_exchange(self):
        result = normalize_query("Check order status for ORD-1008 and can I exchange it?")
        assert result.detected_order_ids == ["ORD-1008"]
        assert "order_status" in result.detected_intents
        assert "return_policy" in result.detected_intents
        assert result.is_multi_intent is True

    def test_order_id_with_only_return_keyword(self):
        # When return keyword is present without order status keywords, return_policy is detected
        result = normalize_query("I want to return ORD-1007")
        assert result.detected_order_ids == ["ORD-1007"]
        assert "return_policy" in result.detected_intents


# ---------------------------------------------------------------------------
# 7. Product Care Queries -> 'product_info' intent
# ---------------------------------------------------------------------------


class TestProductInfoIntent:
    """Test detection of product info and care instructions."""

    def test_product_care_query(self):
        result = normalize_query("How should I care for my product?")
        assert result.detected_intents == ["product_info"]
        assert result.is_multi_intent is False

    def test_dishwasher_safe_query(self):
        result = normalize_query("Is the Breeze Tumbler dishwasher safe?")
        assert "product_info" in result.detected_intents
        assert result.is_multi_intent is False

    def test_product_material_query(self):
        result = normalize_query("What material is the Ridge Daypack made of?")
        assert "product_info" in result.detected_intents
        assert result.is_multi_intent is False

    def test_cleaning_instructions_query(self):
        result = normalize_query("What are the cleaning instructions for the TrailPouch?")
        assert "product_info" in result.detected_intents
        assert result.is_multi_intent is False

    def test_summit_product_query(self):
        result = normalize_query("Tell me about the Summit mug")
        assert "product_info" in result.detected_intents


# ---------------------------------------------------------------------------
# 8. Warranty Queries -> 'warranty' intent
# ---------------------------------------------------------------------------


class TestWarrantyIntent:
    """Test detection of warranty coverage and claims."""

    def test_warranty_coverage_query(self):
        result = normalize_query("What is covered under the lifetime warranty coverage?")
        assert result.detected_intents == ["warranty"]
        assert result.is_multi_intent is False

    def test_warranty_claim_query(self):
        result = normalize_query("How do I file a warranty claim?")
        assert result.detected_intents == ["warranty"]
        assert result.is_multi_intent is False

    def test_manufacturing_defect_query(self):
        result = normalize_query("My zipper has a manufacturing defect")
        assert "warranty" in result.detected_intents

    def test_repair_query(self):
        result = normalize_query("Can I get a repair for my backpack?")
        assert "warranty" in result.detected_intents


# ---------------------------------------------------------------------------
# 9. Cancellation Queries -> 'cancellation' intent
# ---------------------------------------------------------------------------


class TestCancellationIntent:
    """Test detection of cancellation and modification intents."""

    def test_cancel_order_query(self):
        result = normalize_query("I need to cancel order immediately")
        assert result.detected_intents == ["cancellation"]
        assert result.is_multi_intent is False

    def test_cancellation_policy_query(self):
        result = normalize_query("What is your cancellation policy?")
        assert result.detected_intents == ["cancellation"]
        assert result.is_multi_intent is False

    def test_stop_order_query(self):
        result = normalize_query("Please stop order before it ships")
        assert "cancellation" in result.detected_intents

    def test_modify_order_query(self):
        result = normalize_query("Can I modify order items?")
        assert "cancellation" in result.detected_intents

    def test_address_change_query(self):
        result = normalize_query("I need an address change for my order")
        assert "cancellation" in result.detected_intents


# ---------------------------------------------------------------------------
# 10. TrailPlus Membership Queries -> 'membership' intent
# ---------------------------------------------------------------------------


class TestMembershipIntent:
    """Test detection of TrailPlus membership queries."""

    def test_trailplus_membership_query(self):
        result = normalize_query("What are the TrailPlus membership benefits?")
        assert result.detected_intents == ["membership"]
        assert result.is_multi_intent is False

    def test_trail_plus_spaced_query(self):
        result = normalize_query("How do I join Trail Plus?")
        assert result.detected_intents == ["membership"]
        assert result.is_multi_intent is False

    def test_member_discount_query(self):
        result = normalize_query("Do members get an extra discount?")
        assert "membership" in result.detected_intents

    def test_membership_renewal_query(self):
        result = normalize_query("When does my membership renew?")
        assert "membership" in result.detected_intents


# ---------------------------------------------------------------------------
# 11. Multiple Intents Detected -> is_multi_intent=True
# ---------------------------------------------------------------------------


class TestMultiIntentFlag:
    """Test that multiple detected intents properly set is_multi_intent to True."""

    def test_return_policy_and_shipping(self):
        result = normalize_query("What is your return policy and how long is shipping?")
        assert "return_policy" in result.detected_intents
        assert "shipping" in result.detected_intents
        assert result.is_multi_intent is True
        assert len(result.detected_intents) >= 2

    def test_cancellation_and_membership(self):
        result = normalize_query("Can I cancel my TrailPlus membership?")
        assert "cancellation" in result.detected_intents
        assert "membership" in result.detected_intents
        assert result.is_multi_intent is True

    def test_warranty_and_return(self):
        result = normalize_query("Is this covered under warranty or should I return it for a refund?")
        assert "warranty" in result.detected_intents
        assert "return_policy" in result.detected_intents
        assert result.is_multi_intent is True

    def test_three_intents_detected(self):
        result = normalize_query("Can I cancel my membership and get a refund on shipping?")
        assert "cancellation" in result.detected_intents
        assert "membership" in result.detected_intents
        assert "return_policy" in result.detected_intents
        assert "shipping" in result.detected_intents
        assert result.is_multi_intent is True
        assert len(result.detected_intents) >= 3

    def test_single_intent_is_not_multi_intent(self):
        result = normalize_query("What is your return policy?")
        assert result.is_multi_intent is False
        assert len(result.detected_intents) == 1


# ---------------------------------------------------------------------------
# 12. Whitespace Normalization
# ---------------------------------------------------------------------------


class TestWhitespaceNormalization:
    """Test whitespace normalization and preserving original raw query."""

    def test_collapses_multiple_spaces(self):
        query = "Where   is   my   order   ORD-1007?"
        result = normalize_query(query)
        assert result.normalized == "Where is my order ORD-1007?"
        assert result.original == query

    def test_strips_leading_and_trailing_whitespace(self):
        query = "   What is your return policy?   "
        result = normalize_query(query)
        assert result.normalized == "What is your return policy?"
        assert result.original == query

    def test_handles_tabs_and_newlines(self):
        query = "\tTrack \n\n ORD-1007 \r\n status \t"
        result = normalize_query(query)
        assert result.normalized == "Track ORD-1007 status"
        assert result.original == query

    def test_original_always_preserved_verbatim(self):
        query = "  ORD-1001   \t\n  Where is my order?  "
        result = normalize_query(query)
        assert result.original == query
        assert result.normalized == "ORD-1001 Where is my order?"


# ---------------------------------------------------------------------------
# Additional Intents: price_adjustment, damaged_item, gift_card, final_sale
# ---------------------------------------------------------------------------


class TestAdditionalIntents:
    """Test the remaining intent categories configured in _INTENT_PATTERNS."""

    def test_price_adjustment_intent(self):
        result = normalize_query("Can I get a price adjustment? The price went down.")
        assert "price_adjustment" in result.detected_intents

    def test_damaged_item_intent(self):
        result = normalize_query("I received a damaged and defective product")
        assert "damaged_item" in result.detected_intents

    def test_gift_card_intent(self):
        result = normalize_query("Can I pay with a gift card or store credit?")
        assert "gift_card" in result.detected_intents

    def test_final_sale_intent(self):
        result = normalize_query("Are clearance items considered final sale?")
        assert "final_sale" in result.detected_intents
