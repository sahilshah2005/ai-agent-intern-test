"""
test_evidence.py — unit tests for the evidence-level citation architecture.

Tests cover:
  1. build_evidence_list assigning sequential E1, E2, ... IDs.
  2. is_citable=True for active + official + customer facing documents.
  3. is_citable=False for internal, draft, superseded, or non-authoritative documents.
  4. build_evidence_context producing formatted strings with evidence IDs and metadata.
  5. build_evidence_context surfacing conflict warnings when conflicts are detected.
  6. validate_citations recognizing valid [E1] references.
  7. validate_citations flagging invalid [E99] references.
  8. validate_citations flagging non-citable references (e.g. internal documents).
  9. transform_evidence_ids_to_readable replacing [E1] with human-readable citations.
 10. Edge cases including empty evidence lists, missing optional fields, and malformed tags.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evidence import (
    CitationResult,
    Evidence,
    build_evidence_context,
    build_evidence_list,
    transform_evidence_ids_to_readable,
    validate_citations,
)


# ---------------------------------------------------------------------------
# Fixtures & Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunk_active_official() -> dict[str, Any]:
    """A valid, citable customer-facing official policy chunk."""
    return {
        "filename": "01-returns-policy.md",
        "heading": "Return Window",
        "title": "Return & Refund Policy",
        "status": "active",
        "policy_authority": "official",
        "audience": "customer",
        "content": "Items can be returned within 30 days of receipt.",
        "retrieval_method": "hybrid",
        "combined_score": 0.95,
        "precedence_score": 1.0,
    }


@pytest.fixture
def sample_chunk_internal() -> dict[str, Any]:
    """An internal-only chunk that should not be citable."""
    return {
        "filename": "14-internal-procedures.md",
        "heading": "Escalations",
        "title": "Internal Support Runbook",
        "status": "active",
        "policy_authority": "official",
        "audience": "internal",
        "content": "Escalate high risk refunds directly to tier 2.",
        "retrieval_method": "vector",
        "combined_score": 0.80,
        "precedence_score": 0.5,
    }


@pytest.fixture
def sample_chunk_draft() -> dict[str, Any]:
    """A draft chunk that should not be citable."""
    return {
        "filename": "05-warranty-draft.md",
        "heading": "Coverage Limits",
        "title": "Proposed Warranty Overhaul",
        "status": "draft",
        "policy_authority": "official",
        "audience": "customer",
        "content": "Proposed warranty extension to 2 years.",
        "retrieval_method": "lexical",
        "combined_score": 0.70,
        "precedence_score": 0.3,
    }


@pytest.fixture
def sample_chunk_superseded() -> dict[str, Any]:
    """A superseded chunk that should not be citable."""
    return {
        "filename": "01-returns-old.md",
        "heading": "Return Window (Old)",
        "title": "Return Policy 2021",
        "status": "superseded",
        "policy_authority": "official",
        "audience": "customer",
        "content": "Return window was 14 days under the old rules.",
        "retrieval_method": "hybrid",
        "combined_score": 0.65,
        "precedence_score": 0.2,
    }


@pytest.fixture
def sample_retrieved_chunks(
    sample_chunk_active_official: dict[str, Any],
    sample_chunk_internal: dict[str, Any],
    sample_chunk_draft: dict[str, Any],
    sample_chunk_superseded: dict[str, Any],
) -> list[dict[str, Any]]:
    """A mixed list of retrieved chunks with various statuses and audiences."""
    return [
        sample_chunk_active_official,
        sample_chunk_internal,
        sample_chunk_draft,
        sample_chunk_superseded,
    ]


@pytest.fixture
def sample_evidence_list(sample_retrieved_chunks: list[dict[str, Any]]) -> list[Evidence]:
    """An instantiated list of Evidence objects built from sample chunks."""
    return build_evidence_list(sample_retrieved_chunks)


# ---------------------------------------------------------------------------
# 1. Tests for build_evidence_list & sequential IDs
# ---------------------------------------------------------------------------


class TestBuildEvidenceList:
    """Tests for build_evidence_list assignment and mapping."""

    def test_assigns_sequential_evidence_ids(
        self, sample_retrieved_chunks: list[dict[str, Any]]
    ) -> None:
        """Requirement 1: build_evidence_list assigns sequential E1, E2, ... IDs."""
        evidence_list = build_evidence_list(sample_retrieved_chunks)

        assert len(evidence_list) == len(sample_retrieved_chunks)
        expected_ids = [f"E{i}" for i in range(1, len(sample_retrieved_chunks) + 1)]
        actual_ids = [ev.evidence_id for ev in evidence_list]
        assert actual_ids == expected_ids

    def test_preserves_chunk_fields_and_scores(
        self, sample_chunk_active_official: dict[str, Any]
    ) -> None:
        """Verify all fields from retrieved chunks map correctly to the Evidence dataclass."""
        evidence_list = build_evidence_list([sample_chunk_active_official])
        ev = evidence_list[0]

        assert ev.evidence_id == "E1"
        assert ev.filename == "01-returns-policy.md"
        assert ev.heading == "Return Window"
        assert ev.title == "Return & Refund Policy"
        assert ev.status == "active"
        assert ev.policy_authority == "official"
        assert ev.audience == "customer"
        assert ev.content == "Items can be returned within 30 days of receipt."
        assert ev.retrieval_method == "hybrid"
        assert ev.combined_score == 0.95
        assert ev.precedence_score == 1.0

    def test_default_field_fallbacks_when_optional_keys_missing(self) -> None:
        """Verify missing optional keys receive proper default values."""
        minimal_chunk = {
            "filename": "minimal-doc.md",
            "heading": "Minimal Section",
        }
        evidence_list = build_evidence_list([minimal_chunk])
        assert len(evidence_list) == 1
        ev = evidence_list[0]

        assert ev.evidence_id == "E1"
        assert ev.filename == "minimal-doc.md"
        assert ev.heading == "Minimal Section"
        assert ev.title == "minimal-doc.md"  # falls back to filename
        assert ev.status == "unknown"
        assert ev.policy_authority == "unknown"
        assert ev.audience == "customer"
        assert ev.content == ""
        assert ev.retrieval_method == "unknown"
        assert ev.combined_score == 0.0
        assert ev.precedence_score == 0.0
        assert ev.is_citable is False

    def test_empty_retrieved_chunks_returns_empty_list(self) -> None:
        """Empty input list returns an empty list of Evidence."""
        result = build_evidence_list([])
        assert result == []


# ---------------------------------------------------------------------------
# 2 & 3. Tests for is_citable logic
# ---------------------------------------------------------------------------


class TestIsCitable:
    """Tests for citable determination: status=='active' AND policy_authority=='official' AND audience!='internal'."""

    def test_is_citable_true_for_active_official_customer(
        self, sample_chunk_active_official: dict[str, Any]
    ) -> None:
        """Requirement 2: is_citable=True for active + official + customer."""
        evidence_list = build_evidence_list([sample_chunk_active_official])
        assert evidence_list[0].is_citable is True

    def test_is_citable_true_when_audience_not_specified(self) -> None:
        """Chunk without explicit audience defaults to 'customer' and is citable if active + official."""
        chunk = {
            "filename": "official-notice.md",
            "heading": "General Notice",
            "status": "active",
            "policy_authority": "official",
        }
        evidence_list = build_evidence_list([chunk])
        assert evidence_list[0].audience == "customer"
        assert evidence_list[0].is_citable is True

    def test_is_citable_false_for_internal_doc(
        self, sample_chunk_internal: dict[str, Any]
    ) -> None:
        """Requirement 3: is_citable=False for internal audience."""
        evidence_list = build_evidence_list([sample_chunk_internal])
        assert evidence_list[0].is_citable is False

    def test_is_citable_false_for_draft_status(
        self, sample_chunk_draft: dict[str, Any]
    ) -> None:
        """Requirement 3: is_citable=False for draft status."""
        evidence_list = build_evidence_list([sample_chunk_draft])
        assert evidence_list[0].is_citable is False

    def test_is_citable_false_for_superseded_status(
        self, sample_chunk_superseded: dict[str, Any]
    ) -> None:
        """Requirement 3: is_citable=False for superseded status."""
        evidence_list = build_evidence_list([sample_chunk_superseded])
        assert evidence_list[0].is_citable is False

    @pytest.mark.parametrize("status", ["draft", "superseded", "deprecated", "unknown", "archived"])
    def test_is_citable_false_for_non_active_statuses(self, status: str) -> None:
        """Any status other than 'active' renders the chunk non-citable."""
        chunk = {
            "filename": f"doc-{status}.md",
            "heading": "Section",
            "status": status,
            "policy_authority": "official",
            "audience": "customer",
        }
        evidence_list = build_evidence_list([chunk])
        assert evidence_list[0].is_citable is False

    @pytest.mark.parametrize("authority", ["guideline", "informational", "unknown", "draft", "unofficial"])
    def test_is_citable_false_for_non_official_authority(self, authority: str) -> None:
        """Any policy_authority other than 'official' renders the chunk non-citable."""
        chunk = {
            "filename": "guide.md",
            "heading": "Guidance",
            "status": "active",
            "policy_authority": authority,
            "audience": "customer",
        }
        evidence_list = build_evidence_list([chunk])
        assert evidence_list[0].is_citable is False


# ---------------------------------------------------------------------------
# 4 & 5. Tests for build_evidence_context
# ---------------------------------------------------------------------------


class TestBuildEvidenceContext:
    """Tests for build_evidence_context formatting and conflict handling."""

    def test_produces_string_with_evidence_ids(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Requirement 4: build_evidence_context produces string with evidence IDs."""
        context = build_evidence_context(sample_evidence_list, conflicts=[])

        assert isinstance(context, str)
        assert "[E1] Source: 01-returns-policy.md — Return Window" in context
        assert "[E2] Source: 14-internal-procedures.md — Escalations" in context
        assert "[E3] Source: 05-warranty-draft.md — Coverage Limits" in context
        assert "[E4] Source: 01-returns-old.md — Return Window (Old)" in context

    def test_context_includes_metadata_and_content(
        self, sample_chunk_active_official: dict[str, Any]
    ) -> None:
        """Context includes Status, Authority, Audience lines and content text."""
        evidence_list = build_evidence_list([sample_chunk_active_official])
        context = build_evidence_context(evidence_list, conflicts=[])

        assert "Status: active | Authority: official | Audience: customer" in context
        assert "Items can be returned within 30 days of receipt." in context

    def test_context_with_conflicts_includes_warning(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Requirement 5: build_evidence_context with conflicts includes warning."""
        conflicts = [
            (
                {"filename": "01-policy-a.md"},
                {"filename": "02-policy-b.md"},
                "Policy A states 30 days while Policy B states 45 days.",
            )
        ]
        context = build_evidence_context(sample_evidence_list, conflicts=conflicts)

        assert "⚠️  CONFLICT DETECTED — Two active official sources contradict each other on this topic." in context
        assert "You MUST surface this conflict in your response and recommend human confirmation." in context
        assert "• Policy A states 30 days while Policy B states 45 days." in context

    def test_context_without_conflicts_omits_warning(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """When conflicts list is empty, no conflict warning is injected."""
        context = build_evidence_context(sample_evidence_list, conflicts=[])
        assert "CONFLICT DETECTED" not in context
        assert "⚠️" not in context

    def test_context_formats_labels_for_non_active_authority_audience(self) -> None:
        """Context formats warning brackets like [DRAFT], [NON-AUTHORITATIVE], [INTERNAL...]."""
        chunk = {
            "filename": "internal-draft-note.md",
            "heading": "Draft Note",
            "status": "draft",
            "policy_authority": "guideline",
            "audience": "internal",
            "content": "Secret internal draft details.",
        }
        evidence_list = build_evidence_list([chunk])
        context = build_evidence_context(evidence_list, conflicts=[])

        assert "[DRAFT | NON-AUTHORITATIVE | INTERNAL — do not cite as customer-facing authority]" in context

    def test_empty_evidence_list_returns_empty_string(self) -> None:
        """Requirement 10: Empty evidence list returns empty string even if conflicts exist."""
        assert build_evidence_context([], conflicts=[]) == ""
        conflicts = [({}, {}, "Sample conflict description")]
        assert build_evidence_context([], conflicts=conflicts) == ""


# ---------------------------------------------------------------------------
# 6, 7 & 8. Tests for validate_citations
# ---------------------------------------------------------------------------


class TestValidateCitations:
    """Tests for validate_citations covering valid, invalid, non-citable, and filename refs."""

    def test_valid_e_citation(self, sample_evidence_list: list[Evidence]) -> None:
        """Requirement 6: validate_citations: valid [E1] reference."""
        response_text = "Customers can return purchases within 30 days [E1]."
        result = validate_citations(response_text, sample_evidence_list)

        assert isinstance(result, CitationResult)
        assert result.is_valid is True
        assert result.valid_citations == ["E1"]
        assert result.invalid_citations == []
        assert result.non_citable_citations == []
        assert len(result.cited_sources) == 1
        assert result.cited_sources[0] == {
            "filename": "01-returns-policy.md",
            "title": "Return & Refund Policy",
            "heading": "Return Window",
        }

    def test_multiple_valid_citations(self, sample_chunk_active_official: dict[str, Any]) -> None:
        """Multiple valid citable references are captured in valid_citations."""
        chunk2 = dict(sample_chunk_active_official)
        chunk2["filename"] = "02-shipping-policy.md"
        chunk2["heading"] = "Express Shipping"
        chunk2["title"] = "Shipping Policy"

        evidence_list = build_evidence_list([sample_chunk_active_official, chunk2])
        response_text = "As stated in [E1] and confirmed in [E2], shipping is prompt."
        result = validate_citations(response_text, evidence_list)

        assert result.is_valid is True
        assert set(result.valid_citations) == {"E1", "E2"}
        assert result.invalid_citations == []
        assert result.non_citable_citations == []
        assert len(result.cited_sources) == 2

    def test_invalid_e_citation_reference(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Requirement 7: validate_citations: invalid [E99] reference."""
        response_text = "According to obsolete rule [E99], returns are allowed."
        result = validate_citations(response_text, sample_evidence_list)

        assert result.is_valid is False
        assert "E99" in result.invalid_citations
        assert result.valid_citations == []
        assert result.non_citable_citations == []
        assert result.cited_sources == []

    def test_non_citable_internal_doc_reference(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Requirement 8: validate_citations: non-citable reference (internal doc cited)."""
        # In sample_evidence_list, E2 is internal
        response_text = "Internal procedure details are described in [E2]."
        result = validate_citations(response_text, sample_evidence_list)

        assert result.is_valid is False
        assert "E2" in result.non_citable_citations
        assert result.valid_citations == []
        assert result.invalid_citations == []

    def test_non_citable_draft_and_superseded_references(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """validate_citations flags draft [E3] and superseded [E4] as non-citable."""
        response_text = "Proposed warranty [E3] vs old return policy [E4]."
        result = validate_citations(response_text, sample_evidence_list)

        assert result.is_valid is False
        assert "E3" in result.non_citable_citations
        assert "E4" in result.non_citable_citations
        assert result.valid_citations == []

    def test_mixed_valid_and_invalid_and_non_citable(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Mixed citations: valid [E1], non-citable [E2], and invalid [E99]."""
        response_text = "See [E1] for policy, [E2] for runbook, and [E99] for ghost rule."
        result = validate_citations(response_text, sample_evidence_list)

        assert result.is_valid is False
        assert "E1" in result.valid_citations
        assert "E2" in result.non_citable_citations
        assert "E99" in result.invalid_citations

    def test_no_citations_in_response(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Response with no citations defaults to valid with empty lists."""
        response_text = "Hello! I am happy to assist you today with any questions."
        result = validate_citations(response_text, sample_evidence_list)

        assert result.is_valid is True
        assert result.valid_citations == []
        assert result.invalid_citations == []
        assert result.non_citable_citations == []
        assert result.cited_sources == []

    def test_filename_citation_citable_backward_compat(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Backward compatibility: detects citable filename format e.g. 01-returns-policy.md."""
        response_text = "Refer to 01-returns-policy.md for return guidelines."
        result = validate_citations(response_text, sample_evidence_list)

        assert result.is_valid is True
        assert len(result.cited_sources) == 1
        assert result.cited_sources[0]["filename"] == "01-returns-policy.md"

    def test_filename_citation_non_citable_internal(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Backward compatibility: detects internal filename reference and marks invalid."""
        response_text = "Consult 14-internal-procedures.md for details."
        result = validate_citations(response_text, sample_evidence_list)

        assert result.is_valid is False
        assert "file:14-internal-procedures.md" in result.non_citable_citations


# ---------------------------------------------------------------------------
# 9. Tests for transform_evidence_ids_to_readable
# ---------------------------------------------------------------------------


class TestTransformEvidenceIdsToReadable:
    """Tests for transform_evidence_ids_to_readable replacement behavior."""

    def test_replaces_single_valid_evidence_id(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Requirement 9: transform_evidence_ids_to_readable replaces [E1]."""
        response_text = "You can return items within 30 days [E1]."
        transformed = transform_evidence_ids_to_readable(
            response_text, sample_evidence_list
        )

        expected = (
            "You can return items within 30 days "
            "(Source: 01-returns-policy.md — Return Window)."
        )
        assert transformed == expected

    def test_replaces_multiple_valid_evidence_ids(
        self, sample_chunk_active_official: dict[str, Any]
    ) -> None:
        """Multiple citable evidence IDs are each replaced by human-readable citations."""
        chunk2 = dict(sample_chunk_active_official)
        chunk2["filename"] = "02-orders.md"
        chunk2["heading"] = "Order Tracking"

        evidence_list = build_evidence_list([sample_chunk_active_official, chunk2])
        response_text = "Track your shipment [E2] or return it [E1]."
        transformed = transform_evidence_ids_to_readable(response_text, evidence_list)

        assert "(Source: 02-orders.md — Order Tracking)" in transformed
        assert "(Source: 01-returns-policy.md — Return Window)" in transformed
        assert "[E1]" not in transformed
        assert "[E2]" not in transformed

    def test_leaves_non_citable_evidence_id_untouched(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Non-citable evidence IDs (like internal E2 or draft E3) are NOT transformed."""
        response_text = "Internal note [E2] and draft [E3]."
        transformed = transform_evidence_ids_to_readable(
            response_text, sample_evidence_list
        )

        assert transformed == response_text
        assert "[E2]" in transformed
        assert "[E3]" in transformed

    def test_leaves_non_existent_evidence_id_untouched(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Evidence IDs that do not exist in the list are left as-is."""
        response_text = "See section [E99] for details."
        transformed = transform_evidence_ids_to_readable(
            response_text, sample_evidence_list
        )
        assert transformed == "See section [E99] for details."

    def test_text_without_evidence_ids_remains_identical(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Plain response text without any [E...] tags is returned unchanged."""
        response_text = "There are no citations in this answer."
        assert transform_evidence_ids_to_readable(response_text, sample_evidence_list) == response_text


# ---------------------------------------------------------------------------
# 10. Tests for Edge Cases & Boundary Conditions
# ---------------------------------------------------------------------------


class TestEvidenceEdgeCases:
    """Requirement 10: Empty evidence list edge cases and boundary conditions."""

    def test_empty_evidence_list_build(self) -> None:
        """build_evidence_list handles empty inputs gracefully."""
        assert build_evidence_list([]) == []

    def test_empty_evidence_list_context(self) -> None:
        """build_evidence_context returns empty string when evidence list is empty."""
        assert build_evidence_context([], conflicts=[]) == ""
        assert build_evidence_context([], conflicts=[({}, {}, "conflict")]) == ""

    def test_empty_evidence_list_validate_no_citations(self) -> None:
        """validate_citations with empty evidence and no citations in response is valid."""
        result = validate_citations("Normal response without citations.", [])
        assert result.is_valid is True
        assert result.valid_citations == []
        assert result.invalid_citations == []
        assert result.non_citable_citations == []
        assert result.cited_sources == []

    def test_empty_evidence_list_validate_with_citations(self) -> None:
        """validate_citations with citations in response but empty evidence flags all as invalid."""
        result = validate_citations("Response citing [E1] and [E2].", [])
        assert result.is_valid is False
        assert set(result.invalid_citations) == {"E1", "E2"}
        assert result.valid_citations == []
        assert result.non_citable_citations == []

    def test_empty_evidence_list_transform(self) -> None:
        """transform_evidence_ids_to_readable returns original string when evidence list is empty."""
        text = "Response referencing [E1]."
        assert transform_evidence_ids_to_readable(text, []) == text
        assert transform_evidence_ids_to_readable("", []) == ""

    def test_malformed_bracketed_patterns_ignored(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Tags like [E], [Eabc], [1], [E-1] are not parsed as evidence IDs."""
        text = "This [E] and [Eabc] and [1] and [E-1] are not evidence tags."
        result = validate_citations(text, sample_evidence_list)
        assert result.is_valid is True
        assert result.valid_citations == []
        assert result.invalid_citations == []

        transformed = transform_evidence_ids_to_readable(text, sample_evidence_list)
        assert transformed == text

    def test_duplicate_evidence_id_in_response(
        self, sample_evidence_list: list[Evidence]
    ) -> None:
        """Duplicate citations [E1] ... [E1] in response result in single entry in valid_citations."""
        text = "First mentioned in [E1], and later reiterated in [E1]."
        result = validate_citations(text, sample_evidence_list)
        assert result.is_valid is True
        assert result.valid_citations == ["E1"]
        assert len(result.cited_sources) == 1

        transformed = transform_evidence_ids_to_readable(text, sample_evidence_list)
        expected_citation = "(Source: 01-returns-policy.md — Return Window)"
        assert transformed.count(expected_citation) == 2
        assert "[E1]" not in transformed
