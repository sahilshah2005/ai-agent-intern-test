"""
test_confidence.py — unit tests for evidence confidence assessment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from confidence import ConfidenceAssessment, assess_confidence
from config import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD
from evidence import Evidence


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


def make_evidence(
    evidence_id: str = "E1",
    filename: str = "01-returns-policy.md",
    heading: str = "Return Window",
    title: str = "Return Policy",
    status: str = "active",
    policy_authority: str = "official",
    audience: str = "customer",
    content: str = "Items can be returned within 30 days of purchase.",
    is_citable: bool = True,
    retrieval_method: str = "hybrid",
    combined_score: float = 0.85,
    precedence_score: float = 1.0,
) -> Evidence:
    """Helper to create an Evidence instance with sensible defaults."""
    return Evidence(
        evidence_id=evidence_id,
        filename=filename,
        heading=heading,
        title=title,
        status=status,
        policy_authority=policy_authority,
        audience=audience,
        content=content,
        is_citable=is_citable,
        retrieval_method=retrieval_method,
        combined_score=combined_score,
        precedence_score=precedence_score,
    )


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------


class TestConfidenceEmptyInputs:
    """Tests for cases where no evidence or tool results are provided."""

    def test_no_evidence_no_tool_yields_low_and_abstains(self):
        """Empty evidence and no tool result should abstain with low confidence."""
        assessment = assess_confidence(
            evidence_list=[],
            has_tool_result=False,
            tool_error=False,
        )
        assert assessment.level == "low"
        assert assessment.score == 0.0
        assert assessment.should_abstain is True
        assert assessment.should_handoff is True
        assert "No evidence retrieved" in assessment.reasoning

    def test_default_arguments_abstains(self):
        """Calling assess_confidence with only empty list defaults to low/abstain."""
        assessment = assess_confidence([])
        assert assessment.level == "low"
        assert assessment.score == 0.0
        assert assessment.should_abstain is True
        assert assessment.should_handoff is True


class TestConfidenceToolScenarios:
    """Tests for tool results and tool error handling."""

    def test_tool_error_with_tool_result_flag(self):
        """Tool error should produce low confidence, trigger handoff, but not abstain."""
        assessment = assess_confidence(
            evidence_list=[],
            has_tool_result=True,
            tool_error=True,
        )
        assert assessment.level == "low"
        assert assessment.score == 0.1
        assert assessment.should_abstain is False
        assert assessment.should_handoff is True
        assert "Order tool returned an error" in assessment.reasoning

    def test_tool_error_with_evidence_present(self):
        """Tool error takes precedence over evidence, requiring handoff."""
        ev = make_evidence(combined_score=0.9, is_citable=True)
        assessment = assess_confidence(
            evidence_list=[ev],
            tool_error=True,
        )
        assert assessment.level == "low"
        assert assessment.score == 0.1
        assert assessment.should_abstain is False
        assert assessment.should_handoff is True
        assert "Order tool returned an error" in assessment.reasoning

    def test_tool_result_only_yields_high_confidence(self):
        """Successful tool result with no KB evidence gives high confidence."""
        assessment = assess_confidence(
            evidence_list=[],
            has_tool_result=True,
            tool_error=False,
        )
        assert assessment.level == "high"
        assert assessment.score == 0.9
        assert assessment.should_abstain is False
        assert assessment.should_handoff is False
        assert "Tool returned valid result" in assessment.reasoning


class TestConfidenceConflicts:
    """Tests for conflict detection behavior."""

    def test_has_conflicts_yields_medium_confidence_and_handoff(self):
        """Detected conflicts should downgrade to medium confidence and trigger handoff."""
        ev1 = make_evidence("E1", combined_score=0.95, is_citable=True)
        ev2 = make_evidence("E2", combined_score=0.90, is_citable=True)

        assessment = assess_confidence(
            evidence_list=[ev1, ev2],
            has_conflicts=True,
        )
        assert assessment.level == "medium"
        assert assessment.score == 0.5
        assert assessment.should_handoff is True
        assert assessment.should_abstain is False
        assert "Conflict detected" in assessment.reasoning
        assert "among 2 citable sources" in assessment.reasoning

    def test_conflict_reasoning_includes_citable_count(self):
        """Conflict reasoning accurately reports the number of citable sources involved."""
        ev1 = make_evidence("E1", is_citable=True)
        ev2 = make_evidence("E2", is_citable=False)

        assessment = assess_confidence(
            evidence_list=[ev1, ev2],
            has_conflicts=True,
        )
        assert "among 1 citable sources" in assessment.reasoning


class TestConfidenceCitableEvidence:
    """Tests for citable vs non-citable evidence handling."""

    def test_no_citable_evidence_yields_low_confidence_and_abstain(self):
        """When evidence exists but none is citable (e.g. internal/draft), abstain."""
        ev1 = make_evidence(
            evidence_id="E1",
            status="draft",
            policy_authority="none",
            is_citable=False,
            combined_score=0.85,
        )
        ev2 = make_evidence(
            evidence_id="E2",
            audience="internal",
            is_citable=False,
            combined_score=0.75,
        )

        assessment = assess_confidence([ev1, ev2])
        assert assessment.level == "low"
        assert assessment.score == 0.15
        assert assessment.should_abstain is True
        assert assessment.should_handoff is True
        assert "No citable" in assessment.reasoning
        assert "2 retrieved chunks" in assessment.reasoning

    def test_mixed_citable_and_uncitable_uses_highest_score(self):
        """Citable evidence allows assessment based on highest score among chunks."""
        ev1 = make_evidence(evidence_id="E1", is_citable=False, combined_score=0.95)
        ev2 = make_evidence(evidence_id="E2", is_citable=True, combined_score=0.80)

        assessment = assess_confidence([ev1, ev2])
        assert assessment.level == "high"
        assert assessment.score == 0.95
        assert assessment.should_abstain is False
        assert assessment.should_handoff is False


class TestConfidenceScoreThresholds:
    """Tests for score-based categorization: high, medium, and low."""

    def test_high_score_evidence_yields_high_confidence(self):
        """Evidence score above high threshold yields high confidence."""
        ev = make_evidence(combined_score=0.85, is_citable=True)
        assessment = assess_confidence([ev])

        assert assessment.level == "high"
        assert assessment.score == 0.85
        assert assessment.should_abstain is False
        assert assessment.should_handoff is False
        assert f"≥ {CONFIDENCE_HIGH_THRESHOLD}" in assessment.reasoning

    def test_high_score_boundary_exact_threshold(self):
        """Evidence score exactly at CONFIDENCE_HIGH_THRESHOLD is high confidence."""
        ev = make_evidence(combined_score=CONFIDENCE_HIGH_THRESHOLD, is_citable=True)
        assessment = assess_confidence([ev])

        assert assessment.level == "high"
        assert assessment.score == CONFIDENCE_HIGH_THRESHOLD
        assert assessment.should_abstain is False
        assert assessment.should_handoff is False

    def test_high_score_capped_at_one(self):
        """Scores above 1.0 (from bonuses) are clamped to 1.0."""
        ev = make_evidence(combined_score=1.25, is_citable=True)
        assessment = assess_confidence([ev])

        assert assessment.level == "high"
        assert assessment.score == 1.0

    def test_medium_score_evidence_yields_medium_confidence(self):
        """Evidence score between medium and high threshold yields medium confidence."""
        score = (CONFIDENCE_HIGH_THRESHOLD + CONFIDENCE_MEDIUM_THRESHOLD) / 2.0
        ev = make_evidence(combined_score=score, is_citable=True)
        assessment = assess_confidence([ev])

        assert assessment.level == "medium"
        assert assessment.score == score
        assert assessment.should_abstain is False
        assert assessment.should_handoff is False
        assert "answer with caution" in assessment.reasoning

    def test_medium_score_boundary_exact_threshold(self):
        """Evidence score exactly at CONFIDENCE_MEDIUM_THRESHOLD is medium confidence."""
        ev = make_evidence(combined_score=CONFIDENCE_MEDIUM_THRESHOLD, is_citable=True)
        assessment = assess_confidence([ev])

        assert assessment.level == "medium"
        assert assessment.score == CONFIDENCE_MEDIUM_THRESHOLD
        assert assessment.should_abstain is False
        assert assessment.should_handoff is False

    def test_low_score_evidence_yields_low_confidence_and_abstain(self):
        """Evidence score below medium threshold yields low confidence and abstains."""
        score = CONFIDENCE_MEDIUM_THRESHOLD - 0.1
        ev = make_evidence(combined_score=score, is_citable=True)
        assessment = assess_confidence([ev])

        assert assessment.level == "low"
        assert assessment.score == score
        assert assessment.should_abstain is True
        assert assessment.should_handoff is True
        assert f"< {CONFIDENCE_MEDIUM_THRESHOLD}" in assessment.reasoning

    def test_multiple_evidences_uses_top_score(self):
        """Assessment uses maximum combined_score among evidence items."""
        ev1 = make_evidence("E1", combined_score=0.30, is_citable=True)
        ev2 = make_evidence("E2", combined_score=0.88, is_citable=True)
        ev3 = make_evidence("E3", combined_score=0.50, is_citable=True)

        assessment = assess_confidence([ev1, ev2, ev3])
        assert assessment.level == "high"
        assert assessment.score == 0.88
        assert assessment.should_abstain is False
        assert assessment.should_handoff is False


class TestConfidenceAssessmentDataclass:
    """Tests for ConfidenceAssessment dataclass defaults."""

    def test_dataclass_defaults(self):
        ca = ConfidenceAssessment(level="high", score=0.95, reasoning="Good")
        assert ca.level == "high"
        assert ca.score == 0.95
        assert ca.reasoning == "Good"
        assert ca.should_abstain is False
        assert ca.should_handoff is False
