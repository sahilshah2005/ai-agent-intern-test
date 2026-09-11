"""
confidence.py — evidence-confidence assessment.

Provides a simple heuristic model that distinguishes between high, medium,
and low confidence based on evidence quality. This drives abstention and
handoff decisions.

The scoring is deliberately simple and explainable — it does not pretend
to be a calibrated probabilistic model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD
from evidence import Evidence


# ---------------------------------------------------------------------------
# Confidence result
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceAssessment:
    """Result of evidence confidence assessment."""
    level: str          # "high", "medium", "low"
    score: float        # 0.0–1.0 heuristic
    reasoning: str
    should_abstain: bool = False
    should_handoff: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_confidence(
    evidence_list: list[Evidence],
    has_conflicts: bool = False,
    has_tool_result: bool = False,
    tool_error: bool = False,
) -> ConfidenceAssessment:
    """
    Assess confidence based on evidence quality.

    Heuristic factors:
      - Top evidence score + active + official → high
      - Moderate scores or mixed authority → medium
      - No relevant evidence or all internal/draft → low
      - Detected conflict → medium + handoff
      - Tool error → low + handoff
      - No evidence at all → low + abstain
    """
    if not evidence_list and not has_tool_result:
        return ConfidenceAssessment(
            level="low",
            score=0.0,
            reasoning="No evidence retrieved and no tool results",
            should_abstain=True,
            should_handoff=True,
        )

    # Tool error case
    if tool_error:
        return ConfidenceAssessment(
            level="low",
            score=0.1,
            reasoning="Order tool returned an error",
            should_abstain=False,
            should_handoff=True,
        )

    # If only tool results and no KB evidence needed
    if has_tool_result and not evidence_list:
        return ConfidenceAssessment(
            level="high",
            score=0.9,
            reasoning="Tool returned valid result",
        )

    # Evaluate evidence quality
    citable = [e for e in evidence_list if e.is_citable]
    top_score = max((e.combined_score for e in evidence_list), default=0.0)
    citable_count = len(citable)
    total_count = len(evidence_list)

    # Conflict handling
    if has_conflicts:
        return ConfidenceAssessment(
            level="medium",
            score=0.5,
            reasoning=(
                f"Conflict detected among {citable_count} citable sources; "
                "recommend human confirmation"
            ),
            should_handoff=True,
        )

    # Score-based assessment
    if citable_count == 0:
        return ConfidenceAssessment(
            level="low",
            score=0.15,
            reasoning=(
                f"No citable (active+official+customer) evidence among "
                f"{total_count} retrieved chunks"
            ),
            should_abstain=True,
            should_handoff=True,
        )

    if top_score >= CONFIDENCE_HIGH_THRESHOLD and citable_count >= 1:
        return ConfidenceAssessment(
            level="high",
            score=min(top_score, 1.0),
            reasoning=(
                f"Top evidence score {top_score:.2f} ≥ {CONFIDENCE_HIGH_THRESHOLD} "
                f"with {citable_count} citable source(s)"
            ),
        )

    if top_score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return ConfidenceAssessment(
            level="medium",
            score=top_score,
            reasoning=(
                f"Top evidence score {top_score:.2f} between "
                f"{CONFIDENCE_MEDIUM_THRESHOLD}–{CONFIDENCE_HIGH_THRESHOLD}; "
                "answer with caution"
            ),
        )

    return ConfidenceAssessment(
        level="low",
        score=top_score,
        reasoning=(
            f"Top evidence score {top_score:.2f} < {CONFIDENCE_MEDIUM_THRESHOLD}; "
            "insufficient evidence"
        ),
        should_abstain=True,
        should_handoff=True,
    )
