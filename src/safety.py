"""
safety.py — centralized safety / response-decision layer.

Consolidates important decisions (answer, clarify, abstain, handoff, refuse)
that were previously scattered across agent.py, system_prompt.py, and
order_tool.py.

This module does NOT call the LLM — it produces deterministic decisions
based on evidence, confidence, and routing signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from confidence import ConfidenceAssessment
from evidence import Evidence


# ---------------------------------------------------------------------------
# Response decision
# ---------------------------------------------------------------------------


@dataclass
class ResponseDecision:
    """Centralized decision about how to respond."""
    action: str           # "answer", "clarify", "abstain", "handoff", "refuse"
    reason: str
    should_cite: bool = True
    conflict_warning: str | None = None
    confidence_level: str = "unknown"


# ---------------------------------------------------------------------------
# Prompt injection detection patterns
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(your\s+)?(previous|prior|above)\s+(instructions|rules|prompt)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.I),
    re.compile(r"(reveal|show|print|display|output)\s+(your|the)\s+(system\s+)?(prompt|instructions|rules)", re.I),
    re.compile(r"(act\s+as|pretend\s+to\s+be|you\s+are)\s+(a\s+)?system\s+(admin|administrator|developer)", re.I),
    re.compile(r"forget\s+(everything|all|your\s+instructions)", re.I),
    re.compile(r"override\s+(your\s+)?(safety|security|rules|instructions)", re.I),
    re.compile(r"as\s+a\s+(system\s+)?administrator", re.I),
    re.compile(r"(developer|admin)\s+mode", re.I),
]

# Phrases indicating the user wants an action the system can't perform
_ACTION_PATTERNS = [
    re.compile(r"(approve|process|complete)\s+(my\s+)?(refund|return|cancellation|exchange|replacement)", re.I),
    re.compile(r"(cancel|change|modify)\s+(my\s+)?order", re.I),
    re.compile(r"(give|issue|send)\s+(me\s+)?(a\s+)?(coupon|credit|discount|refund)", re.I),
    re.compile(r"change\s+(my\s+)?(shipping\s+)?address", re.I),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_injection(text: str) -> bool:
    """Return True if the text contains prompt injection patterns."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def detect_action_request(text: str) -> bool:
    """Return True if the user is requesting an action the system can't perform."""
    return any(p.search(text) for p in _ACTION_PATTERNS)


def make_response_decision(
    user_message: str,
    evidence_list: list[Evidence],
    confidence: ConfidenceAssessment,
    conflicts: list[tuple[dict, dict, str]],
    tool_calls_made: list[dict[str, Any]],
    tool_error: bool = False,
) -> ResponseDecision:
    """
    Determine how the agent should respond based on all available signals.

    Decision hierarchy:
      1. Prompt injection detected → refuse
      2. Action request detected → handoff
      3. Tool error → handoff with explanation
      4. Active conflicts → answer with conflict warning + handoff
      5. Low confidence → abstain or handoff
      6. Medium confidence → answer with caution
      7. High confidence → answer directly
    """
    # 1. Check for prompt injection
    if detect_injection(user_message):
        return ResponseDecision(
            action="refuse",
            reason="Prompt injection pattern detected in user message",
            should_cite=False,
            confidence_level=confidence.level,
        )

    # 2. Check for action requests
    if detect_action_request(user_message):
        return ResponseDecision(
            action="handoff",
            reason="User is requesting an action that requires human support",
            should_cite=True,
            confidence_level=confidence.level,
        )

    # 3. Tool error
    if tool_error:
        return ResponseDecision(
            action="handoff",
            reason="Order tool returned an error; recommend human support",
            should_cite=False,
            confidence_level="low",
        )

    # 4. Active conflicts
    if conflicts:
        conflict_desc = "; ".join(desc for _, _, desc in conflicts)
        return ResponseDecision(
            action="answer",
            reason="Answering with conflict warning; recommend human confirmation",
            should_cite=True,
            conflict_warning=conflict_desc,
            confidence_level="medium",
        )

    # 5. Low confidence → abstain
    if confidence.should_abstain:
        return ResponseDecision(
            action="abstain",
            reason=confidence.reasoning,
            should_cite=False,
            confidence_level="low",
        )

    # 6. Handoff recommended by confidence assessment
    if confidence.should_handoff:
        return ResponseDecision(
            action="handoff",
            reason=confidence.reasoning,
            should_cite=True,
            confidence_level=confidence.level,
        )

    # 7. Medium confidence → answer with caution
    if confidence.level == "medium":
        return ResponseDecision(
            action="answer",
            reason=confidence.reasoning,
            should_cite=True,
            confidence_level="medium",
        )

    # 8. High confidence → answer directly
    return ResponseDecision(
        action="answer",
        reason=confidence.reasoning,
        should_cite=True,
        confidence_level="high",
    )
