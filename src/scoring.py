"""
scoring.py — pure metadata-based precedence scoring functions.

Extracted here so they can be unit-tested without requiring chromadb.
"""

from __future__ import annotations

from typing import Any

from config import (
    AUTHORITY_BONUS,
    DRAFT_PENALTY,
    INTERNAL_PENALTY,
    LEGACY_PENALTY,
    NO_AUTHORITY_PENALTY,
)


def compute_precedence_score(fm: dict[str, Any]) -> float:
    """
    Compute a metadata-based score adjustment from YAML front-matter fields.

    The score is ADDED to cosine similarity during re-ranking.
    Positive values boost authoritative current documents; negative values
    penalise superseded, draft, or internal documents.
    """
    score = 0.0
    status = str(fm.get("status", "")).lower()
    authority = str(fm.get("policy_authority", "")).lower()
    audience = str(fm.get("audience", "customer")).lower()

    if status == "active" and authority == "official" and audience == "customer":
        score += AUTHORITY_BONUS

    if status == "superseded":
        score -= LEGACY_PENALTY
    elif status == "draft":
        score -= DRAFT_PENALTY

    if audience == "internal":
        score -= INTERNAL_PENALTY

    if authority == "none":
        score -= NO_AUTHORITY_PENALTY

    return score
