"""
evidence.py — evidence-level citation architecture.

Assigns stable evidence IDs (E1, E2, ...) to retrieved passages and
provides programmatic citation validation after LLM generation.

This replaces fragile regex-based filename extraction with deterministic,
verifiable citation tracking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Evidence dataclass
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """A single piece of retrieved evidence with a stable identifier."""
    evidence_id: str           # "E1", "E2", etc.
    filename: str
    heading: str
    title: str
    status: str
    policy_authority: str
    audience: str
    content: str
    is_citable: bool           # active + official + customer
    retrieval_method: str      # "vector", "lexical", "hybrid"
    combined_score: float
    precedence_score: float


@dataclass
class CitationResult:
    """Result of citation validation after LLM generation."""
    valid_citations: list[str] = field(default_factory=list)      # evidence IDs
    invalid_citations: list[str] = field(default_factory=list)    # evidence IDs that don't exist
    non_citable_citations: list[str] = field(default_factory=list)  # internal/draft refs
    cited_sources: list[dict[str, Any]] = field(default_factory=list)  # for output
    is_valid: bool = True


# ---------------------------------------------------------------------------
# Evidence ID pattern
# ---------------------------------------------------------------------------

_EVIDENCE_ID_RE = re.compile(r"\[E(\d+)\]")

# Also detect the human-readable citation format for backward compat
_FILENAME_RE = re.compile(r"\b(\d{2}-[\w-]+\.md)\b")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_evidence_list(retrieved_chunks: list[dict[str, Any]]) -> list[Evidence]:
    """
    Assign stable evidence IDs to retrieved chunks.

    Each chunk gets an ID like E1, E2, etc. A chunk is marked ``is_citable``
    only if it is active, official, and customer-facing.
    """
    evidence_list: list[Evidence] = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        is_citable = (
            chunk.get("status") == "active"
            and chunk.get("policy_authority") == "official"
            and chunk.get("audience", "customer") != "internal"
        )

        evidence_list.append(Evidence(
            evidence_id=f"E{i}",
            filename=chunk["filename"],
            heading=chunk["heading"],
            title=chunk.get("title", chunk["filename"]),
            status=chunk.get("status", "unknown"),
            policy_authority=chunk.get("policy_authority", "unknown"),
            audience=chunk.get("audience", "customer"),
            content=chunk.get("content", ""),
            is_citable=is_citable,
            retrieval_method=chunk.get("retrieval_method", "unknown"),
            combined_score=chunk.get("combined_score", 0.0),
            precedence_score=chunk.get("precedence_score", 0.0),
        ))

    return evidence_list


def build_evidence_context(
    evidence_list: list[Evidence],
    conflicts: list[tuple[dict, dict, str]],
) -> str:
    """
    Format evidence list as a context string for LLM injection.

    Each chunk is labelled with its evidence ID and authority metadata.
    """
    if not evidence_list:
        return ""

    parts: list[str] = []

    if conflicts:
        lines = [
            "⚠️  CONFLICT DETECTED — Two active official sources contradict "
            "each other on this topic.\n"
            "You MUST surface this conflict in your response and "
            "recommend human confirmation.\n"
        ]
        for _, _, desc in conflicts:
            lines.append(f"  • {desc}")
        parts.append("\n".join(lines))

    for ev in evidence_list:
        labels: list[str] = []
        if ev.status != "active":
            labels.append(ev.status.upper())
        if ev.policy_authority != "official":
            labels.append("NON-AUTHORITATIVE")
        if ev.audience == "internal":
            labels.append("INTERNAL — do not cite as customer-facing authority")

        label_str = f"[{' | '.join(labels)}] " if labels else ""

        chunk_text = (
            f"---\n"
            f"[{ev.evidence_id}] Source: {ev.filename} — {ev.heading}\n"
            f"Status: {ev.status} | Authority: {ev.policy_authority} | "
            f"Audience: {ev.audience}\n"
            f"{label_str}\n\n"
            f"{ev.content}\n"
        )
        parts.append(chunk_text)

    return "\n".join(parts)


def validate_citations(
    response_text: str,
    evidence_list: list[Evidence],
) -> CitationResult:
    """
    Validate citations in the LLM response against available evidence.

    Checks:
      1. Every cited evidence ID (e.g. [E1]) actually exists
      2. Cited evidence is citable (active + official + customer)
      3. Filenames mentioned in the response match retrieved evidence

    Returns a CitationResult with valid, invalid, and non-citable lists.
    """
    result = CitationResult()
    evidence_by_id = {ev.evidence_id: ev for ev in evidence_list}

    # Check evidence ID citations [E1], [E2], etc.
    cited_ids = set(_EVIDENCE_ID_RE.findall(response_text))
    for eid_num in cited_ids:
        eid = f"E{eid_num}"
        if eid not in evidence_by_id:
            result.invalid_citations.append(eid)
            result.is_valid = False
        elif not evidence_by_id[eid].is_citable:
            result.non_citable_citations.append(eid)
            result.is_valid = False
        else:
            result.valid_citations.append(eid)
            ev = evidence_by_id[eid]
            result.cited_sources.append({
                "filename": ev.filename,
                "title": ev.title,
                "heading": ev.heading,
            })

    # Also check filename-based citations (backward compat)
    mentioned_files = set(_FILENAME_RE.findall(response_text))
    evidence_by_file = {}
    for ev in evidence_list:
        if ev.filename not in evidence_by_file:
            evidence_by_file[ev.filename] = ev

    for fname in mentioned_files:
        if fname in evidence_by_file:
            ev = evidence_by_file[fname]
            if ev.is_citable:
                # Add to cited sources if not already present via evidence ID
                if not any(s["filename"] == fname for s in result.cited_sources):
                    result.cited_sources.append({
                        "filename": ev.filename,
                        "title": ev.title,
                        "heading": ev.heading,
                    })
            elif ev.audience == "internal" or ev.status != "active":
                if fname not in [e for e in result.non_citable_citations]:
                    result.non_citable_citations.append(f"file:{fname}")
                    result.is_valid = False

    return result


def transform_evidence_ids_to_readable(
    response_text: str,
    evidence_list: list[Evidence],
) -> str:
    """
    Replace evidence ID references [E1] with human-readable citations.

    Example: [E1] → (Source: 01-returns-policy-current.md — Return Window)
    """
    evidence_by_id = {ev.evidence_id: ev for ev in evidence_list}

    def _replacer(match: re.Match) -> str:
        eid = f"E{match.group(1)}"
        if eid in evidence_by_id:
            ev = evidence_by_id[eid]
            if ev.is_citable:
                return f"(Source: {ev.filename} — {ev.heading})"
        return match.group(0)  # leave as-is if not found

    return _EVIDENCE_ID_RE.sub(_replacer, response_text)
