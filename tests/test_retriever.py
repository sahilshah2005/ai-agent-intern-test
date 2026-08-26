"""
test_retriever.py — unit tests for the knowledge-base retriever.

These tests cover pure parsing and scoring functions that do not
require ChromaDB or network access. KnowledgeRetriever integration
tests are in the evaluation suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb_loader import (
    Chunk,
    load_chunks,
    parse_frontmatter,
    split_by_headings,
)
from scoring import compute_precedence_score
from retriever import detect_conflicts
from config import KNOWLEDGE_BASE_DIR


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        text = "---\nstatus: active\npolicy_authority: official\n---\n# Body\nContent."
        fm, body = parse_frontmatter(text)
        assert fm["status"] == "active"
        assert fm["policy_authority"] == "official"
        assert "# Body" in body

    def test_no_frontmatter(self):
        text = "# Just a heading\n\nSome content."
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert "# Just a heading" in body

    def test_empty_frontmatter(self):
        text = "---\n---\n# Body"
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert "# Body" in body

    def test_malformed_yaml_returns_empty_dict(self):
        text = "---\n: invalid: yaml: here:\n---\n# Body"
        fm, body = parse_frontmatter(text)
        # Should not raise; returns empty dict
        assert isinstance(fm, dict)

    def test_all_doc01_fields(self):
        """Parse returns-policy-current and verify all expected fields."""
        doc = (KNOWLEDGE_BASE_DIR / "01-returns-policy-current.md").read_text()
        fm, body = parse_frontmatter(doc)
        assert fm["document_id"] == "RET-2026-01"
        assert fm["status"] == "active"
        assert fm["policy_authority"] == "official"
        assert fm["audience"] == "customer"
        assert fm["supersedes"] == "RET-2024-01"


# ---------------------------------------------------------------------------
# compute_precedence_score
# ---------------------------------------------------------------------------


class TestComputePrecedenceScore:
    def test_active_official_customer_gets_bonus(self):
        fm = {"status": "active", "policy_authority": "official", "audience": "customer"}
        score = compute_precedence_score(fm)
        assert score > 0, "Active official customer doc should have positive score"

    def test_superseded_gets_penalty(self):
        fm = {"status": "superseded", "policy_authority": "official", "audience": "customer"}
        score = compute_precedence_score(fm)
        assert score < 0, "Superseded doc should have negative score"

    def test_draft_gets_penalty(self):
        fm = {"status": "draft", "policy_authority": "none", "audience": "internal"}
        score = compute_precedence_score(fm)
        assert score < -0.5, "Draft internal doc should have large negative score"

    def test_internal_gets_penalty(self):
        fm = {"status": "active", "policy_authority": "official", "audience": "internal"}
        score = compute_precedence_score(fm)
        # Has authority bonus but internal penalty
        assert score < compute_precedence_score(
            {"status": "active", "policy_authority": "official", "audience": "customer"}
        )

    def test_doc01_higher_than_doc02(self):
        """Current returns policy must score higher than superseded legacy."""
        fm_current = {
            "status": "active",
            "policy_authority": "official",
            "audience": "customer",
        }
        fm_legacy = {
            "status": "superseded",
            "policy_authority": "official",
            "audience": "customer",
        }
        assert compute_precedence_score(fm_current) > compute_precedence_score(fm_legacy)

    def test_doc14_has_very_low_score(self):
        """Internal migration scratchpad (draft + internal + no authority) must score low."""
        fm = {"status": "draft", "policy_authority": "none", "audience": "internal"}
        score = compute_precedence_score(fm)
        fm_active = {"status": "active", "policy_authority": "official", "audience": "customer"}
        active_score = compute_precedence_score(fm_active)
        assert score < active_score - 0.5


# ---------------------------------------------------------------------------
# split_by_headings
# ---------------------------------------------------------------------------


class TestSplitByHeadings:
    def test_splits_at_double_hash(self):
        body = "Intro text.\n\n## Section One\n\nContent one.\n\n## Section Two\n\nContent two."
        chunks = split_by_headings(body, "Doc Title")
        headings = [h for h, _ in chunks]
        assert "Section One" in headings
        assert "Section Two" in headings

    def test_intro_uses_fallback_title(self):
        body = "Some intro.\n\n## Section\n\nContent."
        chunks = split_by_headings(body, "My Title")
        assert chunks[0][0] == "My Title"

    def test_empty_sections_excluded(self):
        body = "## Section One\n\nContent.\n\n## Empty Section\n\n## Section Three\n\nMore."
        chunks = split_by_headings(body, "T")
        # Empty sections should be filtered out
        headings = [h for h, _ in chunks]
        assert "Empty Section" not in headings

    def test_real_doc01_produces_expected_sections(self):
        from retriever import parse_frontmatter
        doc = (KNOWLEDGE_BASE_DIR / "01-returns-policy-current.md").read_text()
        fm, body = parse_frontmatter(doc)
        chunks = split_by_headings(body, fm.get("title", ""))
        headings = [h for h, _ in chunks]
        assert any("return window" in h.lower() for h in headings)
        assert any("exclusion" in h.lower() for h in headings)


# ---------------------------------------------------------------------------
# load_chunks — integration (reads real files, no ChromaDB)
# ---------------------------------------------------------------------------


class TestLoadChunks:
    def test_loads_all_14_documents(self):
        chunks = load_chunks(KNOWLEDGE_BASE_DIR)
        filenames = {c.filename for c in chunks}
        assert len(filenames) == 14

    def test_doc01_is_active_official(self):
        chunks = load_chunks(KNOWLEDGE_BASE_DIR)
        doc01 = [c for c in chunks if c.filename == "01-returns-policy-current.md"]
        assert doc01, "doc 01 chunks should exist"
        assert doc01[0].status == "active"
        assert doc01[0].policy_authority == "official"
        assert doc01[0].audience == "customer"

    def test_doc02_is_superseded(self):
        chunks = load_chunks(KNOWLEDGE_BASE_DIR)
        doc02 = [c for c in chunks if c.filename == "02-returns-policy-legacy.md"]
        assert doc02[0].status == "superseded"

    def test_doc14_is_draft_internal(self):
        chunks = load_chunks(KNOWLEDGE_BASE_DIR)
        doc14 = [c for c in chunks if c.filename == "14-internal-content-migration-notes.md"]
        assert doc14[0].status == "draft"
        assert doc14[0].audience == "internal"
        assert doc14[0].policy_authority == "none"

    def test_doc01_precedence_higher_than_doc02(self):
        chunks = load_chunks(KNOWLEDGE_BASE_DIR)
        doc01 = next(c for c in chunks if c.filename == "01-returns-policy-current.md")
        doc02 = next(c for c in chunks if c.filename == "02-returns-policy-legacy.md")
        assert doc01.precedence_score > doc02.precedence_score

    def test_all_chunks_have_required_fields(self):
        chunks = load_chunks(KNOWLEDGE_BASE_DIR)
        for c in chunks:
            assert c.chunk_id
            assert c.filename
            assert c.heading
            assert c.content


# ---------------------------------------------------------------------------
# detect_conflicts
# ---------------------------------------------------------------------------


class TestDetectConflicts:
    def _make_chunk(self, filename: str, heading: str, content: str) -> dict:
        return {
            "filename": filename,
            "status": "active",
            "policy_authority": "official",
            "audience": "customer",
            "heading": heading,
            "content": content,
        }

    def test_detects_breeze_tumbler_conflict(self):
        """Docs 11 and 12 conflict on dishwasher safety."""
        chunk_11 = self._make_chunk(
            "11-product-care.md",
            "Breeze Tumbler",
            "The stainless-steel body of the Breeze Tumbler should be hand-washed.",
        )
        chunk_12 = self._make_chunk(
            "12-breeze-tumbler-product-card.md",
            "Cleaning",
            "All components are dishwasher safe, with the top rack recommended.",
        )
        conflicts = detect_conflicts([chunk_11, chunk_12])
        assert len(conflicts) == 1
        _, _, desc = conflicts[0]
        assert "11-product-care.md" in desc or "12-breeze-tumbler-product-card.md" in desc

    def test_no_conflict_same_file(self):
        """Two chunks from the same file never trigger a conflict."""
        chunk_a = self._make_chunk("11-product-care.md", "Section A", "hand-washed")
        chunk_b = self._make_chunk("11-product-care.md", "Section B", "dishwasher safe")
        conflicts = detect_conflicts([chunk_a, chunk_b])
        assert len(conflicts) == 0

    def test_no_conflict_without_keywords(self):
        """Chunks that do not match the conflict keyword pairs produce no conflict."""
        chunk_a = self._make_chunk("11-product-care.md", "Bags", "Spot-clean with mild soap.")
        chunk_b = self._make_chunk("12-breeze-tumbler-product-card.md", "Details", "20-ounce tumbler.")
        conflicts = detect_conflicts([chunk_a, chunk_b])
        assert len(conflicts) == 0

    def test_superseded_not_flagged_as_conflict(self):
        """Only active+official chunks participate in conflict detection."""
        chunk_superseded = {
            "filename": "02-returns-policy-legacy.md",
            "status": "superseded",
            "policy_authority": "official",
            "audience": "customer",
            "heading": "Return window",
            "content": "hand-washed body",
        }
        chunk_active = self._make_chunk(
            "11-product-care.md", "Breeze", "dishwasher safe"
        )
        conflicts = detect_conflicts([chunk_superseded, chunk_active])
        # Superseded is not active, so no conflict flagged
        assert len(conflicts) == 0
