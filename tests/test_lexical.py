"""
test_lexical.py — unit tests for LexicalRetriever.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb_loader import Chunk
from lexical import LexicalRetriever, _tokenize


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


def make_chunk(
    chunk_id: str = "chunk-1",
    filename: str = "01-returns.md",
    document_id: str = "DOC-01",
    title: str = "Returns Policy",
    status: str = "active",
    policy_authority: str = "official",
    audience: str = "customer",
    heading: str = "Return Window",
    content: str = "Items can be returned within 30 days of delivery.",
    precedence_score: float = 1.0,
) -> Chunk:
    """Create a Chunk object directly for deterministic testing without disk I/O."""
    return Chunk(
        chunk_id=chunk_id,
        filename=filename,
        document_id=document_id,
        title=title,
        status=status,
        policy_authority=policy_authority,
        audience=audience,
        heading=heading,
        content=content,
        precedence_score=precedence_score,
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """A standard corpus of diverse chunks for testing lexical retrieval."""
    return [
        make_chunk(
            chunk_id="chunk-returns",
            filename="01-returns.md",
            document_id="DOC-01",
            title="Returns and Exchanges",
            heading="Standard Return Policy",
            content="Customers can request a refund or return within 30 days of delivery for eligible items.",
            precedence_score=1.0,
        ),
        make_chunk(
            chunk_id="chunk-shipping",
            filename="02-shipping.md",
            document_id="DOC-02",
            title="Shipping Guidelines",
            heading="Expedited Shipping",
            content="Express delivery takes 2 business days. Standard shipping takes 5 business days.",
            precedence_score=0.9,
        ),
        make_chunk(
            chunk_id="chunk-warranty",
            filename="03-warranty.md",
            document_id="DOC-03",
            title="Warranty Coverage",
            heading="Limited Hardware Warranty",
            content="Hardware warranty covers manufacturing defects for 12 months. Accidental damage is excluded.",
            precedence_score=0.85,
        ),
        make_chunk(
            chunk_id="chunk-internal",
            filename="99-agent-notes.md",
            document_id="DOC-99",
            title="Agent Notes",
            status="active",
            policy_authority="none",
            audience="internal",
            heading="Exceptions and Escalations",
            content="Internal guidance on handling VIP refund requests and edge cases.",
            precedence_score=0.2,
        ),
    ]


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------


class TestLexicalEmptyAndNonMatching:
    """Tests for empty queries, non-matching queries, and empty indices."""

    def test_empty_query_returns_empty_results(self, sample_chunks):
        """An empty string query should return an empty list."""
        retriever = LexicalRetriever(sample_chunks)
        assert retriever.search("") == []

    def test_whitespace_query_returns_empty_results(self, sample_chunks):
        """Queries containing only whitespace should return an empty list."""
        retriever = LexicalRetriever(sample_chunks)
        assert retriever.search("   ") == []
        assert retriever.search("\t\n  \r") == []

    def test_punctuation_only_query_returns_empty_results(self, sample_chunks):
        """Queries containing only punctuation tokens should return an empty list."""
        retriever = LexicalRetriever(sample_chunks)
        assert retriever.search("!@#$%^&*()_+-=[]{}|;':,.<>?/") == []

    def test_non_matching_query_returns_empty_results(self, sample_chunks):
        """Search query with terms not in any document should return an empty list."""
        retriever = LexicalRetriever(sample_chunks)
        results = retriever.search("quantum teleportation astrophysics")
        assert results == []

    def test_empty_chunks_index(self):
        """A LexicalRetriever with no chunks should safely return empty results."""
        retriever = LexicalRetriever([])
        assert retriever.search("return") == []
        assert retriever.search("") == []


class TestLexicalKeywordSearch:
    """Tests for keyword search functionality and matching accuracy."""

    def test_known_keyword_search_returns_results(self, sample_chunks):
        """Searching for a known keyword returns matching documents with positive scores."""
        retriever = LexicalRetriever(sample_chunks)
        results = retriever.search("warranty")

        assert len(results) == 1
        assert results[0]["document_id"] == "DOC-03"
        assert results[0]["heading"] == "Limited Hardware Warranty"
        assert results[0]["lexical_score"] > 0.0

    def test_keyword_in_heading_matches(self, sample_chunks):
        """Keywords present only in chunk heading should be matched."""
        retriever = LexicalRetriever(sample_chunks)
        results = retriever.search("expedited")

        assert len(results) >= 1
        assert any(r["document_id"] == "DOC-02" for r in results)

    def test_keyword_in_content_matches(self, sample_chunks):
        """Keywords present only in chunk content should be matched."""
        retriever = LexicalRetriever(sample_chunks)
        results = retriever.search("defects")

        assert len(results) >= 1
        assert results[0]["document_id"] == "DOC-03"

    def test_case_insensitive_matching(self, sample_chunks):
        """Search should be case-insensitive."""
        retriever = LexicalRetriever(sample_chunks)
        lower_res = retriever.search("refund")
        upper_res = retriever.search("REFUND")
        mixed_res = retriever.search("ReFuNd")

        assert len(lower_res) == len(upper_res) == len(mixed_res)
        assert len(lower_res) > 0
        assert lower_res[0]["document_id"] == upper_res[0]["document_id"]
        assert lower_res[0]["lexical_score"] == upper_res[0]["lexical_score"] == mixed_res[0]["lexical_score"]

    def test_multi_term_query_matches_multiple_documents(self, sample_chunks):
        """Multi-term queries match documents containing any of the query terms."""
        retriever = LexicalRetriever(sample_chunks)
        # "refund" appears in DOC-01 and DOC-99, "shipping" appears in DOC-02
        results = retriever.search("refund shipping")
        doc_ids = {r["document_id"] for r in results}

        assert "DOC-01" in doc_ids
        assert "DOC-02" in doc_ids


class TestLexicalRankingAndSorting:
    """Tests for BM25 score ordering and score calculation properties."""

    def test_results_sorted_by_score_descending(self):
        """Results must always be sorted by lexical_score in descending order."""
        c1 = make_chunk(
            chunk_id="c1",
            document_id="DOC-HIGH",
            heading="Refund Policy",
            content="Refund refund refund processing. Request a refund for full refund amount.",
        )
        c2 = make_chunk(
            chunk_id="c2",
            document_id="DOC-LOW",
            heading="Billing Guide",
            content="Under rare conditions customers can obtain a refund.",
        )
        c3 = make_chunk(
            chunk_id="c3",
            document_id="DOC-NONE",
            heading="Account Settings",
            content="Update your password and profile notification preferences.",
        )

        retriever = LexicalRetriever([c1, c2, c3])
        results = retriever.search("refund")

        assert len(results) == 2
        assert results[0]["document_id"] == "DOC-HIGH"
        assert results[1]["document_id"] == "DOC-LOW"
        assert results[0]["lexical_score"] > results[1]["lexical_score"]

        # Ensure strictly descending order for all results
        scores = [r["lexical_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestLexicalStopWords:
    """Tests for stop word filtering behavior."""

    def test_stop_words_only_query_returns_empty_results(self, sample_chunks):
        """Queries composed entirely of stop words must not match anything."""
        retriever = LexicalRetriever(sample_chunks)
        results = retriever.search("the is are was were be been have has do does will would")
        assert results == []

    def test_stop_words_in_mixed_query_are_ignored(self, sample_chunks):
        """Stop words in a query should be stripped without penalizing content terms."""
        retriever = LexicalRetriever(sample_chunks)
        query_with_stop_words = "what is the warranty coverage"
        query_keyword_only = "warranty coverage"

        res_with = retriever.search(query_with_stop_words)
        res_without = retriever.search(query_keyword_only)

        assert len(res_with) == len(res_without)
        assert [r["document_id"] for r in res_with] == [r["document_id"] for r in res_without]
        assert [r["lexical_score"] for r in res_with] == [r["lexical_score"] for r in res_without]


class TestLexicalTopKLimits:
    """Tests for top_k parameter bounds and truncation."""

    def test_top_k_limits_results(self):
        """top_k parameter strictly caps the maximum number of results returned."""
        chunks = [
            make_chunk(
                chunk_id=f"c{i}",
                document_id=f"DOC-{i}",
                heading=f"Policy Section {i}",
                content=f"Important policy guidelines and policy procedures number {i}.",
            )
            for i in range(8)
        ]
        retriever = LexicalRetriever(chunks)

        results_k1 = retriever.search("policy", top_k=1)
        assert len(results_k1) == 1

        results_k3 = retriever.search("policy", top_k=3)
        assert len(results_k3) == 3

        results_k5 = retriever.search("policy", top_k=5)
        assert len(results_k5) == 5

        # Requesting more than total matches returns all available matches
        results_k20 = retriever.search("policy", top_k=20)
        assert len(results_k20) == 8


class TestLexicalResultStructure:
    """Tests for the dictionary structure and metadata preservation of results."""

    def test_results_contain_retrieval_method_lexical(self, sample_chunks):
        """Every returned dictionary must specify retrieval_method='lexical'."""
        retriever = LexicalRetriever(sample_chunks)
        results = retriever.search("delivery")

        assert len(results) > 0
        for res in results:
            assert res["retrieval_method"] == "lexical"

    def test_results_preserve_all_chunk_metadata_and_formatted_content(self, sample_chunks):
        """Result dict preserves chunk fields, formats content as 'heading\n\ncontent', and rounds scores."""
        retriever = LexicalRetriever(sample_chunks)
        results = retriever.search("warranty")

        assert len(results) == 1
        res = results[0]

        expected_keys = {
            "filename",
            "document_id",
            "title",
            "status",
            "policy_authority",
            "audience",
            "heading",
            "content",
            "lexical_score",
            "precedence_score",
            "retrieval_method",
        }
        assert set(res.keys()) == expected_keys
        assert res["filename"] == "03-warranty.md"
        assert res["document_id"] == "DOC-03"
        assert res["title"] == "Warranty Coverage"
        assert res["status"] == "active"
        assert res["policy_authority"] == "official"
        assert res["audience"] == "customer"
        assert res["heading"] == "Limited Hardware Warranty"
        assert res["content"] == (
            "Limited Hardware Warranty\n\n"
            "Hardware warranty covers manufacturing defects for 12 months. Accidental damage is excluded."
        )
        assert isinstance(res["lexical_score"], float)
        assert res["lexical_score"] > 0.0
        assert res["precedence_score"] == 0.85
        assert res["retrieval_method"] == "lexical"


class TestTokenizer:
    """Unit tests for the internal _tokenize function."""

    def test_tokenize_filters_stop_words_and_lowercases(self):
        """_tokenize lowercases words and excludes common stop words."""
        tokens = _tokenize("The 30-day Return Policy is Very Strict!")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "very" not in tokens
        assert "30-day" in tokens
        assert "return" in tokens
        assert "policy" in tokens
        assert "strict" in tokens

    def test_tokenize_empty_string(self):
        """_tokenize on empty string returns empty list."""
        assert _tokenize("") == []
