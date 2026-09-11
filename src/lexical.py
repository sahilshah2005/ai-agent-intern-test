"""
lexical.py — lightweight BM25-style lexical retriever.

Uses Python stdlib only (collections.Counter, math) — no new dependencies.
Provides term-frequency / inverse-document-frequency scoring over the same
Chunk objects used by the vector retriever.

The lexical retriever complements the vector retriever by catching exact
keyword matches that embedding models may miss (e.g., product names,
policy-specific terms like "30 calendar days", order IDs in document text).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from config import LEXICAL_TOP_K
from kb_loader import Chunk


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# Common English stop words to exclude from scoring
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "after", "before", "above", "below", "and", "but", "or",
    "not", "no", "if", "it", "its", "this", "that", "these", "those",
    "i", "my", "me", "we", "our", "you", "your", "he", "she", "they",
    "them", "what", "which", "who", "how", "when", "where", "why",
    "all", "each", "every", "any", "some", "so", "than", "too", "very",
})


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase terms, excluding stop words."""
    return [
        tok for tok in _TOKEN_RE.findall(text.lower())
        if tok not in _STOP_WORDS
    ]


# ---------------------------------------------------------------------------
# BM25 parameters
# ---------------------------------------------------------------------------

_K1 = 1.5   # term frequency saturation
_B = 0.75   # document length normalisation


# ---------------------------------------------------------------------------
# Lexical index
# ---------------------------------------------------------------------------


@dataclass
class _IndexedDoc:
    """Internal representation of an indexed document."""
    chunk_index: int  # position in the chunks list
    tf: Counter       # term frequencies
    length: int       # total token count


class LexicalRetriever:
    """
    BM25-style lexical retriever over knowledge-base chunks.

    Builds an inverted index from the same Chunk objects used by the
    vector retriever. Scores queries using a standard BM25 formula.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._docs: list[_IndexedDoc] = []
        self._df: Counter = Counter()  # document frequency per term
        self._avg_dl: float = 0.0
        self._build_index(chunks)

    def _build_index(self, chunks: list[Chunk]) -> None:
        """Build the inverted index from chunks."""
        total_length = 0
        for i, chunk in enumerate(chunks):
            text = f"{chunk.heading} {chunk.content}"
            tokens = _tokenize(text)
            tf = Counter(tokens)
            doc = _IndexedDoc(chunk_index=i, tf=tf, length=len(tokens))
            self._docs.append(doc)
            total_length += len(tokens)

            # Update document frequency (each unique term counts once per doc)
            for term in tf:
                self._df[term] += 1

        n = len(chunks)
        self._avg_dl = total_length / n if n > 0 else 1.0

    def search(
        self, query: str, top_k: int = LEXICAL_TOP_K
    ) -> list[dict[str, Any]]:
        """
        Score all documents against ``query`` using BM25 and return top-k.

        Each result dict includes:
          - All standard chunk metadata
          - lexical_score: the BM25 score
          - retrieval_method: "lexical"
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        n = len(self._docs)
        scores: list[tuple[int, float]] = []

        for doc in self._docs:
            score = 0.0
            for term in query_tokens:
                if term not in doc.tf:
                    continue
                tf_val = doc.tf[term]
                df_val = self._df.get(term, 0)

                # IDF with smoothing (BM25 variant)
                idf = math.log((n - df_val + 0.5) / (df_val + 0.5) + 1.0)

                # BM25 term score
                numerator = tf_val * (_K1 + 1)
                denominator = tf_val + _K1 * (
                    1 - _B + _B * doc.length / self._avg_dl
                )
                score += idf * (numerator / denominator)

            if score > 0:
                scores.append((doc.chunk_index, score))

        # Sort by score descending, take top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]

        results: list[dict[str, Any]] = []
        for chunk_idx, bm25_score in top_scores:
            chunk = self._chunks[chunk_idx]
            results.append({
                "filename": chunk.filename,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "status": chunk.status,
                "policy_authority": chunk.policy_authority,
                "audience": chunk.audience,
                "heading": chunk.heading,
                "content": f"{chunk.heading}\n\n{chunk.content}",
                "lexical_score": round(bm25_score, 4),
                "precedence_score": round(chunk.precedence_score, 4),
                "retrieval_method": "lexical",
            })

        return results
