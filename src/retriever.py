"""
retriever.py — ChromaDB-backed knowledge-base retriever.

Indexing and re-ranking strategy
──────────────────────────────────
1. Documents are parsed and chunked by kb_loader.load_chunks().
2. Each chunk is indexed with its metadata (status, authority, audience, etc.)
3. At query time, 15 candidates are fetched from ChromaDB (cosine distance).
4. Re-ranked by: combined_score = cosine_similarity + precedence_adjustment
   where precedence_adjustment is computed by scoring.compute_precedence_score().

Conflict detection
──────────────────
After retrieval, detect_conflicts() checks whether two active+official chunks
from DIFFERENT files both match contradictory keyword pairs (e.g. dishwasher
safety for Breeze Tumbler). When a conflict is found, the knowledge context
prepends a clear warning and the system prompt instructs the LLM to surface it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_DIR,
    TOP_K,
    TOP_K_CANDIDATES,
)
from kb_loader import Chunk, load_chunks, parse_frontmatter, split_by_headings  # re-export for tests
from scoring import compute_precedence_score  # re-export for tests

# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

# Pairs of contradictory keyword sets. A conflict is flagged when one chunk
# matches set A and another (from a different file) matches set B.
_CONFLICT_KEYWORD_PAIRS: list[tuple[frozenset[str], frozenset[str]]] = [
    (
        frozenset({"dishwasher safe", "dishwasher"}),          # doc 12 claims
        frozenset({"hand-wash", "hand wash", "hand-washed"}),  # doc 11 claims
    ),
]


def detect_conflicts(
    results: list[dict[str, Any]],
) -> list[tuple[dict, dict, str]]:
    """
    Detect genuine conflicts between active+official chunks from different files.

    Returns a list of (chunk_a, chunk_b, description) triples.
    """
    conflicts: list[tuple[dict, dict, str]] = []

    active_official = [
        r for r in results
        if r.get("status") == "active" and r.get("policy_authority") == "official"
    ]

    for chunk_a in active_official:
        for chunk_b in active_official:
            if chunk_a["filename"] == chunk_b["filename"]:
                continue

            text_a = (chunk_a["heading"] + " " + chunk_a["content"]).lower()
            text_b = (chunk_b["heading"] + " " + chunk_b["content"]).lower()

            for kw_a, kw_b in _CONFLICT_KEYWORD_PAIRS:
                a_matches = any(kw in text_a for kw in kw_a)
                b_matches = any(kw in text_b for kw in kw_b)

                if a_matches and b_matches:
                    # Avoid duplicate pairs (order-independent)
                    already = any(
                        c[0]["filename"] == chunk_b["filename"]
                        and c[1]["filename"] == chunk_a["filename"]
                        for c in conflicts
                    )
                    if already:
                        continue

                    desc = (
                        f"Conflict between active official sources: "
                        f"'{chunk_b['filename']}' ({chunk_b['heading']}) "
                        f"and '{chunk_a['filename']}' ({chunk_a['heading']}) "
                        f"provide contradictory guidance on the same topic."
                    )
                    conflicts.append((chunk_b, chunk_a, desc))

    return conflicts


# ---------------------------------------------------------------------------
# Knowledge retriever
# ---------------------------------------------------------------------------


class KnowledgeRetriever:
    """
    ChromaDB-backed retriever with metadata-based precedence re-ranking.

    The index is built lazily on first instantiation and persisted to disk.
    Call rebuild() to force a fresh index when the knowledge base changes.
    """

    _COLLECTION = "aster_row_kb_v1"

    def __init__(
        self,
        kb_dir: Path = KNOWLEDGE_BASE_DIR,
        persist_dir: Path = CHROMA_PERSIST_DIR,
    ) -> None:
        self._kb_dir = kb_dir
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._COLLECTION,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        if self._collection.count() == 0:
            self._build_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        chunks = load_chunks(self._kb_dir)
        if not chunks:
            return

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[f"{c.heading}\n\n{c.content}" for c in chunks],
            metadatas=[
                {
                    "filename": c.filename,
                    "document_id": c.document_id,
                    "title": c.title,
                    "status": c.status,
                    "policy_authority": c.policy_authority,
                    "audience": c.audience,
                    "heading": c.heading,
                    "precedence_score": c.precedence_score,
                }
                for c in chunks
            ],
        )

    def rebuild(self) -> None:
        """Delete and recreate the index (use when knowledge-base changes)."""
        self._client.delete_collection(self._COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=self._COLLECTION,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._build_index()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks for ``query``, re-ranked by
        combined score = cosine_similarity + metadata_precedence_adjustment.
        """
        total = self._collection.count()
        if total == 0:
            return []

        n_candidates = min(TOP_K_CANDIDATES, total)

        raw = self._collection.query(
            query_texts=[query],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )

        results: list[dict[str, Any]] = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            semantic_sim = max(0.0, 1.0 - float(dist))
            precedence = float(meta.get("precedence_score", 0.0))
            combined = semantic_sim + precedence

            results.append(
                {
                    "filename": meta["filename"],
                    "document_id": meta["document_id"],
                    "title": meta["title"],
                    "status": meta["status"],
                    "policy_authority": meta["policy_authority"],
                    "audience": meta["audience"],
                    "heading": meta["heading"],
                    "content": doc,
                    "semantic_score": round(semantic_sim, 4),
                    "precedence_score": round(precedence, 4),
                    "combined_score": round(combined, 4),
                }
            )

        results.sort(key=lambda r: r["combined_score"], reverse=True)
        return results[:top_k]

    def build_context(
        self, retrieved: list[dict[str, Any]], conflicts: list[tuple]
    ) -> str:
        """
        Format retrieved chunks as a knowledge-context string for LLM injection.
        Each chunk is labelled with its status and authority level.
        """
        if not retrieved:
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

        for r in retrieved:
            status = r["status"]
            authority = r["policy_authority"]
            audience = r["audience"]

            labels: list[str] = []
            if status != "active":
                labels.append(status.upper())
            if authority != "official":
                labels.append("NON-AUTHORITATIVE")
            if audience == "internal":
                labels.append("INTERNAL — do not cite as customer-facing authority")

            label_str = f"[{' | '.join(labels)}] " if labels else ""

            chunk_text = (
                f"---\n"
                f"Source: {r['filename']} — {r['heading']}\n"
                f"Status: {status} | Authority: {authority} | Audience: {audience}\n"
                f"{label_str}\n\n"
                f"{r['content']}\n"
            )
            parts.append(chunk_text)

        return "\n".join(parts)
