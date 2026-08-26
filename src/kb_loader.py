"""
kb_loader.py — knowledge-base document parsing and chunking.

This module is intentionally free of ChromaDB imports so it can be
unit-tested independently and imported by pure-Python tools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from config import KNOWLEDGE_BASE_DIR
from scoring import compute_precedence_score


@dataclass
class Chunk:
    chunk_id: str
    filename: str
    document_id: str
    title: str
    status: str            # active | superseded | draft | unknown
    policy_authority: str  # official | none
    audience: str          # customer | internal
    heading: str
    content: str
    precedence_score: float


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """
    Split YAML front matter from Markdown body.

    Returns (frontmatter_dict, body_text).
    Both are returned even if no front matter is present.
    """
    if not text.startswith("---"):
        return {}, text

    end = text.find("---", 3)
    if end == -1:
        return {}, text

    fm_text = text[3:end].strip()
    body = text[end + 3:].strip()
    try:
        fm: dict[str, Any] = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, body


def split_by_headings(body: str, fallback_title: str) -> list[tuple[str, str]]:
    """
    Split a Markdown body into (heading, content) pairs at # / ## boundaries.

    The first section uses ``fallback_title`` as its heading if the body
    does not begin with a heading.
    """
    chunks: list[tuple[str, str]] = []
    current_heading = fallback_title
    current_lines: list[str] = []

    for line in body.splitlines():
        if re.match(r"^#{1,3} ", line):
            content = "\n".join(current_lines).strip()
            if content:
                chunks.append((current_heading, content))
            current_heading = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    content = "\n".join(current_lines).strip()
    if content:
        chunks.append((current_heading, content))

    return chunks


def load_chunks(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> list[Chunk]:
    """Load, parse, and chunk every Markdown file in the knowledge base."""
    chunks: list[Chunk] = []

    for md_file in sorted(kb_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        precedence = compute_precedence_score(fm)
        fallback_title = str(fm.get("title", md_file.stem))

        for i, (heading, content) in enumerate(split_by_headings(body, fallback_title)):
            chunk = Chunk(
                chunk_id=f"{md_file.stem}__{i}",
                filename=md_file.name,
                document_id=str(fm.get("document_id", md_file.stem)),
                title=str(fm.get("title", md_file.stem)),
                status=str(fm.get("status", "unknown")).lower(),
                policy_authority=str(fm.get("policy_authority", "none")).lower(),
                audience=str(fm.get("audience", "customer")).lower(),
                heading=heading,
                content=content,
                precedence_score=precedence,
            )
            chunks.append(chunk)

    return chunks
