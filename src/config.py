"""
config.py — centralised configuration for the Aster & Row support agent.

Every tunable constant lives here. Nothing outside this file reads
environment variables or uses raw magic numbers.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR: Path = ROOT_DIR / "knowledge-base"
ORDERS_FILE: Path = ROOT_DIR / "data" / "orders.json"
CHROMA_PERSIST_DIR: Path = ROOT_DIR / ".chroma"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Embeddings & retrieval
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # local, ~90 MB, no API key needed
TOP_K: int = 5                                # final chunks sent to model
TOP_K_CANDIDATES: int = 15                   # over-fetch before re-ranking

# Metadata-based score adjustments (added to / subtracted from cosine similarity)
AUTHORITY_BONUS: float = 0.25   # active + official + customer-facing
LEGACY_PENALTY: float = 0.35    # status == superseded
INTERNAL_PENALTY: float = 0.20  # audience == internal
DRAFT_PENALTY: float = 0.30     # status == draft
NO_AUTHORITY_PENALTY: float = 0.25  # policy_authority == none

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
MAX_HISTORY_TURNS: int = 8   # (user + assistant) pairs kept per session

# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------
def require_openai_key() -> str:
    """Return the API key or raise a helpful error."""
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key:\n"
            "  cp .env.example .env\n"
            "  # then edit .env"
        )
    return OPENAI_API_KEY
