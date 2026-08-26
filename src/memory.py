"""
memory.py — lightweight per-session conversation memory.

Stores a sliding window of (role, content) turns.
Session state is in-memory only; no cross-session persistence.
Sensitive fields are never stored here; only what the user/agent said.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str


class ConversationMemory:
    """
    Maintains a sliding window of conversation turns.

    Only stores the clean user message and the assistant response —
    not the augmented knowledge-context injections, which are
    re-computed fresh on every turn.
    """

    def __init__(self, max_turns: int = 8) -> None:
        self._max_turns = max_turns          # (user + assistant) pairs
        self._turns: list[Turn] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, role: Literal["user", "assistant"], content: str) -> None:
        """Append a turn and enforce the sliding-window limit."""
        self._turns.append(Turn(role=role, content=content))
        max_messages = self._max_turns * 2   # each turn = 1 user + 1 assistant
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def clear(self) -> None:
        """Reset the session (new conversation)."""
        self._turns.clear()

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def as_messages(self) -> list[dict[str, str]]:
        """Return turns in the OpenAI messages format."""
        return [{"role": t.role, "content": t.content} for t in self._turns]

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        return not self._turns

    def last_user_message(self) -> str | None:
        """Return the most recent user message content, if any."""
        for t in reversed(self._turns):
            if t.role == "user":
                return t.content
        return None
