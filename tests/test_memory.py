"""
test_memory.py — unit tests for ConversationMemory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import ConversationMemory


class TestConversationMemory:
    def test_empty_on_creation(self):
        mem = ConversationMemory()
        assert mem.is_empty
        assert mem.turn_count == 0
        assert mem.as_messages() == []

    def test_add_user_and_assistant(self):
        mem = ConversationMemory()
        mem.add("user", "Hello")
        mem.add("assistant", "Hi!")
        messages = mem.as_messages()
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi!"}

    def test_sliding_window_enforced(self):
        mem = ConversationMemory(max_turns=2)
        # Add 3 pairs (6 messages) → should keep only 2 pairs (4 messages)
        for i in range(3):
            mem.add("user", f"User {i}")
            mem.add("assistant", f"Agent {i}")
        messages = mem.as_messages()
        assert len(messages) == 4  # 2 * 2
        assert messages[0]["content"] == "User 1"
        assert messages[-1]["content"] == "Agent 2"

    def test_clear_resets_memory(self):
        mem = ConversationMemory()
        mem.add("user", "test")
        mem.add("assistant", "reply")
        mem.clear()
        assert mem.is_empty
        assert mem.as_messages() == []

    def test_last_user_message(self):
        mem = ConversationMemory()
        mem.add("user", "First question")
        mem.add("assistant", "First answer")
        mem.add("user", "Second question")
        assert mem.last_user_message() == "Second question"

    def test_last_user_message_none_when_empty(self):
        mem = ConversationMemory()
        assert mem.last_user_message() is None

    def test_messages_in_order(self):
        mem = ConversationMemory()
        messages = ["one", "two", "three", "four"]
        roles = ["user", "assistant", "user", "assistant"]
        for role, content in zip(roles, messages):
            mem.add(role, content)
        result = mem.as_messages()
        for i, (role, content) in enumerate(zip(roles, messages)):
            assert result[i]["role"] == role
            assert result[i]["content"] == content

    def test_does_not_modify_source_list(self):
        mem = ConversationMemory()
        mem.add("user", "Hello")
        msgs = mem.as_messages()
        msgs.append({"role": "user", "content": "extra"})
        # Internal list should not be affected
        assert mem.turn_count == 1
