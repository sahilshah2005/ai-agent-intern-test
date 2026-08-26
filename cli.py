"""
cli.py — command-line interface for the Aster & Row support agent.

Usage:
    python cli.py               # normal mode
    python cli.py --debug       # debug mode (structured traces on stderr)
    python cli.py --rebuild     # rebuild the knowledge-base index, then start

Special commands in the chat loop:
    reset    — start a new conversation session
    quit     — exit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is on the path regardless of working directory
sys.path.insert(0, str(Path(__file__).parent / "src"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aster & Row AI Support Agent")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug traces on stderr.",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the knowledge-base index before starting.",
    )
    return p.parse_args()


_BANNER = """
╔══════════════════════════════════════════════════════╗
║         Aster & Row — AI Customer Support            ║
║  Type your question. Commands: reset | quit          ║
╚══════════════════════════════════════════════════════╝
"""

_SEP = "─" * 60


def _format_result(result: dict) -> str:
    """Format a structured agent result for terminal display."""
    lines: list[str] = [_SEP, result["response"]]

    if result["sources"]:
        lines.append("\n📎 Sources:")
        for s in result["sources"]:
            lines.append(f"   • {s['filename']} — {s['heading']}")

    if result["handoff"]:
        lines.append("\n⚠️  Human handoff recommended.")

    lines.append(_SEP)
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()

    if args.debug:
        os.environ["DEBUG"] = "true"

    # Late imports so DEBUG env var is set before config is evaluated
    from agent import SupportAgent

    print(_BANNER)

    if args.debug:
        print("[Debug mode ON — JSON traces on stderr]\n")

    try:
        agent = SupportAgent(rebuild_index=args.rebuild)
    except EnvironmentError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Agent ready. Ask anything about Aster & Row.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("Session reset. Starting a new conversation.\n")
            continue

        try:
            result = agent.chat(user_input)
            print(f"\nAgent:\n{_format_result(result)}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}", file=sys.stderr)
            if args.debug:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
