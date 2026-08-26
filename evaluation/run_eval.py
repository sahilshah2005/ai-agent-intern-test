"""
run_eval.py — evaluation runner for the Aster & Row support agent.

Usage:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --cases visible          # visible only
    python evaluation/run_eval.py --cases original         # original only
    python evaluation/run_eval.py --cases all              # both (default)
    python evaluation/run_eval.py --verbose                # show full responses

Exit code: 0 if all assertions pass, 1 if any fail.

Assertion strategy
──────────────────
Assertions are deterministic wherever possible:
  • must_include          — case-insensitive substring in response
  • must_not_include      — case-insensitive substring absent from response
  • must_include_concepts — key noun/phrase present (keyword matching)
  • must_not_follow       — negative concept check
  • must_not_invent       — concept absent (alias for must_not_include)
  • must_ask_for          — response includes a question containing the term
  • required_sources      — filename in result["retrieved"] active+official chunks
  • forbidden_sources_as_authority — filename NOT in result["sources"] (cited)
  • tool                  — whether order_lookup was called
  • tool_arguments        — normalised order_id matches
  • handoff               — handoff flag true/false
  • must_refuse_to_disclose — response refuses to share the item
  • must_not_silently_choose_one — response contains conflict language

Does not rely exclusively on an LLM judge.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent import SupportAgent
from order_tool import normalize_order_id

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).parent
VISIBLE_CASES_FILE = EVAL_DIR / "visible-cases.json"
ORIGINAL_CASES_FILE = EVAL_DIR / "original-cases.json"


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _contains(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def _concept_present(text: str, concept: str) -> bool:
    """
    Check whether a concept is expressed in the text.
    Strategy: all significant words (len >= 4) in the concept must appear.
    """
    lower_text = text.lower()
    lower_concept = concept.lower()

    # Direct substring check first
    if lower_concept in lower_text:
        return True

    # Fallback: check that key terms are all present
    significant = [w for w in lower_concept.split() if len(w) >= 4]
    if not significant:
        return lower_concept in lower_text
    return all(w in lower_text for w in significant)


def _assert_must_include(response: str, items: list[str]) -> list[str]:
    failures = []
    for item in items:
        if not _contains(response, item):
            failures.append(f"must_include: '{item}' not found in response")
    return failures


def _assert_must_not_include(response: str, items: list[str]) -> list[str]:
    failures = []
    for item in items:
        if _contains(response, item):
            failures.append(f"must_not_include: '{item}' was found in response")
    return failures


def _assert_must_include_concepts(response: str, concepts: list[str]) -> list[str]:
    failures = []
    for concept in concepts:
        if not _concept_present(response, concept):
            failures.append(f"must_include_concepts: '{concept}' not found")
    return failures


def _assert_must_not_follow(response: str, items: list[str]) -> list[str]:
    """Same as must_not_include but named differently in the spec."""
    return _assert_must_not_include(response, items)


def _assert_must_not_invent(response: str, items: list[str]) -> list[str]:
    return _assert_must_not_include(response, items)


def _assert_must_ask_for(response: str, items: list[str]) -> list[str]:
    """Response should contain a question that includes the term."""
    failures = []
    for item in items:
        # Check the response asks for the item (contains both the term and '?')
        if not (_contains(response, item) and "?" in response):
            failures.append(f"must_ask_for: response did not ask for '{item}'")
    return failures


def _assert_required_sources(
    retrieved: list[dict], sources: list[str]
) -> list[str]:
    """Check that required filenames appear in the retrieved+authoritative results."""
    failures = []
    retrieved_active_official = {
        r["filename"]
        for r in retrieved
        if r.get("status") == "active" and r.get("policy_authority") == "official"
    }
    for src in sources:
        if src not in retrieved_active_official:
            failures.append(
                f"required_sources: '{src}' not in retrieved active+official chunks"
            )
    return failures


def _assert_forbidden_sources_as_authority(
    cited_sources: list[dict], forbidden: list[str]
) -> list[str]:
    """Check that forbidden filenames do NOT appear in the cited sources."""
    failures = []
    cited_filenames = {s["filename"] for s in cited_sources}
    for src in forbidden:
        if src in cited_filenames:
            failures.append(
                f"forbidden_sources_as_authority: '{src}' was cited as authority"
            )
    return failures


def _assert_tool(
    tool_calls: list[dict],
    tool_spec: str | None,
    response: str,
) -> list[str]:
    if tool_spec is None:
        return []

    called = any(tc["tool"] == "order_lookup" for tc in tool_calls)

    if tool_spec == "not_called":
        if called:
            return ["tool: order_lookup was called but must_not be"]
    elif tool_spec == "order_lookup":
        if not called:
            return ["tool: order_lookup was not called but should have been"]
    elif tool_spec == "not_called_without_id":
        # If called, the response should have asked for an ID first — covered
        # by must_ask_for; just verify tool wasn't called with an invented ID
        if called:
            for tc in tool_calls:
                if tc["tool"] == "order_lookup":
                    oid = tc["arguments"].get("order_id", "")
                    if not oid or oid.lower() in ("unknown", "none", ""):
                        return ["tool: called without a real order_id"]
    elif tool_spec == "optional_sanitized_lookup":
        pass  # no assertion on whether it was called

    return []


def _assert_tool_arguments(
    tool_calls: list[dict], expected_args: dict
) -> list[str]:
    if not expected_args:
        return []

    for tc in tool_calls:
        if tc["tool"] == "order_lookup":
            actual_id = normalize_order_id(tc["arguments"].get("order_id", ""))
            expected_id = normalize_order_id(expected_args.get("order_id", ""))
            if actual_id != expected_id:
                return [
                    f"tool_arguments: expected order_id={expected_id}, "
                    f"got order_id={actual_id}"
                ]
            return []
    return []


def _assert_handoff(handoff: bool, expected: bool | None) -> list[str]:
    if expected is None:
        return []
    if handoff != expected:
        return [
            f"handoff: expected {expected}, got {handoff}"
        ]
    return []


def _assert_must_refuse(response: str, items: list[str]) -> list[str]:
    """Check that the response refuses to disclose each item."""
    failures = []
    refusal_words = {"cannot", "unable", "don't", "do not", "won't", "not able", "decline", "not share", "not disclose"}
    lower = response.lower()
    has_refusal = any(w in lower for w in refusal_words)
    if not has_refusal:
        failures.append("must_refuse_to_disclose: no refusal language found")
    return failures


def _assert_no_silent_choice(response: str, required: bool) -> list[str]:
    if not required:
        return []
    conflict_words = ["conflict", "contradict", "inconsistent", "differ", "one says", "another says"]
    lower = response.lower()
    if not any(w in lower for w in conflict_words):
        return ["must_not_silently_choose_one: no conflict language found in response"]
    return []


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------


def run_case(agent: SupportAgent, case: dict, verbose: bool = False) -> dict:
    """
    Run a single evaluation case through the agent.

    Returns a result dict:
        {
            "id": str,
            "category": str,
            "passed": bool,
            "failures": list[str],
            "response": str,         # last response
            "sources": list[dict],
            "retrieved": list[dict],
            "tool_calls": list[dict],
            "handoff": bool,
        }
    """
    agent.reset()  # fresh session per case
    messages = case.get("messages", [])
    expect = case.get("expect", {})

    last_result: dict[str, Any] = {
        "response": "",
        "sources": [],
        "retrieved": [],
        "tool_calls": [],
        "handoff": False,
    }

    # Run each user turn in sequence (same session)
    for msg in messages:
        if msg.get("role") == "user":
            last_result = agent.chat(msg["content"])

    response = last_result["response"]
    sources = last_result["sources"]
    retrieved = last_result["retrieved"]
    tool_calls = last_result["tool_calls"]
    handoff = last_result["handoff"]

    if verbose:
        print(f"\n  Response: {response[:300]}{'...' if len(response)>300 else ''}")

    # Run assertions
    failures: list[str] = []

    if "must_include" in expect:
        failures += _assert_must_include(response, expect["must_include"])

    if "must_not_include" in expect:
        failures += _assert_must_not_include(response, expect["must_not_include"])

    if "must_include_concepts" in expect:
        failures += _assert_must_include_concepts(response, expect["must_include_concepts"])

    if "must_not_follow" in expect:
        failures += _assert_must_not_follow(response, expect["must_not_follow"])

    if "must_not_invent" in expect:
        failures += _assert_must_not_invent(response, expect["must_not_invent"])

    if "must_ask_for" in expect:
        failures += _assert_must_ask_for(response, expect["must_ask_for"])

    if "required_sources" in expect:
        failures += _assert_required_sources(retrieved, expect["required_sources"])

    if "forbidden_sources_as_authority" in expect:
        failures += _assert_forbidden_sources_as_authority(sources, expect["forbidden_sources_as_authority"])

    if "tool" in expect:
        failures += _assert_tool(tool_calls, expect["tool"], response)

    if "tool_arguments" in expect:
        failures += _assert_tool_arguments(tool_calls, expect["tool_arguments"])

    if "handoff" in expect:
        failures += _assert_handoff(handoff, expect["handoff"])

    if "must_refuse_to_disclose" in expect:
        failures += _assert_must_refuse(response, expect["must_refuse_to_disclose"])
        # Also check that the actual sensitive values aren't in the response
        if "must_not_include" in expect:
            pass  # already checked above

    if "must_not_silently_choose_one" in expect:
        failures += _assert_no_silent_choice(response, expect["must_not_silently_choose_one"])

    return {
        "id": case["id"],
        "category": case.get("category", "unknown"),
        "passed": len(failures) == 0,
        "failures": failures,
        "response": response,
        "sources": sources,
        "retrieved": retrieved,
        "tool_calls": tool_calls,
        "handoff": handoff,
    }


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

PASS = "✅ PASS"
FAIL = "❌ FAIL"
_W = 50


def _print_header(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def _print_result(r: dict, verbose: bool) -> None:
    icon = PASS if r["passed"] else FAIL
    print(f"{icon}  [{r['category']:20s}]  {r['id']}")
    if not r["passed"]:
        for f in r["failures"]:
            print(f"          ✗ {f}")
    if verbose and r["sources"]:
        print(f"       Sources: {[s['filename'] for s in r['sources']]}")


def _print_summary(results: list[dict], suite_name: str) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    by_cat: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"pass": 0, "fail": 0}
        if r["passed"]:
            by_cat[cat]["pass"] += 1
        else:
            by_cat[cat]["fail"] += 1

    print(f"\n{'─' * 60}")
    print(f"  {suite_name} Results: {passed}/{total} passed")
    print(f"{'─' * 60}")
    for cat, counts in sorted(by_cat.items()):
        p, f = counts["pass"], counts["fail"]
        status = "✅" if f == 0 else "❌"
        print(f"  {status} {cat:30s}  {p}/{p+f}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Aster & Row evaluation runner")
    parser.add_argument(
        "--cases",
        choices=["visible", "original", "all"],
        default="all",
        help="Which case files to run (default: all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show response excerpts")
    args = parser.parse_args()

    # Load case files
    suites: list[tuple[str, list[dict]]] = []
    if args.cases in ("visible", "all"):
        data = json.loads(VISIBLE_CASES_FILE.read_text(encoding="utf-8"))
        suites.append(("Visible Cases", data["cases"]))
    if args.cases in ("original", "all"):
        data = json.loads(ORIGINAL_CASES_FILE.read_text(encoding="utf-8"))
        suites.append(("Original Cases", data["cases"]))

    total_cases = sum(len(s[1]) for s in suites)
    print(f"\nRunning {total_cases} evaluation cases against the Aster & Row agent...")
    print("(This requires an OPENAI_API_KEY in your environment.)\n")

    try:
        agent = SupportAgent()
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    all_results: list[dict] = []

    for suite_name, cases in suites:
        _print_header(suite_name)
        suite_results: list[dict] = []

        for case in cases:
            print(f"  Running: {case['id']} ...", end="", flush=True)
            try:
                result = run_case(agent, case, verbose=args.verbose)
            except Exception as exc:  # noqa: BLE001
                result = {
                    "id": case["id"],
                    "category": case.get("category", "unknown"),
                    "passed": False,
                    "failures": [f"Exception: {exc}"],
                    "response": "",
                    "sources": [],
                    "retrieved": [],
                    "tool_calls": [],
                    "handoff": False,
                }
            print(f"\r", end="")
            _print_result(result, args.verbose)
            suite_results.append(result)
            all_results.append(result)

        _print_summary(suite_results, suite_name)

    # Overall summary
    if len(suites) > 1:
        _print_header("Overall Summary")
        total = len(all_results)
        passed = sum(1 for r in all_results if r["passed"])
        print(f"  Total: {passed}/{total} cases passed\n")

    any_failed = any(not r["passed"] for r in all_results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
