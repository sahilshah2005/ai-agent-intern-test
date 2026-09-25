"""Offline checks for end-to-end evaluation accounting and report output."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_runner():
    # The report logic needs no API key or model downloads.
    agent_stub = types.ModuleType("agent")
    agent_stub.SupportAgent = object
    order_stub = types.ModuleType("order_tool")
    order_stub.normalize_order_id = lambda value: value
    path = Path(__file__).resolve().parents[1] / "evaluation" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("eval_runner_reporting", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"agent": agent_stub, "order_tool": order_stub}):
        spec.loader.exec_module(module)
    return module


class FakeAgent:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def chat(self, _message):
        return next(self.responses)


def response(text, prompt, completion, calls):
    return {"response": text, "sources": [], "retrieved": [],
            "tool_calls": [], "handoff": False,
            "token_usage": {"prompt_tokens": prompt,
                            "completion_tokens": completion,
                            "api_calls": calls, "unreported_calls": 0}}


class EvaluationReportingTests(unittest.TestCase):
    def test_multi_turn_usage_and_case_assertions(self):
        runner = load_runner()
        agent = FakeAgent([response("first", 10, 2, 1),
                           response("final", 25, 6, 2)])
        case = {"id": "multi", "category": "conversation",
                "messages": [{"role": "user", "content": "one"},
                             {"role": "user", "content": "two"}],
                "expect": {"must_include": ["final"]}}
        result = runner.run_case(agent, case)
        self.assertTrue(result["passed"])
        self.assertEqual(agent.reset_count, 1)
        self.assertEqual(result["token_usage"], {
            "prompt_tokens": 35, "completion_tokens": 8,
            "api_calls": 3, "unreported_calls": 0})
        self.assertEqual(len(result["turn_latencies_seconds"]), 2)
        report = runner.build_report([result], 1.0, 2.0)
        self.assertEqual(report["cases"]["pass_rate"], 1.0)
        self.assertEqual(report["estimated_llm_cost_usd"], 0.000051)
        self.assertEqual(report["latency_seconds_per_turn"]["count"], 2)

    def test_failures_are_listed_without_customer_text(self):
        runner = load_runner()
        agent = FakeAgent([response("private response text", 4, 2, 1)])
        case = {"id": "bad", "category": "groundedness",
                "messages": [{"role": "user", "content": "private prompt text"}],
                "expect": {"must_include": ["missing"]}}
        result = runner.run_case(agent, case)
        report = runner.build_report([result])
        self.assertFalse(result["passed"])
        self.assertEqual(report["cases"]["passed"], 0)
        self.assertEqual(report["failures"][0]["id"], "bad")
        self.assertNotIn("private response text", str(report))
        self.assertNotIn("private prompt text", str(report))

    def test_cost_is_unknown_if_usage_is_missing(self):
        runner = load_runner()
        result = {"id": "partial", "category": "tool", "passed": True,
                  "failures": [], "turn_latencies_seconds": [0.2],
                  "token_usage": {"api_calls": 2, "unreported_calls": 1,
                                  "prompt_tokens": 100, "completion_tokens": 20}}
        report = runner.build_report([result], 0.5, 2.0)
        self.assertIsNone(report["estimated_llm_cost_usd"])


if __name__ == "__main__":
    unittest.main()
