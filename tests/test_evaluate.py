"""Unit tests for traccia.eval (builtins + local-only evaluate)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traccia.eval.builtins import run_builtin_scorer
from traccia.eval.evaluate import EvaluateResult, evaluate, _score_span_name
from traccia.eval.errors import EvaluateError


def test_builtin_exact_match():
    assert run_builtin_scorer("exact_match", output=" Hi ", expected="hi")["passed"] is True
    assert run_builtin_scorer("exact_match", output="a", expected="b")["passed"] is False


def test_builtin_contains_and_json():
    assert run_builtin_scorer("contains", output="hello world", expected="world")["passed"]
    assert run_builtin_scorer("json_valid", output='{"a":1}')["passed"]
    assert run_builtin_scorer("json_valid", output="nope")["passed"] is False


def test_evaluate_empty_raises():
    with pytest.raises(EvaluateError):
        evaluate("x", data=[], task=lambda inp: "y", persist=False, progress=False)


def test_builtin_stamps_type_and_scorer_name():
    scored = run_builtin_scorer("contains", output="hello world", expected="world")
    assert scored["type"] == "contains"
    assert scored["scorer_name"] == "contains"
    assert scored["name"] == "contains"


def test_evaluate_local_only():
    with patch("traccia.eval.evaluate._ensure_eval_tracing", return_value=True), patch(
        "traccia.get_tracer"
    ) as gt, patch("traccia.force_flush"):
        span = MagicMock()
        span.context.trace_id = "abc"
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=False)
        tracer = MagicMock()
        tracer.start_as_current_span.return_value = span
        gt.return_value = tracer

        result = evaluate(
            "local-smoke",
            data=[{"input": {"q": "hi"}, "expected": "hi"}],
            task=lambda inp: inp["q"],
            scorers=["exact_match"],
            persist=False,
            progress=False,
            max_concurrency=2,
        )
    assert isinstance(result, EvaluateResult)
    assert result.url is None
    assert result.aggregates["item_count"] == 1
    assert result.aggregates["pass_count"] == 1
    assert "local-smoke" in result.summary()
    assert result.rows[0]["panels"][0].get("trace_id") == "abc"
    panel = result.rows[0]["panels"][0]
    assert panel.get("label") == "Task"
    assert panel.get("source") == "evaluate"
    assert panel["scores"][0].get("type") == "exact_match"
    assert panel["scores"][0].get("scorer_name") == "exact_match"
    assert isinstance(panel.get("latency_ms"), (int, float))
    assert result.aggregates.get("source") == "evaluate"
    span_names = [c.args[0] for c in tracer.start_as_current_span.call_args_list]
    assert "evaluate.item" in span_names
    assert "scorer.exact_match" in span_names


def test_evaluate_skips_trace_id_without_export():
    with patch("traccia.eval.evaluate._ensure_eval_tracing", return_value=False), patch(
        "traccia.get_tracer"
    ) as gt, patch("traccia.force_flush"):
        span = MagicMock()
        span.context.trace_id = "phantom"
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=False)
        tracer = MagicMock()
        tracer.start_as_current_span.return_value = span
        gt.return_value = tracer

        result = evaluate(
            "no-export",
            data=[{"input": {"q": "hi"}, "expected": "hi"}],
            task=lambda inp: inp["q"],
            scorers=["exact_match"],
            persist=False,
            progress=False,
        )
    assert result.rows[0]["panels"][0].get("trace_id") is None


def test_evaluate_error_isolation():
    with patch("traccia.eval.evaluate._ensure_eval_tracing", return_value=True), patch(
        "traccia.get_tracer"
    ) as gt, patch("traccia.force_flush"):
        span = MagicMock()
        span.context.trace_id = "t1"
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=False)
        tracer = MagicMock()
        tracer.start_as_current_span.return_value = span
        gt.return_value = tracer

        def task(inp):
            if inp.get("q") == "boom":
                raise RuntimeError("task failed")
            return inp["q"]

        result = evaluate(
            "iso",
            data=[
                {"input": {"q": "ok"}, "expected": "ok"},
                {"input": {"q": "boom"}, "expected": "x"},
            ],
            task=task,
            scorers=["exact_match"],
            persist=False,
            progress=False,
        )
    assert len(result.rows) == 2
    assert len(result.errors) == 1
    assert result.rows[0]["panels"][0]["passed"] is True
    assert result.rows[1]["panels"][0]["error"]


def test_evaluate_prompt_label_and_expected_output_alias():
    with patch("traccia.eval.evaluate._ensure_eval_tracing", return_value=False), patch(
        "traccia.get_tracer"
    ) as gt, patch("traccia.force_flush"):
        span = MagicMock()
        span.context.trace_id = "t1"
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=False)
        tracer = MagicMock()
        tracer.start_as_current_span.return_value = span
        gt.return_value = tracer

        labeled = evaluate(
            "labeled",
            data=[{"input": {"q": "hi"}, "expected": "hi"}],
            task=lambda inp: inp["q"],
            scorers=["exact_match"],
            prompt="support-reply",
            persist=False,
            progress=False,
        )
        aliased = evaluate(
            "alias",
            data=[{"input": {"q": "hi"}, "expected_output": "hi"}],
            task=lambda inp: inp["q"],
            scorers=["exact_match"],
            persist=False,
            progress=False,
        )
    assert labeled.rows[0]["panels"][0]["label"] == "support-reply"
    assert aliased.aggregates["pass_count"] == 1


def test_score_span_name_and_judge_annotation():
    assert _score_span_name("Helpfulness") == "scorer.Helpfulness"
    span = MagicMock()
    from traccia.eval.evaluate import _annotate_score_span

    _annotate_score_span(
        span,
        {
            "type": "llm_judge",
            "scorer_name": "Helpfulness",
            "passed": True,
            "model": "gemini-3.6-flash",
            "cost_usd": 0.001,
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        },
    )
    attrs = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
    assert attrs["span.type"] == "llm"
    assert attrs["llm.model"] == "gemini-3.6-flash"
    assert attrs["traccia.eval.scorer"] == "Helpfulness"
    assert attrs["llm.usage.total_tokens"] == 14
