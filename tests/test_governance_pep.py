from unittest.mock import MagicMock, patch

import pytest

from traccia.governance.pep import check_policy, enforce_llm_call, enforce_tool_call
from traccia.governance.policy import AgentBlockedError
from traccia.instrumentation.requests import _should_skip_http_instrumentation
from traccia import runtime_config


def test_skips_policy_check_url():
    assert _should_skip_http_instrumentation("https://app.traccia.ai/api/v1/policy/check")
    assert _should_skip_http_instrumentation("http://localhost:8001/api/v1/policy/settle")


def test_enforce_llm_noop_without_pep_flag():
    kwargs = {"model": "gpt-4o"}
    assert enforce_llm_call(kwargs) is None
    assert kwargs["model"] == "gpt-4o"


def test_before_llm_deny_raises_readable_error():
    decision = {
        "id": "dec-1",
        "effect": "deny",
        "would_have": False,
        "reasons": ["spend exceeded cap"],
        "remaining_budget_usd": 0.0,
        "obligations": {},
    }
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                with pytest.raises(AgentBlockedError) as exc:
                    check_policy(action={"type": "llm_call", "model": "gpt-4o"})
                assert "spend exceeded cap" in str(exc.value)
                assert exc.value.decision_id == "dec-1"


def test_before_llm_reshape_swaps_model():
    decision = {
        "id": "dec-2",
        "effect": "reshape",
        "would_have": False,
        "reasons": ["use cheaper model"],
        "obligations": {"cheaper_model": "gpt-4o-mini"},
        "remaining_budget_usd": 1.0,
    }
    kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                out = enforce_llm_call(kwargs)
                assert out["effect"] == "reshape"
                assert kwargs["model"] == "gpt-4o-mini"


def test_before_llm_warn_proceeds():
    decision = {
        "id": "dec-3",
        "effect": "deny",
        "would_have": True,
        "reasons": ["would deny"],
        "obligations": {},
    }
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                assert check_policy(action={"type": "llm_call"})["would_have"] is True


def test_before_tool_calls_check():
    decision = {"id": "dec-4", "effect": "allow", "would_have": False, "reasons": []}
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200, json=lambda: decision)
                out = enforce_tool_call("search", {"q": "x"})
                assert out["effect"] == "allow"
                body = session.post.call_args.kwargs["json"]
                assert body["action"]["type"] == "tool_call"
                assert body["action"]["name"] == "search"


def test_trace_ids_format_ints_as_hex():
    from traccia.governance.pep import _as_otel_hex

    assert _as_otel_hex(1, 32) == ("0" * 31) + "1"
    assert _as_otel_hex("abc", 32) == "abc"
    assert _as_otel_hex(None, 32) is None


def test_check_http_error_allows():
    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=503, text="down")
                out = check_policy(action={"type": "llm_call"})
                assert out["effect"] == "allow"
                assert out["reasons"] == ["check_http_error"]


def test_settle_includes_trace_id():
    from traccia.governance.pep import settle_policy

    with runtime_config.run_identity(agent_id="support", pep_enabled=True):
        with patch("traccia.governance.pep._credentials", return_value=("key", "https://app.traccia.ai/v1/traces")):
            with patch("traccia.governance.pep._http_session") as session:
                session.post.return_value = MagicMock(status_code=200)
                settle_policy({"obligations": {"reserved_usd": 0.25}}, actual_usd=0.1)
                body = session.post.call_args.kwargs["json"]
                assert body["agent_id"] == "support"
                assert body["reserved_usd"] == 0.25
                assert body["actual_usd"] == 0.1
                assert "trace_id" in body

