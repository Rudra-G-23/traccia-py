"""Tests for the GitHub Copilot hooks integration.

Covers: mapping (event -> span shape), the local session-log state store,
end-to-end span materialization (parent/child hierarchy, timing, redaction),
the hook subprocess entry point (must never exit non-zero -- see
docs/github-copilot-integration.md Section 6), and flush orchestration.
"""

from __future__ import annotations

import io
import json
import time
from unittest.mock import patch

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traccia.integrations.github_copilot import flush as flush_mod
from traccia.integrations.github_copilot import hook as hook_mod
from traccia.integrations.github_copilot import install as install_fn
from traccia.integrations.github_copilot import mapping
from traccia.integrations.github_copilot import spans as spans_mod
from traccia.integrations.github_copilot import state
from traccia.tracer.provider import TracerProvider


# ---------------------------------------------------------------------------
# mapping.py
# ---------------------------------------------------------------------------


def test_strip_content_fields_default_strips_to_length_only():
    payload = {"sessionId": "s1", "toolName": "shell", "toolArgs": {"cmd": "ls -la /secret"}}
    out = mapping.strip_content_fields("preToolUse", payload, capture_content=False)
    assert out["toolArgs"] == {"_stripped": True, "length": len(json.dumps(payload["toolArgs"]))}
    assert "secret" not in json.dumps(out)


def test_strip_content_fields_capture_content_keeps_capped_text():
    payload = {"sessionId": "s1", "toolName": "shell", "toolArgs": {"cmd": "ls -la"}}
    out = mapping.strip_content_fields("preToolUse", payload, capture_content=True)
    assert isinstance(out["toolArgs"], str)
    assert "ls -la" in out["toolArgs"]


def test_strip_content_fields_caps_error_message_length():
    payload = {"sessionId": "s1", "error": {"message": "x" * 5000}}
    out = mapping.strip_content_fields("postToolUseFailure", payload, capture_content=False)
    assert len(out["error"]["message"]) == mapping._MAX_ERROR_CHARS


def test_strip_content_fields_missing_field_is_noop():
    payload = {"sessionId": "s1", "toolName": "shell"}  # no toolArgs at all
    out = mapping.strip_content_fields("preToolUse", payload, capture_content=False)
    assert "toolArgs" not in out


def test_span_name_for_tool_and_subagent_and_session():
    assert mapping.span_name_for("sessionStart", {}) == "github_copilot.session"
    assert mapping.span_name_for("preToolUse", {"toolName": "shell"}) == "github_copilot.tool.shell"
    assert mapping.span_name_for("postToolUse", {"toolName": "shell"}) == "github_copilot.tool.shell"
    assert (
        mapping.span_name_for("subagentStart", {"agentName": "reviewer"})
        == "github_copilot.subagent.reviewer"
    )
    assert mapping.span_name_for("preCompact", {}) == "github_copilot.preCompact"


def test_start_attributes_session_stripped_prompt():
    payload = {
        "sessionId": "s1",
        "cwd": "/repo",
        "source": "new",
        "initialPrompt": {"_stripped": True, "length": 42},
    }
    attrs = mapping.start_attributes("sessionStart", payload)
    assert attrs["session.id"] == "s1"
    assert attrs["gen_ai.system"] == "github_copilot"
    assert attrs["github_copilot.prompt.length"] == 42
    assert "github_copilot.prompt.preview" not in attrs


def test_end_attributes_post_tool_use_failure_marks_error():
    result = mapping.end_attributes(
        "postToolUseFailure", {"error": {"message": "boom"}}
    )
    assert result["is_error"] is True
    assert result["error_message"] == "boom"
    assert result["attributes"]["error.message"] == "boom"


def test_end_attributes_post_tool_use_success():
    result = mapping.end_attributes(
        "postToolUse",
        {"toolResult": {"resultType": "success", "textResultForLlm": "ok"}},
    )
    assert result["is_error"] is False
    assert result["attributes"]["agent.tool.output"] == "ok"


# ---------------------------------------------------------------------------
# state.py
# ---------------------------------------------------------------------------


def test_state_append_read_clear_roundtrip(tmp_path):
    state.append_event("sess-1", "sessionStart", {"sessionId": "sess-1"}, state_dir=tmp_path)
    state.append_event("sess-1", "preToolUse", {"sessionId": "sess-1", "toolName": "shell"}, state_dir=tmp_path)

    events = state.read_events("sess-1", state_dir=tmp_path)
    assert [e["event"] for e in events] == ["sessionStart", "preToolUse"]

    assert state.list_sessions(state_dir=tmp_path) == ["sess-1"]

    state.clear_session("sess-1", state_dir=tmp_path)
    assert state.read_events("sess-1", state_dir=tmp_path) == []
    assert state.list_sessions(state_dir=tmp_path) == []


def test_state_read_events_missing_session_returns_empty(tmp_path):
    assert state.read_events("does-not-exist", state_dir=tmp_path) == []


def test_state_tolerates_partially_written_last_line(tmp_path):
    path = state.session_log_path("sess-1", state_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"event": "sessionStart", "payload": {}, "received_at": 1.0}\n{"event": "preTool', encoding="utf-8")
    events = state.read_events("sess-1", state_dir=tmp_path)
    assert len(events) == 1


def test_state_sanitizes_session_id_for_filesystem(tmp_path):
    # A path-separator-bearing session id must never escape state_dir -- the
    # sanitizer strips separators, so even a ".." in the id can't traverse
    # (no separator survives to make it a distinct path segment).
    malicious = "../../etc/passwd"
    state.append_event(malicious, "sessionStart", {}, state_dir=tmp_path)
    path = state.session_log_path(malicious, state_dir=tmp_path)
    assert path.parent == tmp_path
    assert "/" not in path.name and "\\" not in path.name


def test_state_clear_missing_session_does_not_raise(tmp_path):
    state.clear_session("never-existed", state_dir=tmp_path)  # should not raise


# ---------------------------------------------------------------------------
# spans.py -- end-to-end materialization against a real in-memory OTel exporter
# ---------------------------------------------------------------------------


def _make_tracer():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("github_copilot"), exporter


def _events(*items):
    """items: list of (event_name, payload, offset_seconds)."""
    base = time.time()
    return [
        {"event": name, "payload": payload, "received_at": base + offset}
        for name, payload, offset in items
    ]


def test_build_trace_empty_events_returns_none():
    tracer, _ = _make_tracer()
    assert spans_mod.build_trace(tracer, []) is None


def test_build_trace_session_and_tool_hierarchy():
    tracer, exporter = _make_tracer()
    events = _events(
        ("sessionStart", {"sessionId": "s1", "cwd": "/repo", "source": "new"}, 0),
        ("preToolUse", {"sessionId": "s1", "toolName": "shell", "toolArgs": {"cmd": "ls"}}, 1),
        (
            "postToolUse",
            {
                "sessionId": "s1",
                "toolName": "shell",
                "toolResult": {"resultType": "success", "textResultForLlm": "file.txt"},
            },
            3,
        ),
        ("sessionEnd", {"sessionId": "s1", "reason": "complete"}, 4),
    )
    summary = spans_mod.build_trace(tracer, events)
    assert summary == {"tool_spans": 1, "subagent_spans": 0, "errors": 0}

    finished = exporter.get_finished_spans()
    by_name = {s.name: s for s in finished}
    assert "github_copilot.session" in by_name
    assert "github_copilot.tool.shell" in by_name

    session_span = by_name["github_copilot.session"]
    tool_span = by_name["github_copilot.tool.shell"]

    # Real parent/child relationship, not just matching trace ids.
    assert tool_span.parent.span_id == session_span.context.span_id
    assert tool_span.context.trace_id == session_span.context.trace_id

    # Real durations reconstructed from event timestamps, not materialization time.
    tool_duration_s = (tool_span.end_time - tool_span.start_time) / 1e9
    assert 1.5 < tool_duration_s < 2.5  # postToolUse at +3s, preToolUse at +1s

    assert tool_span.attributes["agent.tool.name"] == "shell"
    assert tool_span.attributes["agent.tool.output"] == "file.txt"


def test_build_trace_post_tool_use_failure_marks_error_status():
    from opentelemetry.trace import StatusCode

    tracer, exporter = _make_tracer()
    events = _events(
        ("sessionStart", {"sessionId": "s1"}, 0),
        ("preToolUse", {"sessionId": "s1", "toolName": "shell"}, 1),
        ("postToolUseFailure", {"sessionId": "s1", "toolName": "shell", "error": {"message": "boom"}}, 2),
    )
    spans_mod.build_trace(tracer, events)
    finished = exporter.get_finished_spans()
    tool_span = next(s for s in finished if s.name == "github_copilot.tool.shell")
    assert tool_span.status.status_code == StatusCode.ERROR


def test_build_trace_subagent_hierarchy():
    tracer, exporter = _make_tracer()
    events = _events(
        ("sessionStart", {"sessionId": "s1"}, 0),
        ("subagentStart", {"sessionId": "s1", "agentId": "a1", "agentName": "reviewer"}, 1),
        ("subagentStop", {"sessionId": "s1", "agentId": "a1", "response": {"_stripped": True, "length": 10}}, 2),
    )
    summary = spans_mod.build_trace(tracer, events)
    assert summary["subagent_spans"] == 1

    finished = exporter.get_finished_spans()
    by_name = {s.name: s for s in finished}
    subagent_span = by_name["github_copilot.subagent.reviewer"]
    session_span = by_name["github_copilot.session"]
    assert subagent_span.parent.span_id == session_span.context.span_id
    assert subagent_span.attributes["agent.response.length"] == 10


def test_build_trace_error_occurred_attaches_to_innermost_open_span():
    tracer, exporter = _make_tracer()
    events = _events(
        ("sessionStart", {"sessionId": "s1"}, 0),
        ("preToolUse", {"sessionId": "s1", "toolName": "shell"}, 1),
        ("errorOccurred", {"sessionId": "s1", "error": {"message": "kaboom"}}, 1.5),
        ("postToolUse", {"sessionId": "s1", "toolName": "shell", "toolResult": {"textResultForLlm": "ok"}}, 2),
    )
    summary = spans_mod.build_trace(tracer, events)
    assert summary["errors"] == 1

    finished = exporter.get_finished_spans()
    tool_span = next(s for s in finished if s.name == "github_copilot.tool.shell")
    event_names = [e.name for e in tool_span.events]
    assert "github_copilot.error" in event_names


def test_build_trace_error_occurred_message_is_redacted():
    tracer, exporter = _make_tracer()
    events = _events(
        ("sessionStart", {"sessionId": "s1"}, 0),
        ("errorOccurred", {"sessionId": "s1", "error": {"message": "contact bob@example.com for quota"}}, 1),
        ("sessionEnd", {"sessionId": "s1"}, 2),
    )
    spans_mod.build_trace(tracer, events)
    finished = exporter.get_finished_spans()
    session_span = next(s for s in finished if s.name == "github_copilot.session")
    error_events = [e for e in session_span.events if e.name == "github_copilot.error"]
    assert len(error_events) == 1
    message = error_events[0].attributes["error.message"]
    assert "bob@example.com" not in message
    assert "[REDACTED_EMAIL]" in message


def test_build_trace_missing_session_start_still_produces_hierarchy():
    """Hooks were enabled mid-session: tool events arrive with no sessionStart."""
    tracer, exporter = _make_tracer()
    events = _events(
        ("preToolUse", {"sessionId": "s1", "toolName": "shell"}, 0),
        ("postToolUse", {"sessionId": "s1", "toolName": "shell", "toolResult": {"textResultForLlm": "ok"}}, 1),
    )
    spans_mod.build_trace(tracer, events)
    finished = exporter.get_finished_spans()
    by_name = {s.name: s for s in finished}
    assert "github_copilot.session" in by_name
    assert by_name["github_copilot.session"].attributes["github_copilot.session.source"] == (
        "recovered_missing_session_start"
    )
    assert by_name["github_copilot.tool.shell"].parent.span_id == by_name["github_copilot.session"].context.span_id


def test_build_trace_post_tool_use_without_matching_pre_tool_use():
    tracer, exporter = _make_tracer()
    events = _events(
        ("sessionStart", {"sessionId": "s1"}, 0),
        ("postToolUse", {"sessionId": "s1", "toolName": "shell", "toolResult": {"textResultForLlm": "ok"}}, 1),
    )
    summary = spans_mod.build_trace(tracer, events)
    assert summary["tool_spans"] == 0  # never opened via preToolUse
    finished = exporter.get_finished_spans()
    assert any(s.name == "github_copilot.tool.shell" for s in finished)


def test_build_trace_dangling_spans_closed_when_session_end_missing():
    """sessionEnd never arrived (process was killed) -- everything still closes."""
    tracer, exporter = _make_tracer()
    events = _events(
        ("sessionStart", {"sessionId": "s1"}, 0),
        ("preToolUse", {"sessionId": "s1", "toolName": "shell"}, 1),
        # no postToolUse, no sessionEnd
    )
    spans_mod.build_trace(tracer, events)
    finished = exporter.get_finished_spans()
    assert len(finished) == 2
    assert all(s.end_time is not None for s in finished)


def test_build_trace_content_capture_still_redacts_pii():
    tracer, exporter = _make_tracer()
    events = _events(
        ("sessionStart", {"sessionId": "s1"}, 0),
        (
            "preToolUse",
            {
                "sessionId": "s1",
                "toolName": "email_tool",
                "toolArgs": "contact bob@example.com about this",
            },
            1,
        ),
        ("postToolUse", {"sessionId": "s1", "toolName": "email_tool", "toolResult": {"textResultForLlm": "sent"}}, 2),
    )
    spans_mod.build_trace(tracer, events)
    finished = exporter.get_finished_spans()
    tool_span = next(s for s in finished if s.name == "github_copilot.tool.email_tool")
    assert "bob@example.com" not in tool_span.attributes["agent.tool.input"]
    assert "[REDACTED_EMAIL]" in tool_span.attributes["agent.tool.input"]


def test_build_trace_ignores_unmapped_events_gracefully():
    tracer, exporter = _make_tracer()
    events = _events(
        ("sessionStart", {"sessionId": "s1"}, 0),
        ("someBrandNewEventGitHubAddsLater", {"sessionId": "s1"}, 1),
        ("sessionEnd", {"sessionId": "s1"}, 2),
    )
    summary = spans_mod.build_trace(tracer, events)
    assert summary is not None
    finished = exporter.get_finished_spans()
    assert len(finished) == 1  # just the session span


# ---------------------------------------------------------------------------
# hook.py -- must never exit non-zero, must always print valid JSON
# ---------------------------------------------------------------------------


def _run_hook(argv, stdin_text, state_dir):
    with patch("sys.stdin", io.StringIO(stdin_text)), patch("sys.stdout", new_callable=io.StringIO) as out:
        with patch(
            "traccia.integrations.github_copilot.state.default_state_dir",
            return_value=state_dir,
        ):
            rc = hook_mod.main(["hook.py"] + argv)
    return rc, out.getvalue()


def test_hook_main_malformed_json_still_exits_zero(tmp_path):
    rc, out = _run_hook(["preToolUse"], "not json{{{", tmp_path)
    assert rc == 0
    assert json.loads(out) == {}


def test_hook_main_no_event_arg_exits_zero(tmp_path):
    rc, out = _run_hook([], '{"sessionId": "s1"}', tmp_path)
    assert rc == 0
    assert out == "{}"


def test_hook_main_missing_session_id_exits_zero_and_does_not_persist(tmp_path):
    rc, out = _run_hook(["preToolUse"], '{"toolName": "shell"}', tmp_path)
    assert rc == 0
    assert json.loads(out) == {}
    assert state.list_sessions(state_dir=tmp_path) == []


def test_hook_main_valid_event_persists_and_exits_zero(tmp_path):
    payload = json.dumps({"sessionId": "s1", "toolName": "shell", "toolArgs": {"cmd": "ls"}})
    rc, out = _run_hook(["preToolUse"], payload, tmp_path)
    assert rc == 0
    assert json.loads(out) == {}
    events = state.read_events("s1", state_dir=tmp_path)
    assert len(events) == 1
    assert events[0]["event"] == "preToolUse"
    # Default capture_content=False: raw content never touched disk.
    assert events[0]["payload"]["toolArgs"] == {"_stripped": True, "length": len('{"cmd": "ls"}')}


def test_hook_main_unknown_event_name_ignored(tmp_path):
    payload = json.dumps({"sessionId": "s1"})
    rc, out = _run_hook(["someFutureEvent"], payload, tmp_path)
    assert rc == 0
    assert json.loads(out) == {}
    assert state.list_sessions(state_dir=tmp_path) == []


def test_hook_main_session_end_spawns_detached_flush(tmp_path):
    payload = json.dumps({"sessionId": "s1", "reason": "complete"})
    with patch("subprocess.Popen") as mock_popen:
        rc, out = _run_hook(["sessionEnd"], payload, tmp_path)
    assert rc == 0
    assert json.loads(out) == {}
    mock_popen.assert_called_once()
    call_args = mock_popen.call_args[0][0]
    assert "traccia.integrations.github_copilot.flush" in call_args
    assert "--session" in call_args and "s1" in call_args


def test_hook_main_survives_internal_exception(tmp_path):
    payload = json.dumps({"sessionId": "s1", "toolName": "shell"})
    with patch(
        "traccia.integrations.github_copilot.state.append_event",
        side_effect=RuntimeError("disk full"),
    ):
        rc, out = _run_hook(["preToolUse"], payload, tmp_path)
    assert rc == 0
    assert json.loads(out) == {}


def test_hook_main_disabled_via_config_is_noop(tmp_path):
    class _Cfg:
        class instrumentation:
            github_copilot = False
            github_copilot_capture_content = False

    payload = json.dumps({"sessionId": "s1", "toolName": "shell"})
    with patch("traccia.config.load_config", return_value=_Cfg()):
        rc, out = _run_hook(["preToolUse"], payload, tmp_path)
    assert rc == 0
    assert json.loads(out) == {}
    assert state.list_sessions(state_dir=tmp_path) == []


# ---------------------------------------------------------------------------
# flush.py
# ---------------------------------------------------------------------------


def test_flush_session_no_buffered_events_returns_none(tmp_path):
    assert flush_mod.flush_session("nope", state_dir=tmp_path) is None


def test_flush_session_inits_and_tears_down_fresh_provider(tmp_path):
    state.append_event("s1", "sessionStart", {"sessionId": "s1"}, state_dir=tmp_path)
    state.append_event("s1", "sessionEnd", {"sessionId": "s1"}, state_dir=tmp_path)

    with patch("traccia.auto._started", False), patch("traccia.init") as mock_init, patch(
        "traccia.end_auto_trace"
    ) as mock_end_auto, patch("traccia.get_tracer") as mock_get_tracer, patch(
        "traccia.force_flush"
    ) as mock_flush, patch("traccia.stop_tracing") as mock_stop:
        result = flush_mod.flush_session("s1", state_dir=tmp_path)

    mock_init.assert_called_once()
    mock_end_auto.assert_called_once()  # guards against init()'s auto-trace-priority quirk -- see flush.py
    mock_get_tracer.assert_called_once_with("github_copilot")
    mock_flush.assert_called_once()
    mock_stop.assert_called_once()
    assert result is not None
    assert state.read_events("s1", state_dir=tmp_path) == []  # cleared after flush


def test_flush_session_reuses_already_started_provider(tmp_path):
    state.append_event("s1", "sessionStart", {"sessionId": "s1"}, state_dir=tmp_path)
    state.append_event("s1", "sessionEnd", {"sessionId": "s1"}, state_dir=tmp_path)

    with patch("traccia.auto._started", True), patch("traccia.init") as mock_init, patch(
        "traccia.get_tracer"
    ), patch("traccia.force_flush") as mock_flush, patch("traccia.stop_tracing") as mock_stop:
        flush_mod.flush_session("s1", state_dir=tmp_path)

    mock_init.assert_not_called()
    mock_flush.assert_called_once()
    mock_stop.assert_not_called()  # must not tear down a provider we didn't start


def test_flush_all_skips_sessions_younger_than_max_age(tmp_path):
    state.append_event("fresh", "sessionStart", {"sessionId": "fresh"}, state_dir=tmp_path)

    with patch("traccia.integrations.github_copilot.flush.flush_session") as mock_flush_one:
        results = flush_mod.flush_all(max_age_seconds=3600, state_dir=tmp_path)

    mock_flush_one.assert_not_called()
    assert results == {}


def test_flush_all_flushes_old_sessions(tmp_path):
    state.append_event("old", "sessionStart", {"sessionId": "old"}, state_dir=tmp_path)
    path = state.session_log_path("old", state_dir=tmp_path)
    old_time = time.time() - 7200
    import os

    os.utime(path, (old_time, old_time))

    with patch(
        "traccia.integrations.github_copilot.flush.flush_session", return_value={"tool_spans": 0}
    ) as mock_flush_one:
        results = flush_mod.flush_all(max_age_seconds=3600, state_dir=tmp_path)

    mock_flush_one.assert_called_once()
    assert results == {"old": {"tool_spans": 0}}


# ---------------------------------------------------------------------------
# __init__.py -- install()
# ---------------------------------------------------------------------------


def test_install_enabled_by_default():
    assert install_fn() is True


def test_install_explicit_disable():
    assert install_fn(enabled=False) is False


def test_install_disabled_via_runtime_config():
    from traccia import runtime_config

    runtime_config.set_config_value("github_copilot", False)
    try:
        assert install_fn(enabled=None) is False
    finally:
        runtime_config.set_config_value("github_copilot", None)


# ---------------------------------------------------------------------------
# Durability / concurrency hardening (follow-up review)
# ---------------------------------------------------------------------------


def test_strip_content_fields_redacts_error_message_and_keeps_type():
    payload = {
        "sessionId": "s1",
        "error": {"message": "email bob@example.com then retry", "type": "ToolError"},
    }
    out = mapping.strip_content_fields("postToolUseFailure", payload, capture_content=False)
    assert "bob@example.com" not in out["error"]["message"]
    assert "[REDACTED_EMAIL]" in out["error"]["message"]
    assert out["error"]["type"] == "ToolError"


def test_end_attributes_post_tool_use_failure_carries_error_type():
    result = mapping.end_attributes(
        "postToolUseFailure", {"error": {"message": "boom", "type": "Timeout"}}
    )
    assert result["attributes"]["error.type"] == "Timeout"


def test_state_claim_discard_restore_roundtrip(tmp_path):
    state.append_event("s1", "sessionStart", {"sessionId": "s1"}, state_dir=tmp_path)

    claimed = state.claim_session("s1", state_dir=tmp_path)
    assert claimed is not None and claimed.name == "s1.jsonl.flushing"
    # A claimed log is no longer a plain *.jsonl session.
    assert state.list_sessions(state_dir=tmp_path) == []
    # A second concurrent flush cannot claim the same session.
    assert state.claim_session("s1", state_dir=tmp_path) is None

    state.restore_claim(claimed)
    assert state.list_sessions(state_dir=tmp_path) == ["s1"]

    claimed = state.claim_session("s1", state_dir=tmp_path)
    state.discard_claim(claimed)
    assert state.read_events("s1", state_dir=tmp_path) == []


def test_state_has_end_event(tmp_path):
    state.append_event("s1", "sessionStart", {"sessionId": "s1"}, state_dir=tmp_path)
    assert state.has_end_event("s1", state_dir=tmp_path) is False
    state.append_event("s1", "sessionEnd", {"sessionId": "s1"}, state_dir=tmp_path)
    assert state.has_end_event("s1", state_dir=tmp_path) is True


def test_flush_session_claim_prevents_double_export(tmp_path):
    state.append_event("s1", "sessionStart", {"sessionId": "s1"}, state_dir=tmp_path)
    state.append_event("s1", "sessionEnd", {"sessionId": "s1"}, state_dir=tmp_path)

    held = state.claim_session("s1", state_dir=tmp_path)  # simulate an in-flight flush
    assert held is not None
    assert flush_mod.flush_session("s1", state_dir=tmp_path) is None


def test_flush_session_archives_on_failed_export_instead_of_deleting(tmp_path):
    state.append_event("s1", "sessionStart", {"sessionId": "s1"}, state_dir=tmp_path)
    state.append_event("s1", "sessionEnd", {"sessionId": "s1"}, state_dir=tmp_path)

    with patch("traccia.auto._started", True), patch("traccia.get_tracer"), patch(
        "traccia.force_flush", return_value=False
    ) as mock_flush, patch("traccia.stop_tracing"):
        flush_mod.flush_session("s1", state_dir=tmp_path)

    mock_flush.assert_called_once()
    # Buffer was NOT dropped -- it moved to failed/ for retry.
    assert state.read_events("s1", state_dir=tmp_path) == []
    failed = state.list_failed(state_dir=tmp_path)
    assert len(failed) == 1
    assert [e["event"] for e in state.read_events_from_path(failed[0])] == [
        "sessionStart",
        "sessionEnd",
    ]


def test_flush_session_clears_on_confirmed_export(tmp_path):
    state.append_event("s1", "sessionStart", {"sessionId": "s1"}, state_dir=tmp_path)
    state.append_event("s1", "sessionEnd", {"sessionId": "s1"}, state_dir=tmp_path)

    with patch("traccia.auto._started", True), patch("traccia.get_tracer"), patch(
        "traccia.force_flush", return_value=True
    ), patch("traccia.stop_tracing"):
        flush_mod.flush_session("s1", state_dir=tmp_path)

    assert state.read_events("s1", state_dir=tmp_path) == []
    assert state.list_failed(state_dir=tmp_path) == []


def test_flush_all_skips_live_session_without_session_end(tmp_path):
    state.append_event("live", "sessionStart", {"sessionId": "live"}, state_dir=tmp_path)
    with patch("traccia.integrations.github_copilot.flush.flush_session") as mock_one:
        results = flush_mod.flush_all(state_dir=tmp_path)
    mock_one.assert_not_called()
    assert results == {}


def test_flush_all_flushes_session_with_recorded_session_end(tmp_path):
    state.append_event("done", "sessionStart", {"sessionId": "done"}, state_dir=tmp_path)
    state.append_event("done", "sessionEnd", {"sessionId": "done"}, state_dir=tmp_path)
    with patch(
        "traccia.integrations.github_copilot.flush.flush_session", return_value={"tool_spans": 0}
    ) as mock_one:
        results = flush_mod.flush_all(state_dir=tmp_path)
    mock_one.assert_called_once()
    assert results == {"done": {"tool_spans": 0}}


def test_list_stale_claims_only_returns_old_markers(tmp_path):
    import os as _os

    fresh = tmp_path / "fresh.jsonl.flushing"
    fresh.write_text("{}\n", encoding="utf-8")
    stale = tmp_path / "stale.jsonl.flushing"
    stale.write_text("{}\n", encoding="utf-8")
    old = time.time() - 7200
    _os.utime(stale, (old, old))

    found = state.list_stale_claims(3600, state_dir=tmp_path)
    assert found == [stale]


def test_flush_all_reclaims_stale_flushing_marker(tmp_path):
    marker = tmp_path / "orphan.jsonl.flushing"
    marker.write_text(
        json.dumps({"event": "sessionStart", "payload": {"sessionId": "orphan"}, "received_at": 1.0})
        + "\n",
        encoding="utf-8",
    )
    import os as _os

    old = time.time() - 7200
    _os.utime(marker, (old, old))

    with patch("traccia.auto._started", True), patch("traccia.get_tracer"), patch(
        "traccia.force_flush", return_value=True
    ), patch("traccia.stop_tracing"):
        results = flush_mod.flush_all(state_dir=tmp_path)

    assert "orphan.jsonl.flushing" in results
    assert state.list_stale_claims(0, state_dir=tmp_path) == []  # marker consumed


def test_flush_all_include_active_flushes_live_session(tmp_path):
    state.append_event("live", "sessionStart", {"sessionId": "live"}, state_dir=tmp_path)
    with patch(
        "traccia.integrations.github_copilot.flush.flush_session", return_value={}
    ) as mock_one:
        flush_mod.flush_all(include_active=True, state_dir=tmp_path)
    mock_one.assert_called_once()


def test_retry_failed_reexports_parked_logs(tmp_path):
    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    (failed_dir / "s1.123.jsonl").write_text(
        json.dumps({"event": "sessionStart", "payload": {"sessionId": "s1"}, "received_at": 1.0})
        + "\n"
        + json.dumps({"event": "sessionEnd", "payload": {"sessionId": "s1"}, "received_at": 2.0})
        + "\n",
        encoding="utf-8",
    )
    with patch("traccia.auto._started", True), patch("traccia.get_tracer"), patch(
        "traccia.force_flush", return_value=True
    ), patch("traccia.stop_tracing"):
        results = flush_mod.retry_failed(state_dir=tmp_path)
    assert list(results) == ["s1.123.jsonl"]
    assert state.list_failed(state_dir=tmp_path) == []


# ---------------------------------------------------------------------------
# spans.py -- timing robustness
# ---------------------------------------------------------------------------


def test_build_trace_clamps_negative_duration_from_skewed_timestamps():
    tracer, exporter = _make_tracer()
    base = time.time()
    events = [
        {"event": "sessionStart", "payload": {"sessionId": "s1", "timestamp": base}, "received_at": base},
        {
            "event": "preToolUse",
            "payload": {"sessionId": "s1", "toolName": "shell", "timestamp": base + 5},
            "received_at": base + 1,
        },
        {
            # arrives later but its own timestamp is *earlier* than preToolUse's
            "event": "postToolUse",
            "payload": {
                "sessionId": "s1",
                "toolName": "shell",
                "timestamp": base + 2,
                "toolResult": {"textResultForLlm": "ok"},
            },
            "received_at": base + 2,
        },
    ]
    spans_mod.build_trace(tracer, events)
    tool_span = next(
        s for s in exporter.get_finished_spans() if s.name == "github_copilot.tool.shell"
    )
    assert tool_span.end_time >= tool_span.start_time  # never negative


def test_build_trace_safety_net_uses_last_event_time_not_wall_clock():
    tracer, exporter = _make_tracer()
    old = time.time() - 10_000  # session happened long before this flush runs
    events = [
        {"event": "sessionStart", "payload": {"sessionId": "s1", "timestamp": old}, "received_at": old},
        {
            "event": "preToolUse",
            "payload": {"sessionId": "s1", "toolName": "shell", "timestamp": old + 1},
            "received_at": old + 1,
        },
        # no postToolUse, no sessionEnd -> safety net closes both
    ]
    spans_mod.build_trace(tracer, events)
    cutoff_ns = int((old + 300) * 1e9)
    for s in exporter.get_finished_spans():
        assert s.end_time <= cutoff_ns  # closed near the last event, not "now"


def test_build_trace_adds_vcs_attributes_from_cwd(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*a):
        subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)

    try:
        _git("init", "-q")
        _git("config", "user.email", "t@example.com")
        _git("config", "user.name", "t")
        _git("commit", "--allow-empty", "-m", "init", "-q")
        _git("branch", "-M", "feat/x")
    except (FileNotFoundError, subprocess.CalledProcessError):
        import pytest

        pytest.skip("git not available")

    tracer, exporter = _make_tracer()
    events = _events(
        ("sessionStart", {"sessionId": "s1", "cwd": str(repo)}, 0),
        ("sessionEnd", {"sessionId": "s1"}, 1),
    )
    spans_mod.build_trace(tracer, events)
    session_span = next(
        s for s in exporter.get_finished_spans() if s.name == "github_copilot.session"
    )
    assert session_span.attributes.get("vcs.branch.name") == "feat/x"
    assert "vcs.commit.sha" in session_span.attributes


# ---------------------------------------------------------------------------
# cli.py -- install-hooks
# ---------------------------------------------------------------------------


def test_install_hooks_quotes_spaced_interpreter_path(tmp_path, monkeypatch):
    import types

    from traccia import cli

    monkeypatch.chdir(tmp_path)
    args = types.SimpleNamespace(
        scope="repo", force=True, python="/opt/py 3.13/bin/python"
    )
    assert cli._copilot_install_hooks(args) == 0

    cfg = json.loads((tmp_path / ".github" / "hooks" / "traccia.json").read_text())
    command = cfg["hooks"]["sessionStart"][0]["command"]
    assert command.startswith('"/opt/py 3.13/bin/python" -m traccia.integrations.github_copilot.hook')
