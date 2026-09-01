"""Build Traccia spans from a GitHub Copilot session's buffered hook events.

Runs once per session (from flush.py), not once per hook invocation -- see
docs/github-copilot-integration.md for why. Uses each event's own recorded
timestamp (via Tracer.start_span(start_time=...) / Span.end(end_time=...))
so span duration reflects when things actually happened, not when this
materializer ran.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from traccia.integrations.github_copilot import mapping
from traccia.processors.redaction_processor import redact_attributes, redact_string
from traccia.tracer.span import SpanStatus


def _vcs_attributes(cwd: Optional[str]) -> Dict[str, str]:
    """Best-effort repo/branch/commit for a session's working directory.

    Runs here (in flush.py's materialization step), never on Copilot's blocking
    hook path. Never raises; capped at a couple seconds; returns ``{}`` when
    ``cwd`` isn't a git checkout or ``git`` isn't on PATH.
    """
    if not cwd:
        return {}
    import re
    import subprocess

    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", cwd, *args],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        val = (out.stdout or "").strip()
        return val or None

    attrs: Dict[str, str] = {}
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch and branch != "HEAD":
        attrs["vcs.branch.name"] = branch
    sha = _git("rev-parse", "HEAD")
    if sha:
        attrs["vcs.commit.sha"] = sha
    remote = _git("config", "--get", "remote.origin.url")
    if remote:
        # Never let credentials in an embedded userinfo (https://user:token@host/…)
        # ride along into a span attribute.
        attrs["vcs.repository.url"] = re.sub(r"//[^/@]*@", "//", remote)
    return attrs


def _epoch_to_ns(value: Any) -> Optional[int]:
    """Best-effort conversion of an epoch timestamp of unknown unit to ns.

    Copilot's hooks reference documents `timestamp: number` without a unit.
    Bucket by magnitude (valid across seconds/ms/us/ns for any date in the
    2020s-2030s) rather than assuming one unit and silently mis-scaling
    durations.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value > 1e17:
        return int(value)  # already nanoseconds
    if value > 1e14:
        return int(value * 1_000)  # microseconds
    if value > 1e11:
        return int(value * 1_000_000)  # milliseconds
    return int(value * 1_000_000_000)  # seconds


def _event_time_ns(payload: Dict[str, Any], received_at: float) -> int:
    ns = _epoch_to_ns(payload.get("timestamp"))
    if ns is not None:
        return ns
    return int(received_at * 1_000_000_000)


def _set_attrs(span: Any, attrs: Dict[str, Any]) -> None:
    for key, value in redact_attributes(attrs).items():
        span.set_attribute(key, value)


def _clamped_end(span: Any, end_ns: int) -> int:
    """Never let a span end before it started.

    Events are ordered by local ``received_at`` but timed by their own
    ``timestamp`` field (a different clock), so a close event can carry a
    timestamp earlier than its open event -- which would otherwise produce a
    negative-duration span. Floor the end at the span's start.
    """
    start = getattr(span, "start_time_ns", None)
    if isinstance(start, int) and end_ns < start:
        return start
    return end_ns


def _discard(stack: List[Any], span: Any) -> None:
    for i, existing in enumerate(stack):
        if existing is span:
            del stack[i]
            return


def build_trace(tracer: Any, events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Replay one session's buffered events onto `tracer`.

    Returns a small summary dict (span/error counts) for callers/tests, or
    None if there was nothing to build.
    """
    if not events:
        return None

    events = sorted(events, key=lambda e: e.get("received_at", 0))

    session_span: Optional[Any] = None
    tool_queues: Dict[str, Deque[Any]] = defaultdict(deque)
    subagent_spans: Dict[str, Any] = {}
    open_spans: List[Any] = []  # innermost-open-last, for errorOccurred attribution
    summary = {"tool_spans": 0, "subagent_spans": 0, "errors": 0}
    last_event_ns: Optional[int] = None

    def ensure_session_span(payload: Dict[str, Any], start_ns: int) -> Any:
        nonlocal session_span
        if session_span is not None:
            return session_span
        # A tool/subagent event arrived with no sessionStart in this session's
        # log (e.g. hooks were only just enabled mid-session) -- open a
        # session span anyway so the event has somewhere to attach rather
        # than being silently dropped.
        attrs = mapping.start_attributes("sessionStart", payload)
        attrs["github_copilot.session.source"] = "recovered_missing_session_start"
        attrs.update(_vcs_attributes(payload.get("cwd")))
        session_span = tracer.start_span(
            mapping.span_name_for("sessionStart", payload),
            start_time=start_ns,
        )
        _set_attrs(session_span, attrs)
        open_spans.append(session_span)
        return session_span

    for record in events:
        event_name = record.get("event")
        payload = record.get("payload") or {}
        received_at = record.get("received_at", 0)
        start_ns = _event_time_ns(payload, received_at)
        last_event_ns = start_ns if last_event_ns is None else max(last_event_ns, start_ns)

        if event_name in mapping.SESSION_START_EVENTS:
            if session_span is not None:
                continue  # duplicate sessionStart in the log; ignore
            session_span = tracer.start_span(
                mapping.span_name_for(event_name, payload), start_time=start_ns
            )
            _set_attrs(session_span, mapping.start_attributes(event_name, payload))
            _set_attrs(session_span, _vcs_attributes(payload.get("cwd")))
            open_spans.append(session_span)

        elif event_name in mapping.TOOL_START_EVENTS:
            parent = ensure_session_span(payload, start_ns)
            tool_name = payload.get("toolName") or "unknown"
            span = tracer.start_span(
                mapping.span_name_for(event_name, payload),
                parent=parent,
                start_time=start_ns,
            )
            _set_attrs(span, mapping.start_attributes(event_name, payload))
            tool_queues[tool_name].append(span)
            open_spans.append(span)
            summary["tool_spans"] += 1

        elif event_name in mapping.TOOL_END_EVENTS:
            tool_name = payload.get("toolName") or "unknown"
            queue = tool_queues[tool_name]
            span = queue.popleft() if queue else None
            if span is None:
                # postToolUse(Failure) with no matching preToolUse captured in
                # this session's log -- open+close a zero-duration span so the
                # event isn't silently dropped.
                parent = ensure_session_span(payload, start_ns)
                span = tracer.start_span(
                    mapping.span_name_for(event_name, payload),
                    parent=parent,
                    start_time=start_ns,
                )
                _set_attrs(span, mapping.start_attributes("preToolUse", payload))
            else:
                _discard(open_spans, span)
            result = mapping.end_attributes(event_name, payload)
            _set_attrs(span, result["attributes"])
            if result["is_error"]:
                span.set_status(SpanStatus.ERROR, result["error_message"])
                summary["errors"] += 1
            else:
                span.set_status(SpanStatus.OK)
            span.end(end_time=_clamped_end(span, start_ns))

        elif event_name in mapping.SUBAGENT_START_EVENTS:
            parent = ensure_session_span(payload, start_ns)
            span = tracer.start_span(
                mapping.span_name_for(event_name, payload),
                parent=parent,
                start_time=start_ns,
            )
            _set_attrs(span, mapping.start_attributes(event_name, payload))
            key = payload.get("agentId") or payload.get("agentName") or "unknown"
            subagent_spans[key] = span
            open_spans.append(span)
            summary["subagent_spans"] += 1

        elif event_name in mapping.SUBAGENT_END_EVENTS:
            key = payload.get("agentId") or payload.get("agentName") or "unknown"
            span = subagent_spans.pop(key, None)
            if span is None:
                continue  # no matching subagentStart captured in this session's log
            _discard(open_spans, span)
            result = mapping.end_attributes(event_name, payload)
            _set_attrs(span, result["attributes"])
            span.set_status(SpanStatus.OK)
            span.end(end_time=_clamped_end(span, start_ns))

        elif event_name == "errorOccurred":
            summary["errors"] += 1
            target = open_spans[-1] if open_spans else session_span
            if target is not None:
                error = payload.get("error") or {}
                message = error.get("message") if isinstance(error, dict) else str(error)
                err_type = error.get("type") if isinstance(error, dict) else None
                # add_event() attaches straight to the OTel span, bypassing
                # _set_attrs()'s redact_attributes() call -- redact explicitly
                # here so an error message containing e.g. an email doesn't
                # slip through unredacted the way a regular attribute wouldn't.
                # (strip_content_fields already redacted this before persistence;
                # this is belt-and-suspenders for the direct-call path in tests.)
                event_attrs = {"error.message": redact_string((message or "")[:200])}
                if err_type:
                    event_attrs["error.type"] = str(err_type)[:200]
                target.add_event(
                    "github_copilot.error",
                    event_attrs,
                    timestamp_ns=start_ns,
                )

        elif event_name in mapping.SESSION_EVENT_ONLY:
            if session_span is not None:
                session_span.add_event(f"github_copilot.{event_name}", {}, timestamp_ns=start_ns)

        elif event_name in mapping.SESSION_END_EVENTS:
            if session_span is None:
                continue
            reason = payload.get("reason")
            if reason:
                session_span.set_attribute("github_copilot.session.end_reason", reason)
            if reason == "error":
                session_span.set_status(SpanStatus.ERROR, "session ended with an error")
            else:
                session_span.set_status(SpanStatus.OK)
            session_span.end(end_time=_clamped_end(session_span, start_ns))
            _discard(open_spans, session_span)

        # IGNORED_EVENTS and unrecognized event names: intentionally no-op.

    # Safety net: close anything still open (e.g. sessionEnd never arrived) so
    # a flush always yields a complete, exportable trace, never dangling spans.
    # Use the last observed event time, not time.time_ns(): a session recovered
    # by `flush --all` hours later must not get an hours-long bogus duration.
    for span in reversed(open_spans):
        try:
            if last_event_ns is not None:
                span.end(end_time=_clamped_end(span, last_event_ns))
            else:
                span.end()
        except Exception:
            pass

    return summary
