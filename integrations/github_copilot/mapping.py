"""Pure mapping from GitHub Copilot hook events to Traccia span shape.

No I/O here -- keeps this module trivially unit-testable in isolation from
the hook subprocess, the local session log, and the tracer. See
docs/github-copilot-integration.md for the full event -> span table and the
exact field names sourced from GitHub's hooks reference.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

# Events that open a span.
SESSION_START_EVENTS = frozenset({"sessionStart"})
TOOL_START_EVENTS = frozenset({"preToolUse"})
SUBAGENT_START_EVENTS = frozenset({"subagentStart"})

# Events that close a previously-opened span.
SESSION_END_EVENTS = frozenset({"sessionEnd"})
TOOL_END_EVENTS = frozenset({"postToolUse", "postToolUseFailure"})
SUBAGENT_END_EVENTS = frozenset({"subagentStop"})

# Events recorded as a span *event* on the session span, but that never open
# or close a span of their own.
SESSION_EVENT_ONLY = frozenset(
    {"preCompact", "userPromptSubmitted", "userPromptTransformed"}
)

# Events buffered for completeness but with no tracing value -- intentionally
# not turned into spans or span events, to avoid an explosion of low-value
# noise (see docs/github-copilot-integration.md limitations).
IGNORED_EVENTS = frozenset({"agentStop", "notification", "permissionRequest"})

ALL_KNOWN_EVENTS = (
    SESSION_START_EVENTS
    | TOOL_START_EVENTS
    | SUBAGENT_START_EVENTS
    | SESSION_END_EVENTS
    | TOOL_END_EVENTS
    | SUBAGENT_END_EVENTS
    | SESSION_EVENT_ONLY
    | IGNORED_EVENTS
    | {"errorOccurred"}
)

# Fields that hold free-text content (prompts, code, tool I/O) per event,
# stripped to length-only unless capture_content is enabled.
_CONTENT_FIELDS_BY_EVENT: Dict[str, tuple] = {
    "sessionStart": ("initialPrompt",),
    "preToolUse": ("toolArgs",),
    "postToolUse": ("toolResult",),
    "subagentStop": ("response",),
    "userPromptSubmitted": ("prompt",),
    "userPromptTransformed": ("prompt", "transformedPrompt"),
}

_MAX_ERROR_CHARS = 200
_MAX_CONTENT_CHARS = 2000


def _length_of(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value))
    except (TypeError, ValueError):
        return len(str(value))


def _safe_text(value: Any) -> str:
    """Best-effort, size-capped string form of an arbitrary JSON value."""
    if isinstance(value, str):
        return value[:_MAX_CONTENT_CHARS]
    try:
        return json.dumps(value)[:_MAX_CONTENT_CHARS]
    except (TypeError, ValueError):
        return str(value)[:_MAX_CONTENT_CHARS]


def strip_content_fields(
    event_name: str, payload: Dict[str, Any], capture_content: bool
) -> Dict[str, Any]:
    """Return a copy of `payload` with content-bearing fields stripped or capped.

    Called before an event is ever written to the local session log, so with
    capture_content=False the raw content never touches disk in the first
    place. After this call, every field named in _CONTENT_FIELDS_BY_EVENT is
    either ``{"_stripped": True, "length": N}`` or an already size-capped
    plain string -- callers never need to re-truncate.
    """
    payload = dict(payload or {})
    for field in _CONTENT_FIELDS_BY_EVENT.get(event_name, ()):
        if field not in payload or payload[field] is None:
            continue
        raw = payload[field]
        if capture_content:
            payload[field] = _safe_text(raw)
        else:
            payload[field] = {"_stripped": True, "length": _length_of(raw)}

    error = payload.get("error")
    if error is not None:
        if isinstance(error, dict):
            message = error.get("message")
            err_type = error.get("type") or error.get("name")
        else:
            message = str(error)
            err_type = None
        # Error text lands in the on-disk session log, so scrub PII here rather
        # than only at span-build time -- a tool stack trace can carry a path,
        # token or email. Lazy import keeps the (error-free) hot path lean.
        from traccia.processors.redaction_processor import redact_string

        cleaned: Dict[str, Any] = {
            "message": redact_string((message or "")[:_MAX_ERROR_CHARS])
        }
        if err_type:
            cleaned["type"] = str(err_type)[:_MAX_ERROR_CHARS]
        payload["error"] = cleaned

    return payload


def span_name_for(event_name: str, payload: Dict[str, Any]) -> str:
    if event_name in SESSION_START_EVENTS:
        return "github_copilot.session"
    if event_name in TOOL_START_EVENTS or event_name in TOOL_END_EVENTS:
        tool_name = payload.get("toolName") or "unknown"
        return f"github_copilot.tool.{tool_name}"
    if event_name in SUBAGENT_START_EVENTS or event_name in SUBAGENT_END_EVENTS:
        agent_name = payload.get("agentName") or payload.get("agentType") or "unknown"
        return f"github_copilot.subagent.{agent_name}"
    return f"github_copilot.{event_name}"


def _content_value(value: Any) -> Optional[str]:
    """Read back a field already processed by strip_content_fields. Returns
    None for a stripped (length-only) field."""
    if isinstance(value, dict) and value.get("_stripped"):
        return None
    if value is None:
        return None
    return value if isinstance(value, str) else _safe_text(value)


def _content_length(value: Any) -> Optional[int]:
    if isinstance(value, dict) and value.get("_stripped"):
        return value.get("length")
    return None


def start_attributes(event_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Attributes to set when a span is opened for this event."""
    attrs: Dict[str, Any] = {
        "github_copilot.event": event_name,
        "agent.span.type": "github_copilot",
    }
    session_id = payload.get("sessionId")
    if session_id:
        attrs["session.id"] = session_id
        attrs["github_copilot.session.id"] = session_id
    cwd = payload.get("cwd")
    if cwd:
        attrs["github_copilot.cwd"] = cwd

    if event_name in SESSION_START_EVENTS:
        attrs["agent.type"] = "github_copilot"
        attrs["agent.name"] = "github-copilot"
        attrs["gen_ai.system"] = "github_copilot"
        source = payload.get("source")
        if source:
            attrs["github_copilot.session.source"] = source
        prompt = payload.get("initialPrompt")
        preview = _content_value(prompt)
        length = _content_length(prompt)
        if preview is not None:
            attrs["github_copilot.prompt.preview"] = preview
        elif length is not None:
            attrs["github_copilot.prompt.length"] = length

    elif event_name in TOOL_START_EVENTS:
        tool_name = payload.get("toolName")
        if tool_name:
            attrs["agent.tool.name"] = tool_name
        tool_args = payload.get("toolArgs")
        preview = _content_value(tool_args)
        length = _content_length(tool_args)
        if preview is not None:
            attrs["agent.tool.input"] = preview
        elif length is not None:
            attrs["agent.tool.input.length"] = length

    elif event_name in SUBAGENT_START_EVENTS:
        agent_name = payload.get("agentName") or payload.get("agentType")
        if agent_name:
            attrs["agent.name"] = agent_name
        display_name = payload.get("agentDisplayName")
        if display_name:
            attrs["agent.display_name"] = display_name
        description = payload.get("agentDescription")
        if description:
            attrs["agent.description"] = _safe_text(description)
        attrs["agent.handoff.from"] = "github-copilot"

    return attrs


def end_attributes(event_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Attributes/status to apply when a span is closed for this event.

    Returns {"attributes": {...}, "is_error": bool, "error_message": str|None}.
    """
    attrs: Dict[str, Any] = {}
    is_error = False
    error_message: Optional[str] = None

    if event_name == "postToolUse":
        result = payload.get("toolResult")
        if isinstance(result, dict) and not result.get("_stripped"):
            text = result.get("textResultForLlm")
            if text is not None:
                attrs["agent.tool.output"] = _safe_text(text)
            result_type = result.get("resultType")
            if result_type:
                attrs["agent.tool.result_type"] = result_type
        else:
            length = _content_length(result)
            if length is not None:
                attrs["agent.tool.output.length"] = length

    elif event_name == "postToolUseFailure":
        is_error = True
        error = payload.get("error")
        if isinstance(error, dict):
            error_message = error.get("message")
            if error_message:
                attrs["error.message"] = error_message
            err_type = error.get("type")
            if err_type:
                attrs["error.type"] = err_type

    elif event_name == "subagentStop":
        response = payload.get("response")
        preview = _content_value(response)
        length = _content_length(response)
        if preview is not None:
            attrs["agent.response"] = preview
        elif length is not None:
            attrs["agent.response.length"] = length

    return {"attributes": attrs, "is_error": is_error, "error_message": error_message}
