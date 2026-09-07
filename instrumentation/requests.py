"""Requests monkey patching for HTTP client tracing."""

from __future__ import annotations

from typing import Any, Dict
from traccia.tracer.span import SpanStatus
from traccia.context import get_current_span, inject_traceparent, inject_tracestate

_patched = False

# Skip instrumentation for Traccia platform HTTP — avoids feedback loops (OTLP)
# and bookkeeping traces (eval/prompt/govern) polluting Observe.
# Keep in sync with traccia-node `src/instrumentation/http-skip.ts`.
_SKIP_URL_SUBSTRINGS = (
    "/v1/traces",
    "/v2/traces",
    "/api/v1/traces",
    "/api/v2/traces",
    "/v1/metrics",
    "/v2/metrics",
    "/api/v1/metrics",
    "/api/v2/metrics",
    "/api/v1/eval-runtime/",
    "/api/v1/prompt-runtime/",
    "/api/v1/agents/",
    "/api/v1/policy/",
)


def _should_skip_http_instrumentation(url: str) -> bool:
    if any(path in url for path in _SKIP_URL_SUBSTRINGS):
        return True
    normalized = url.rstrip("/")
    if "/agents/" in url and (
        normalized.endswith("/status") or "/blocks" in url
    ):
        return True
    return False


def patch_requests() -> bool:
    """Patch requests.Session.request; returns True if patched, False otherwise."""
    global _patched
    if _patched:
        return True
    try:
        import requests
    except Exception:
        return False

    original_request = requests.sessions.Session.request
    if getattr(original_request, "_agent_trace_patched", False):
        _patched = True
        return True

    def wrapped_request(self, method, url, *args, **kwargs):
        url_str = str(url) if url else ""
        if _should_skip_http_instrumentation(url_str):
            return original_request(self, method, url, *args, **kwargs)

        tracer = _get_tracer("requests")
        attributes: Dict[str, Any] = {
            "http.method": method,
            "http.url": url,
        }
        headers = kwargs.get("headers")
        if headers is None:
            headers = {}
            kwargs["headers"] = headers
        current = get_current_span()
        if current:
            inject_traceparent(headers, current.context)
            inject_tracestate(headers, current.context)
        with tracer.start_as_current_span("http.client", attributes=attributes) as span:
            try:
                resp = original_request(self, method, url, *args, **kwargs)
                span.set_attribute("http.status_code", getattr(resp, "status_code", None))
                return resp
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(SpanStatus.ERROR, str(exc))
                raise

    wrapped_request._agent_trace_patched = True
    requests.sessions.Session.request = wrapped_request
    _patched = True
    return True


def _get_tracer(name: str):
    import traccia

    return traccia.get_tracer(name)

