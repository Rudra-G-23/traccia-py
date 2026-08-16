"""Groq monkey patching for chat completions (OpenAI-compatible API)."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Callable
from traccia.tracer.span import SpanStatus

_patched = False


def _compute_cost(model: Optional[str], prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> Optional[float]:
    """Compute cost from token usage using pricing config."""
    if not model or prompt_tokens is None or completion_tokens is None:
        return None
    try:
        from traccia.processors.cost_engine import compute_cost as _compute
        from traccia.pricing_config import load_pricing
        return _compute(model, prompt_tokens, completion_tokens, load_pricing())
    except Exception:
        return None


def _record_llm_metrics(
    model: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    duration: Optional[float],
    cost: Optional[float]
):
    """Record LLM metrics if metrics are enabled."""
    try:
        from traccia.metrics.recorder import get_metrics_recorder
        recorder = get_metrics_recorder()
        if not recorder:
            return

        # Build attributes — always include the per-run agent identity so the
        # ingestion metrics converter can attribute the data point to the correct
        # agent rather than falling back to the resource-level service.name.
        attributes: Dict[str, Any] = {
            "gen_ai.system": "groq",
        }
        if model:
            attributes["gen_ai.request.model"] = model

        # Attach per-run agent identity from runtime_config (set by run_identity context manager).
        try:
            from traccia import runtime_config as _rc
            _aid = _rc.get_agent_id()
            _aname = _rc.get_agent_name()
            _env = _rc.get_env()
            if _aid:
                attributes["agent.id"] = _aid
                attributes["agent_id"] = _aid
            if _aname:
                attributes["agent.name"] = _aname
            if _env:
                attributes["environment"] = _env
        except Exception:
            pass

        # Record token usage
        if prompt_tokens is not None or completion_tokens is not None:
            recorder.record_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                attributes=attributes
            )

        # Record duration
        if duration is not None:
            recorder.record_duration(duration, attributes=attributes)

        # Record cost
        if cost is not None:
            recorder.record_cost(cost, attributes=attributes)
    except Exception:
        # Silently fail if metrics recording fails
        pass


def _safe_get(obj, path: str, default=None):
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, (list, tuple)) and part.lstrip("-").isdigit():
            # Dotted paths use numeric segments for list indices (e.g. "choices.0.message"),
            # which plain getattr() can't resolve since sequences aren't attribute-addressable.
            idx = int(part)
            cur = cur[idx] if -len(cur) <= idx < len(cur) else None
        else:
            cur = getattr(cur, part, None)
    return cur if cur is not None else default


def patch_groq() -> bool:
    """Patch Groq chat completions (sync + async clients)."""
    global _patched
    if _patched:
        return True
    try:
        import groq
    except Exception:
        return False

    def _extract_messages(kwargs, args):
        messages = kwargs.get("messages")
        # For new client, first arg after self is messages
        if messages is None and len(args) >= 2:
            messages = args[1]
        if not messages or not isinstance(messages, (list, tuple)):
            return None
        # Keep only JSON-friendly, small fields to avoid huge/sensitive payloads.
        slim = []
        for m in list(messages)[:50]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            name = m.get("name")
            content = m.get("content")
            if isinstance(content, (list, dict)):
                content = str(content)
            elif content is not None and not isinstance(content, str):
                content = str(content)
            item = {"role": role, "content": content}
            if name:
                item["name"] = name
            slim.append(item)
        return slim or None

    def _extract_prompt_text(messages_slim) -> Optional[str]:
        if not messages_slim:
            return None
        parts = []
        for m in messages_slim:
            role = m.get("role")
            content = m.get("content")
            if not content:
                continue
            parts.append(f"{role}: {content}" if role else str(content))
        return "\n".join(parts) if parts else None

    def _extract_prompt(kwargs, args) -> Optional[str]:
        messages = kwargs.get("messages")
        if messages is None and len(args) >= 2:
            messages = args[1]
        if not messages:
            return None
        parts = []
        for m in messages:
            content = m.get("content")
            role = m.get("role")
            if content:
                parts.append(f"{role}: {content}" if role else str(content))
        return "\n".join(parts) if parts else None

    def _build_span_attrs(kwargs, args):
        model = kwargs.get("model") or _safe_get(args, "0.model", None)
        messages_slim = _extract_messages(kwargs, args)
        prompt_text = _extract_prompt_text(messages_slim) or _extract_prompt(kwargs, args)
        attributes: Dict[str, Any] = {"llm.vendor": "groq"}
        if model:
            attributes["llm.model"] = model
        if messages_slim:
            import json
            try:
                attributes["llm.groq.messages"] = json.dumps(messages_slim)[:1000]
            except Exception:
                attributes["llm.groq.messages"] = str(messages_slim)[:1000]
        if prompt_text:
            attributes["llm.prompt"] = prompt_text
        return model, attributes

    def _populate_response(span, resp, model):
        # A stream=True call returns an (Async)Stream generator that hasn't been
        # consumed yet, so model/usage/choices aren't available here — the span
        # closes with only the request-side attributes, matching the OpenAI patch.
        resp_model = getattr(resp, "model", None) or (_safe_get(resp, "model"))
        if resp_model and "llm.model" not in span.attributes:
            span.set_attribute("llm.model", resp_model)
        usage = getattr(resp, "usage", None) or (resp.get("usage") if isinstance(resp, dict) else None)
        prompt_tokens_val = None
        completion_tokens_val = None
        if usage:
            span.set_attribute("llm.usage.source", "provider_usage")
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = getattr(usage, k, None) if not isinstance(usage, dict) else usage.get(k)
                if val is not None:
                    span.set_attribute(f"llm.usage.{k}", val)
                    if k == "prompt_tokens":
                        prompt_tokens_val = val
                    elif k == "completion_tokens":
                        completion_tokens_val = val
            if "llm.usage.prompt_tokens" in span.attributes:
                span.set_attribute("llm.usage.prompt_source", "provider_usage")
            if "llm.usage.completion_tokens" in span.attributes:
                span.set_attribute("llm.usage.completion_source", "provider_usage")
        finish_reason = _safe_get(resp, "choices.0.finish_reason")
        if finish_reason:
            span.set_attribute("llm.finish_reason", finish_reason)
        completion = _safe_get(resp, "choices.0.message.content")
        if completion:
            span.set_attribute("llm.completion", completion)
        return resp_model, prompt_tokens_val, completion_tokens_val

    def _record_exception(span, exc, model):
        span.record_exception(exc)
        span.set_status(SpanStatus.ERROR, str(exc))
        try:
            from traccia.metrics.recorder import get_metrics_recorder
            rec = get_metrics_recorder()
            if rec:
                rec.record_exception(attributes={"gen_ai.system": "groq", "gen_ai.request.model": model or "unknown"})
        except Exception:
            pass

    def _wrap_sync(create_fn: Callable):
        if getattr(create_fn, "_agent_trace_patched", False):
            return create_fn

        def wrapped_create(*args, **kwargs):
            tracer = _get_tracer("groq")
            model, attributes = _build_span_attrs(kwargs, args)
            t0 = time.perf_counter()
            with tracer.start_as_current_span("llm.groq.chat.completions", attributes=attributes) as span:
                try:
                    resp = create_fn(*args, **kwargs)
                    resp_model, prompt_tokens_val, completion_tokens_val = _populate_response(span, resp, model)

                    duration_val = time.perf_counter() - t0
                    cost_val = _compute_cost(resp_model or model, prompt_tokens_val, completion_tokens_val)
                    _record_llm_metrics(
                        model=resp_model or model,
                        prompt_tokens=prompt_tokens_val,
                        completion_tokens=completion_tokens_val,
                        duration=duration_val,
                        cost=cost_val
                    )

                    return resp
                except Exception as exc:
                    _record_exception(span, exc, model)
                    raise

        wrapped_create._agent_trace_patched = True
        return wrapped_create

    def _wrap_async(create_fn: Callable):
        if getattr(create_fn, "_agent_trace_patched", False):
            return create_fn

        async def wrapped_create(*args, **kwargs):
            tracer = _get_tracer("groq")
            model, attributes = _build_span_attrs(kwargs, args)
            t0 = time.perf_counter()
            with tracer.start_as_current_span("llm.groq.chat.completions", attributes=attributes) as span:
                try:
                    resp = await create_fn(*args, **kwargs)
                    resp_model, prompt_tokens_val, completion_tokens_val = _populate_response(span, resp, model)

                    duration_val = time.perf_counter() - t0
                    cost_val = _compute_cost(resp_model or model, prompt_tokens_val, completion_tokens_val)
                    _record_llm_metrics(
                        model=resp_model or model,
                        prompt_tokens=prompt_tokens_val,
                        completion_tokens=completion_tokens_val,
                        duration=duration_val,
                        cost=cost_val
                    )

                    return resp
                except Exception as exc:
                    _record_exception(span, exc, model)
                    raise

        wrapped_create._agent_trace_patched = True
        return wrapped_create

    patched_any = False

    # New client: Groq.chat.completions.create
    client_cls = getattr(groq, "Groq", None)
    if client_cls and hasattr(client_cls, "chat"):
        chat = getattr(client_cls, "chat", None)
        if chat and hasattr(chat, "completions"):
            completions = getattr(chat, "completions")
            if hasattr(completions, "create"):
                setattr(completions, "create", _wrap_sync(completions.create))
                patched_any = True

    # New async client: AsyncGroq.chat.completions.create
    async_client_cls = getattr(groq, "AsyncGroq", None)
    if async_client_cls and hasattr(async_client_cls, "chat"):
        async_chat = getattr(async_client_cls, "chat", None)
        if async_chat and hasattr(async_chat, "completions"):
            async_completions = getattr(async_chat, "completions")
            if hasattr(async_completions, "create"):
                setattr(async_completions, "create", _wrap_async(async_completions.create))
                patched_any = True

    # Resource classes directly, mirroring the OpenAI patch: covers SDK versions
    # where the client-class attribute lookup above doesn't reach the bound method.
    try:
        from groq.resources.chat.completions import Completions  # type: ignore

        if hasattr(Completions, "create"):
            Completions.create = _wrap_sync(Completions.create)
            patched_any = True
    except Exception:
        pass

    try:
        from groq.resources.chat.completions import AsyncCompletions  # type: ignore

        if hasattr(AsyncCompletions, "create"):
            AsyncCompletions.create = _wrap_async(AsyncCompletions.create)
            patched_any = True
    except Exception:
        pass

    if patched_any:
        _patched = True

    return patched_any


def _get_tracer(name: str):
    import traccia

    return traccia.get_tracer(name)
