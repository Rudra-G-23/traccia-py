"""Tests for Groq auto-instrumentation (instrumentation/groq.py)."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from traccia import get_tracer_provider, set_tracer_provider
from traccia.tracer import TracerProvider
from traccia.tracer.provider import SpanProcessor

import traccia.instrumentation.groq as groq_instr


class _CapturingProcessor(SpanProcessor):
    """Enrichment processor that snapshots span state on_end.

    Snapshotting here (rather than keeping a span reference for later
    inspection) matters: Span.attributes only mirrors the underlying OTel
    span's attributes while the span is still active — on_end fires before
    Span.end() marks it ended, so this is the only reliable read point.
    """

    def __init__(self):
        self.snapshots = []

    def on_end(self, span) -> None:
        self.snapshots.append({"attributes": dict(span.attributes), "status": span.status})


def _fake_chat_completion(model="llama-3.3-70b-versatile", content="Hello there!"):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4, total_tokens=16),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ],
    )


class _FakeStream:
    """Stand-in for groq.Stream: no model/usage/choices until consumed."""

    def __iter__(self):
        return iter(())


class TestPatchGroq(unittest.TestCase):
    """Import-time / idempotency behavior of patch_groq()."""

    def setUp(self):
        groq_instr._patched = False

    def tearDown(self):
        groq_instr._patched = False

    def test_patch_groq_returns_true_when_sdk_installed(self):
        self.assertTrue(groq_instr.patch_groq())

    def test_patch_groq_is_idempotent(self):
        self.assertTrue(groq_instr.patch_groq())
        self.assertTrue(groq_instr.patch_groq())

    def test_patch_marks_wrapped_functions(self):
        groq_instr.patch_groq()
        from groq.resources.chat.completions import Completions, AsyncCompletions

        self.assertTrue(getattr(Completions.create, "_agent_trace_patched", False))
        self.assertTrue(getattr(AsyncCompletions.create, "_agent_trace_patched", False))

    def test_patch_groq_soft_fails_without_sdk(self):
        import sys

        real_groq = sys.modules.get("groq")
        sys.modules["groq"] = None  # forces ImportError on `import groq`
        try:
            self.assertFalse(groq_instr.patch_groq())
        finally:
            if real_groq is not None:
                sys.modules["groq"] = real_groq
            else:
                sys.modules.pop("groq", None)


class TestGroqSpanAttributes(unittest.TestCase):
    """Verify span attributes captured by the patched create() calls."""

    def setUp(self):
        import groq
        from groq.resources.chat.completions import Completions, AsyncCompletions

        # Reset patch state and stub the underlying HTTP-calling methods with
        # fakes *before* patching, so patch_groq() wraps our fakes instead of
        # making real network calls.
        groq_instr._patched = False
        self._orig_sync_create = Completions.create
        self._orig_async_create = AsyncCompletions.create

        def fake_sync_create(self_, *args, **kwargs):
            if kwargs.get("model") == "trigger-error":
                raise RuntimeError("boom")
            if kwargs.get("stream"):
                return _FakeStream()
            return _fake_chat_completion(model=kwargs.get("model"))

        async def fake_async_create(self_, *args, **kwargs):
            if kwargs.get("model") == "trigger-error":
                raise RuntimeError("boom")
            if kwargs.get("stream"):
                return _FakeStream()
            return _fake_chat_completion(model=kwargs.get("model"))

        Completions.create = fake_sync_create
        AsyncCompletions.create = fake_async_create

        self.assertTrue(groq_instr.patch_groq())

        self.provider = TracerProvider()
        set_tracer_provider(self.provider)
        self.capture = _CapturingProcessor()
        self.provider.add_span_processor(self.capture)

        self.client = groq.Groq(api_key="test-key")
        self.async_client = groq.AsyncGroq(api_key="test-key")

    def tearDown(self):
        from groq.resources.chat.completions import Completions, AsyncCompletions

        Completions.create = self._orig_sync_create
        AsyncCompletions.create = self._orig_async_create
        groq_instr._patched = False

    def _last_snapshot(self):
        self.assertEqual(len(self.capture.snapshots), 1)
        return self.capture.snapshots[0]

    def test_sync_success_sets_attributes(self):
        resp = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertEqual(resp.choices[0].message.content, "Hello there!")

        attrs = self._last_snapshot()["attributes"]
        self.assertEqual(attrs["llm.vendor"], "groq")
        self.assertEqual(attrs["llm.model"], "llama-3.3-70b-versatile")
        self.assertEqual(attrs["llm.usage.prompt_tokens"], 12)
        self.assertEqual(attrs["llm.usage.completion_tokens"], 4)
        self.assertEqual(attrs["llm.usage.total_tokens"], 16)
        self.assertEqual(attrs["llm.finish_reason"], "stop")
        self.assertEqual(attrs["llm.completion"], "Hello there!")
        self.assertIn("llm.prompt", attrs)
        self.assertIn("llm.groq.messages", attrs)

    def test_sync_error_records_exception_status(self):
        from traccia.tracer.span import SpanStatus

        with self.assertRaises(RuntimeError):
            self.client.chat.completions.create(
                model="trigger-error",
                messages=[{"role": "user", "content": "hi"}],
            )

        snapshot = self._last_snapshot()
        self.assertEqual(snapshot["status"], SpanStatus.ERROR)
        attrs = snapshot["attributes"]
        self.assertEqual(attrs["llm.vendor"], "groq")
        self.assertEqual(attrs["llm.model"], "trigger-error")

    def test_sync_streaming_does_not_populate_completion(self):
        resp = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        self.assertIsInstance(resp, _FakeStream)

        attrs = self._last_snapshot()["attributes"]
        # Request-side attributes are present...
        self.assertEqual(attrs["llm.vendor"], "groq")
        self.assertEqual(attrs["llm.model"], "llama-3.3-70b-versatile")
        # ...but nothing that requires draining the stream is.
        self.assertNotIn("llm.completion", attrs)
        self.assertNotIn("llm.usage.prompt_tokens", attrs)

    def test_async_success_sets_attributes(self):
        async def _run():
            return await self.async_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "hi"}],
            )

        resp = asyncio.run(_run())
        self.assertEqual(resp.choices[0].message.content, "Hello there!")

        attrs = self._last_snapshot()["attributes"]
        self.assertEqual(attrs["llm.vendor"], "groq")
        self.assertEqual(attrs["llm.model"], "llama-3.3-70b-versatile")
        self.assertEqual(attrs["llm.usage.prompt_tokens"], 12)
        self.assertEqual(attrs["llm.usage.completion_tokens"], 4)
        self.assertEqual(attrs["llm.completion"], "Hello there!")

    def test_async_streaming_does_not_populate_completion(self):
        async def _run():
            return await self.async_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )

        resp = asyncio.run(_run())
        self.assertIsInstance(resp, _FakeStream)

        attrs = self._last_snapshot()["attributes"]
        self.assertNotIn("llm.completion", attrs)
        self.assertNotIn("llm.usage.prompt_tokens", attrs)


if __name__ == "__main__":
    unittest.main()
