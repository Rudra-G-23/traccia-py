from traccia.instrumentation.requests import _should_skip_http_instrumentation


def test_skips_agent_status_and_block_urls():
    assert _should_skip_http_instrumentation(
        "http://localhost:8000/api/v1/agents/policy-tool-storm-smoke/status"
    )
    assert _should_skip_http_instrumentation(
        "http://localhost:8000/api/v1/agents/policy-tool-storm-smoke/blocks"
    )
    assert _should_skip_http_instrumentation(
        "https://custom.example/agents/agent-x/status"
    )
    assert _should_skip_http_instrumentation(
        "https://custom.example/agents/agent-x/blocks"
    )


def test_skips_platform_bookkeeping_urls():
    assert _should_skip_http_instrumentation(
        "https://api.traccia.ai/api/v1/eval-runtime/score"
    )
    assert _should_skip_http_instrumentation(
        "http://localhost:8001/api/v1/prompt-runtime/prompts/support-reply"
    )
    assert _should_skip_http_instrumentation("https://api.traccia.ai/v2/traces")
    assert _should_skip_http_instrumentation("https://app.traccia.ai/api/v1/policy/check")


def test_does_not_skip_ordinary_http():
    assert not _should_skip_http_instrumentation("https://api.github.com/repos/traccia/x")
    assert not _should_skip_http_instrumentation("http://localhost:8000/health")
