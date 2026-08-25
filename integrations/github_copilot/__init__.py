"""Traccia integration for GitHub Copilot (CLI + cloud coding agent hooks).

Unlike the LangChain/CrewAI/OpenAI-Agents integrations, there is no Python
object to patch here -- GitHub Copilot runs as an external, non-Python
process. Tracing is instead driven by Copilot's own hooks mechanism: run
``traccia copilot install-hooks`` once to register a Traccia hook script with
Copilot, then hook events flow through
``hook.py -> state.py -> spans.py`` (materialized once per session, on
sessionEnd, via ``flush.py``) and are exported through Traccia's normal,
already-configured exporter pipeline.

See docs/github-copilot-integration.md for the full design, the exact hook
event -> span mapping, and known limitations (notably: no token/model/
generation-level data -- Copilot's hooks don't expose it, only its native
OTLP stream does, and this SDK has no ingestion endpoint to receive that).
"""
from typing import Optional

_installed = False
_capture_content = False


def install(enabled: Optional[bool] = None, capture_content: Optional[bool] = None) -> bool:
    """
    Enable the GitHub Copilot hooks integration.

    Note this does NOT modify your repository or Copilot's own configuration --
    it only marks the integration enabled/disabled in this process (surfaced by
    ``traccia doctor``). The hook script Copilot actually invokes runs in a
    separate OS process each time and always re-reads config fresh from
    ``traccia.toml``/env vars, since it has no visibility into this process's
    ``init()`` call.

    To wire Copilot up to Traccia, run the CLI command once per repo (or once
    for your user profile for the Copilot CLI):

        traccia copilot install-hooks

    Args:
        enabled: If False, disable. If None, check config (default: enabled).
        capture_content: If True, capture (redacted) tool/prompt content
            instead of length-only metadata. If None, check config
            (default: False -- metadata only).

    Returns:
        True if enabled, False otherwise.
    """
    global _installed, _capture_content

    if enabled is False:
        _installed = False
        return False

    if enabled is None:
        from traccia import runtime_config
        if runtime_config.get_config_value("github_copilot") is False:
            _installed = False
            return False

    if capture_content is not None:
        _capture_content = bool(capture_content)

    _installed = True
    return True


__all__ = ["install"]
