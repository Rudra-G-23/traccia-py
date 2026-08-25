"""Local per-session event log for GitHub Copilot hook events.

Each hook invocation is a fresh OS process, so span state cannot live in
memory across events -- it is buffered here as JSON Lines and replayed once,
at sessionEnd (or via `traccia copilot flush` for orphaned sessions), by
spans.build_trace(). Keeping each hook invocation to a local disk append (no
network call) is what keeps preToolUse/postToolUse from adding export latency
to a tool call Copilot is waiting on -- see docs/github-copilot-integration.md.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_SESSION_ID_RE = re.compile(r"[^A-Za-z0-9._-]")


def default_state_dir() -> Path:
    # hook.py/flush.py run in a fresh subprocess that never calls
    # traccia.init()/start_tracing(), so runtime_config's in-process globals
    # are never populated here -- read straight from traccia.toml/env instead,
    # exactly like hook.py does for its own enabled/capture_content lookup.
    try:
        from traccia import config as sdk_config

        configured = sdk_config.load_config().instrumentation.github_copilot_state_dir
    except Exception:
        configured = None
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".traccia" / "copilot" / "sessions"


def _safe_session_filename(session_id: str) -> str:
    # session_id is attacker/tool-controlled input read from stdin -- never
    # build a filesystem path from it without sanitizing first.
    cleaned = _SESSION_ID_RE.sub("_", str(session_id))[:200]
    return f"{cleaned or 'unknown'}.jsonl"


def session_log_path(session_id: str, state_dir: Optional[Path] = None) -> Path:
    base = state_dir or default_state_dir()
    return Path(base) / _safe_session_filename(session_id)


def append_event(
    session_id: str,
    event_name: str,
    payload: Dict[str, Any],
    state_dir: Optional[Path] = None,
) -> None:
    """Append one event record to this session's log (creates the file/dir if needed)."""
    path = session_log_path(session_id, state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event_name, "payload": payload, "received_at": time.time()}
    line = (json.dumps(record, default=str) + "\n").encode("utf-8")
    # A single write() of a short line is atomic against concurrent appenders
    # on POSIX (well under PIPE_BUF); O_APPEND makes each write seek-to-end first.
    fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def read_events(session_id: str, state_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = session_log_path(session_id, state_dir)
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a partially-written last line
    return events


def clear_session(session_id: str, state_dir: Optional[Path] = None) -> None:
    path = session_log_path(session_id, state_dir)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def list_sessions(state_dir: Optional[Path] = None) -> List[str]:
    base = Path(state_dir) if state_dir else default_state_dir()
    if not base.exists():
        return []
    return [p.stem for p in base.glob("*.jsonl")]


def session_age_seconds(session_id: str, state_dir: Optional[Path] = None) -> Optional[float]:
    path = session_log_path(session_id, state_dir)
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None
