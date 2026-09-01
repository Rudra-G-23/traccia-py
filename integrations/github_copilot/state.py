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

try:  # POSIX advisory locking; absent on Windows
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore

_SESSION_ID_RE = re.compile(r"[^A-Za-z0-9._-]")

# Suffix appended to a session log while a flush holds it (see claim_session).
_CLAIM_SUFFIX = ".flushing"


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
    # O_APPEND makes each write seek-to-end first. A single short write() is
    # atomic against concurrent appenders on POSIX only while it stays under
    # PIPE_BUF (4096 on Linux) -- which a record can exceed once
    # capture_content=True inlines size-capped tool/prompt text. Take an
    # exclusive advisory lock for the write so concurrent hook processes for the
    # same session can never interleave a partial line, regardless of size.
    fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError:
                pass  # e.g. some network filesystems; fall back to bare append
        os.write(fd, line)
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def read_events_from_path(path: Path) -> List[Dict[str, Any]]:
    """Parse a JSONL session log at an explicit path (used for claimed logs)."""
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


def read_events(session_id: str, state_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    return read_events_from_path(session_log_path(session_id, state_dir))


def has_end_event(session_id: str, state_dir: Optional[Path] = None) -> bool:
    """True if this session's log already contains a sessionEnd record.

    Cheap scan (no JSON parse) so `flush --all` can tell a genuinely-finished
    session apart from one that's still live.
    """
    path = session_log_path(session_id, state_dir)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return any('"event": "sessionEnd"' in line or '"event":"sessionEnd"' in line for line in fh)
    except OSError:
        return False


def clear_session(session_id: str, state_dir: Optional[Path] = None) -> None:
    path = session_log_path(session_id, state_dir)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def claim_session(session_id: str, state_dir: Optional[Path] = None) -> Optional[Path]:
    """Atomically take exclusive ownership of a session log for flushing.

    The claim marker is ``<id>.jsonl.flushing``, created with ``O_CREAT|O_EXCL``
    -- the kernel guarantees exactly one caller wins that create, so a
    concurrent flush or a re-delivered ``sessionEnd`` gets ``None`` and a
    session is never exported twice. Any pending ``<id>.jsonl`` data is then
    moved into the marker. Returns the claimed path, or ``None`` if someone
    else holds the claim or there was nothing buffered.

    A claim orphaned by a hard-killed flush is *not* stolen here (no liveness
    signal); ``flush_all`` sweeps stale ``.flushing`` files instead -- see
    ``list_stale_claims``.
    """
    src = session_log_path(session_id, state_dir)
    claimed = src.with_name(src.name + _CLAIM_SUFFIX)

    if not src.exists() and not claimed.exists():
        return None

    try:
        fd = os.open(str(claimed), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    except FileExistsError:
        return None  # another flush holds it
    except OSError:
        return None

    # We own the (currently empty) marker. Fold in any buffered data.
    if src.exists():
        try:
            os.replace(src, claimed)  # atomic on the same filesystem
        except OSError:
            pass

    try:
        if claimed.stat().st_size == 0:
            discard_claim(claimed)
            return None
    except OSError:
        return None
    return claimed


def list_stale_claims(
    older_than_seconds: float, state_dir: Optional[Path] = None
) -> List[Path]:
    """``.flushing`` markers left behind by a crashed flush, older than the cutoff."""
    base = Path(state_dir) if state_dir else default_state_dir()
    if not base.exists():
        return []
    now = time.time()
    stale: List[Path] = []
    for p in base.glob(f"*{_CLAIM_SUFFIX}"):
        try:
            if now - p.stat().st_mtime >= older_than_seconds:
                stale.append(p)
        except OSError:
            continue
    return sorted(stale)


def discard_claim(claimed_path: Path) -> None:
    """Delete a successfully-flushed claimed log."""
    try:
        Path(claimed_path).unlink(missing_ok=True)
    except OSError:
        pass


def restore_claim(claimed_path: Path) -> None:
    """Return a claimed log to its normal name so it can be retried later."""
    claimed_path = Path(claimed_path)
    if claimed_path.name.endswith(_CLAIM_SUFFIX):
        original = claimed_path.with_name(claimed_path.name[: -len(_CLAIM_SUFFIX)])
        try:
            os.replace(claimed_path, original)
        except OSError:
            pass


def archive_failed_claim(claimed_path: Path) -> Optional[Path]:
    """Move a claimed log whose export failed into a ``failed/`` sibling dir.

    Keeps the buffered events recoverable (via ``traccia copilot flush
    --retry-failed``) instead of deleting them when the network export didn't
    land. Returns the archived path.
    """
    claimed_path = Path(claimed_path)
    name = claimed_path.name
    if name.endswith(_CLAIM_SUFFIX):
        name = name[: -len(_CLAIM_SUFFIX)]
    failed_dir = claimed_path.parent / "failed"
    try:
        failed_dir.mkdir(parents=True, exist_ok=True)
        dst = failed_dir / f"{Path(name).stem}.{int(time.time())}.jsonl"
        os.replace(claimed_path, dst)
        return dst
    except OSError:
        return None


def list_failed(state_dir: Optional[Path] = None) -> List[Path]:
    base = Path(state_dir) if state_dir else default_state_dir()
    failed_dir = base / "failed"
    if not failed_dir.exists():
        return []
    return sorted(failed_dir.glob("*.jsonl"))


def list_sessions(state_dir: Optional[Path] = None) -> List[str]:
    base = Path(state_dir) if state_dir else default_state_dir()
    if not base.exists():
        return []
    # *.jsonl only -- never picks up "<id>.jsonl.flushing" claimed logs or the
    # failed/ subdirectory (glob is non-recursive).
    return [p.stem for p in base.glob("*.jsonl")]


def session_age_seconds(session_id: str, state_dir: Optional[Path] = None) -> Optional[float]:
    path = session_log_path(session_id, state_dir)
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None
