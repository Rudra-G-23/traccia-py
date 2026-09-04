"""Materialize and export a GitHub Copilot session's buffered hook events.

Two callers:
  - hook.py spawns this detached (`python -m ...flush --session <id>`) when a
    sessionEnd event arrives, so the real network export never blocks a hook
    Copilot is waiting on.
  - `traccia copilot flush` (cli.py) calls flush_session()/flush_all()
    in-process, for orphaned sessions (sessionEnd never arrived) or to flush
    on demand.

Concurrency + durability:
  - Each session log is *claimed* (atomic rename to ``<id>.jsonl.flushing``)
    before it is read, so a duplicate ``sessionEnd`` or an overlapping
    ``flush --all`` can never export the same session twice.
  - The claimed log is only deleted once the exporter reports a successful
    flush. If the export fails (bad endpoint, offline, timeout) the log is
    moved to a ``failed/`` sibling directory instead of being dropped, and can
    be replayed with ``traccia copilot flush --retry-failed``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from traccia.integrations.github_copilot import mapping, state

# Give the exporter meaningfully longer than the default 5s here: this runs off
# Copilot's blocking path, and a slow-but-eventually-successful flush should not
# be misreported as a failure (which would archive the log for a needless retry).
_FLUSH_TIMEOUT_SECONDS = 15.0

# A `.flushing` claim marker still around after this long belonged to a flush
# that was hard-killed; `flush_all` reclaims it. Well above any realistic export.
_STALE_CLAIM_SECONDS = 3600.0


def _has_session_end(events: List[Dict[str, Any]]) -> bool:
    return any(e.get("event") in mapping.SESSION_END_EVENTS for e in events)


def _export_events(events: List[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], bool]:
    """Build spans for one session's events and flush them. Returns (summary, flush_ok)."""
    import traccia
    from traccia import auto as _auto_mod
    from traccia.integrations.github_copilot import spans as spans_mod

    # If we're running inside a process that already called traccia.init()
    # (e.g. flush_session() imported directly into a long-running app), reuse
    # its already-configured provider instead of standing up (and later
    # tearing down) a second one.
    started_here = not bool(getattr(_auto_mod, "_started", False))
    if started_here:
        traccia.init(
            auto_start_trace=False,
            openai_agents=False,
            crewai=False,
            github_copilot=False,
        )
        # init()'s config-file priority resolution can silently re-enable
        # auto_start_trace (a traccia.toml with no explicit `auto_start_trace`
        # key defaults to true and wins over this call's auto_start_trace=False
        # kwarg -- a pre-existing quirk in how init() merges file config vs.
        # named params). end_auto_trace() is the public, idempotent way to
        # guarantee no ambient "root" span is active as an accidental parent
        # for every span this materializer creates below.
        traccia.end_auto_trace()

    tracer = traccia.get_tracer("github_copilot")
    summary = spans_mod.build_trace(tracer, events)

    force_flush_result = traccia.force_flush(_FLUSH_TIMEOUT_SECONDS)
    # Older/custom providers may return None; preserve compatibility for
    # those providers, while treating an explicit False as a failed export.
    flush_ok = True if force_flush_result is None else bool(force_flush_result)
    if started_here:
        # Shutdown is still useful for releasing resources, but it does not
        # provide a reliable export acknowledgement. Never turn an explicit
        # failed force_flush into success, or the durable journal is deleted.
        traccia.stop_tracing()

    return summary, flush_ok


def flush_session(
    session_id: str, *, state_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Claim one session's buffered events, build+export its trace, then clear it.

    Returns the build summary dict, or None if there was nothing buffered or the
    session was already claimed by another flush.
    """
    claimed = state.claim_session(session_id, state_dir=state_dir)
    if claimed is None:
        return None  # nothing to flush, or a concurrent flush owns it

    events = state.read_events_from_path(claimed)
    initial_fingerprint = state.claim_fingerprint(claimed)
    if not events:
        state.discard_claim(claimed)
        return None

    try:
        summary, flush_ok = _export_events(events)
    except BaseException:
        # Never lose the buffer because span-building raised -- put it back for
        # a later retry and re-raise so the CLI surfaces the error.
        state.restore_claim(claimed)
        raise

    # A hook can finish writing while export is in flight. Keep the journal if
    # it changed so the late event is never silently discarded.
    if flush_ok and state.claim_unchanged(claimed, initial_fingerprint):
        state.discard_claim(claimed)
    else:
        archived = state.archive_failed_claim(claimed)
        print(
            f"traccia copilot flush: export for session {session_id} did not "
            f"confirm; buffered events kept at {archived or claimed} "
            f"(retry with `traccia copilot flush --retry-failed`)",
            file=sys.stderr,
        )
    return summary


def flush_all(
    *,
    max_age_seconds: Optional[float] = None,
    include_active: bool = False,
    state_dir: Optional[Path] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Flush buffered sessions.

    A session is flushed when:
      - its log contains a ``sessionEnd`` record (genuinely finished), or
      - ``include_active`` is True (force everything), or
      - ``max_age_seconds`` is given and the log has been idle at least that
        long (orphan recovery -- ended abnormally, no ``sessionEnd``).

    Without any of those, a session is left untouched: a still-live Copilot
    session must not be exported as a premature partial trace whose buffer then
    gets deleted (the rest of that session would later materialize as a second,
    disjoint trace).
    """
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    for session_id in state.list_sessions(state_dir=state_dir):
        ended = state.has_end_event(session_id, state_dir=state_dir)
        old_enough = False
        if max_age_seconds is not None:
            age = state.session_age_seconds(session_id, state_dir=state_dir)
            old_enough = age is not None and age >= max_age_seconds

        if not (ended or include_active or old_enough):
            continue

        results[session_id] = flush_session(session_id, state_dir=state_dir)

    # Reclaim claims stranded by a hard-killed flush so their events aren't lost.
    for stale in state.list_stale_claims(_STALE_CLAIM_SECONDS, state_dir=state_dir):
        events = state.read_events_from_path(stale)
        if not events:
            state.discard_claim(stale)
            continue
        try:
            _summary, flush_ok = _export_events(events)
        except BaseException as exc:  # noqa: BLE001 - keep going through the backlog
            print(f"traccia copilot flush: stale claim {stale.name}: {exc}", file=sys.stderr)
            continue
        if flush_ok:
            state.discard_claim(stale)
        else:
            state.archive_failed_claim(stale)
        results[stale.name] = _summary
    return results


def retry_failed(*, state_dir: Optional[Path] = None) -> Dict[str, Optional[Dict[str, Any]]]:
    """Re-attempt export for every log parked in the ``failed/`` directory."""
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    for path in state.list_failed(state_dir=state_dir):
        events = state.read_events_from_path(path)
        if not events:
            state.discard_claim(path)
            continue
        try:
            summary, flush_ok = _export_events(events)
        except BaseException as exc:  # keep going through the rest of the backlog
            results[path.name] = None
            print(f"traccia copilot flush --retry-failed: {path.name}: {exc}", file=sys.stderr)
            continue
        if flush_ok:
            state.discard_claim(path)
        results[path.name] = summary
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="traccia-copilot-flush")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", help="Flush a single session id")
    group.add_argument("--all", action="store_true", help="Flush every eligible buffered session")
    group.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-attempt export for sessions parked in failed/ after a prior export error",
    )
    parser.add_argument(
        "--max-age-seconds",
        type=float,
        default=None,
        help="With --all, also flush sessions with no sessionEnd that have been idle at least this long",
    )
    parser.add_argument(
        "--include-active",
        action="store_true",
        help="With --all, also flush sessions that appear still active (no sessionEnd yet)",
    )
    args = parser.parse_args(argv)

    try:
        if args.session:
            flush_session(args.session)
        elif args.retry_failed:
            retry_failed()
        else:
            flush_all(
                max_age_seconds=args.max_age_seconds,
                include_active=args.include_active,
            )
        return 0
    except Exception as exc:
        print(f"traccia copilot flush failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
