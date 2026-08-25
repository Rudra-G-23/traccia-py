"""Materialize and export a GitHub Copilot session's buffered hook events.

Two callers:
  - hook.py spawns this detached (`python -m ...flush --session <id>`) when a
    sessionEnd event arrives, so the real network export never blocks a hook
    Copilot is waiting on.
  - `traccia copilot flush` (cli.py) calls flush_session()/flush_all()
    in-process, for orphaned sessions (sessionEnd never arrived) or to flush
    on demand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from traccia.integrations.github_copilot import state


def flush_session(session_id: str, *, state_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Read one session's buffered events, build+export its trace, and clear the log.

    Returns the build summary dict, or None if there was nothing buffered.
    """
    events = state.read_events(session_id, state_dir=state_dir)
    if not events:
        return None

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

    traccia.force_flush()
    if started_here:
        traccia.stop_tracing()

    state.clear_session(session_id, state_dir=state_dir)
    return summary


def flush_all(
    *, max_age_seconds: Optional[float] = None, state_dir: Optional[Path] = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Flush every session currently buffered on disk.

    max_age_seconds, if given, skips sessions whose log was modified more
    recently than that (they're probably still active, not orphaned).
    """
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    for session_id in state.list_sessions(state_dir=state_dir):
        if max_age_seconds is not None:
            age = state.session_age_seconds(session_id, state_dir=state_dir)
            if age is not None and age < max_age_seconds:
                continue
        results[session_id] = flush_session(session_id, state_dir=state_dir)
    return results


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="traccia-copilot-flush")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", help="Flush a single session id")
    group.add_argument("--all", action="store_true", help="Flush every buffered session")
    parser.add_argument(
        "--max-age-seconds",
        type=float,
        default=None,
        help="With --all, only flush sessions untouched for at least this long",
    )
    args = parser.parse_args(argv)

    try:
        if args.session:
            flush_session(args.session)
        else:
            flush_all(max_age_seconds=args.max_age_seconds)
        return 0
    except Exception as exc:
        print(f"traccia copilot flush failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
