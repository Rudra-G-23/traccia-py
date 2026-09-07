"""@govern inherits agent_id from init; decorator arg is an override only."""

from __future__ import annotations

from unittest.mock import patch

from traccia import runtime_config
from traccia.governance.govern import govern


def test_govern_inherits_agent_id_from_init():
    runtime_config.set_agent_id("from-init")
    try:
        with patch("traccia.governance.govern.check_agent_status") as check:
            @govern(fail_open=False, name="run")
            def run() -> str:
                return "ok"

            assert run() == "ok"
            check.assert_called_once_with("from-init", fail_open=False)
    finally:
        runtime_config.set_agent_id(None)


def test_govern_explicit_agent_id_overrides_init():
    runtime_config.set_agent_id("from-init")
    try:
        with patch("traccia.governance.govern.check_agent_status") as check:
            @govern(agent_id="override", fail_open=True, name="run")
            def run() -> str:
                return "ok"

            assert run() == "ok"
            check.assert_called_once_with("override", fail_open=True)
    finally:
        runtime_config.set_agent_id(None)


def test_govern_falls_back_to_env(monkeypatch):
    runtime_config.set_agent_id(None)
    monkeypatch.setenv("TRACCIA_AGENT_ID", "from-env")
    with patch("traccia.governance.govern.check_agent_status") as check:
        @govern(name="run")
        def run() -> str:
            return "ok"

        assert run() == "ok"
        check.assert_called_once_with("from-env", fail_open=True)


def test_govern_warns_when_identity_missing(monkeypatch, caplog):
    runtime_config.set_agent_id(None)
    monkeypatch.delenv("TRACCIA_AGENT_ID", raising=False)
    with patch("traccia.governance.govern.check_agent_status") as check:
        @govern(name="run")
        def run() -> str:
            return "ok"

        assert run() == "ok"
        check.assert_not_called()
    assert "Skipping policy check" in caplog.text
