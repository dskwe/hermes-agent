"""Kanban dashboard aux-LLM routes run with the launch profile's secret scope bound.

Regression for #123372: Decompose/Specify (and the same-file Estimate / Describe-auto
siblings) resolve their provider key through the profile secret scope
(``agent.auxiliary_client.call_llm`` → ``get_secret``), but nothing on the route bound
one — ``_with_board_pinned`` pins only the board. Under multi-profile hosting
(``set_multiplex_active(True)``, post-#119279 fail-closed semantics) the aux call raised
``UnscopedSecretError``, which ``_call_aux`` surfaces as
``ok=False / "LLM error: UnscopedSecretError"``; the CLI path worked because it runs
scope-bound. The fix enters ``launch_profile_scope_if_multiplexed`` around the aux work.

Driven through the real route handlers over a bare FastAPI app with the real scope
machinery and the real ``call_llm``; only the OpenAI-compatible response is stubbed, and
the credential is read exactly the way ``call_llm``'s key resolution reads it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.secret_scope import UnscopedSecretError, get_secret, set_multiplex_active
from hermes_cli import kanban_db as kb

# Read by the aux call the same way call_llm's provider-key resolution reads a key:
# ~/.hermes/.env first (via load_env()), then os.environ through the secret scope.
KEY_ENV = "KANBAN_AUX_SCOPE_PROBE"
KEY_VALUE = "launch-dotenv-key"

_PLUGIN_FILE = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"


def _load_plugin_module():
    """Load plugins/kanban/dashboard/plugin_api.py fresh (same harness as the plugin tests)."""
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_aux_scope_test", _PLUGIN_FILE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_llm(seen: dict, content: dict):
    """Stand-in auxiliary client: read the provider key like call_llm does, return ``content``."""

    def _call(**_kwargs):
        seen["key"] = get_secret(KEY_ENV)
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps(content)
        return resp

    return _call


@pytest.fixture
def env(tmp_path, monkeypatch, _isolate_hermes_home):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # The launch profile's own credential source: what a scoped read must return.
    (home / ".env").write_text(f"{KEY_ENV}={KEY_VALUE}\n", encoding="utf-8")
    kb.init_db()
    return home


@pytest.fixture
def multiplexed():
    """The dashboard hosting a profile other than its launch one (the reported trigger)."""
    set_multiplex_active(True)
    try:
        yield
    finally:
        set_multiplex_active(False)


def _client(env):
    mod = _load_plugin_module()
    app = FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/kanban")
    return TestClient(app), mod


def _triage_task(client) -> dict:
    task = client.post("/api/plugins/kanban/tasks", json={"title": "one-liner", "triage": True}).json()["task"]
    assert task["status"] == "triage"
    return task


def _assert_scope_restored():
    """After the request the context must be unbound again (threadpool reuse safety)."""
    try:
        get_secret(KEY_ENV)
    except UnscopedSecretError:
        return
    raise AssertionError("secret scope leaked past the request")


def test_specify_resolves_provider_key_in_launch_scope(env, multiplexed, monkeypatch):
    client, _mod = _client(env)
    task = _triage_task(client)
    seen: dict = {}
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_llm(seen, {"title": "Polished", "body": "b"}))

    r = client.post(f"/api/plugins/kanban/tasks/{task['id']}/specify", json={"author": "ui-tester"})

    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True, r.json()
    assert seen["key"] == KEY_VALUE  # the launch profile's .env, via the bound scope
    _assert_scope_restored()


def test_decompose_resolves_provider_key_in_launch_scope(env, multiplexed, monkeypatch):
    client, _mod = _client(env)
    task = _triage_task(client)
    seen: dict = {}
    content = {"fanout": False, "title": "Spec'd", "body": "b"}
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_llm(seen, content))

    r = client.post(f"/api/plugins/kanban/tasks/{task['id']}/decompose", json={"author": "ui-tester"})

    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True, r.json()
    assert seen["key"] == KEY_VALUE
    _assert_scope_restored()


def test_estimate_resolves_provider_key_in_launch_scope(env, multiplexed, monkeypatch):
    client, _mod = _client(env)
    seen: dict = {}
    content = {"est_tokens": 1200, "complexity": "M", "rationale": "few files"}
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_llm(seen, content))

    r = client.post("/api/plugins/kanban/estimate", json={"title": "t", "body": "b"})

    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True, r.json()
    assert seen["key"] == KEY_VALUE
    _assert_scope_restored()


def test_describe_auto_resolves_provider_key_in_launch_scope(env, multiplexed, monkeypatch):
    client, _mod = _client(env)
    seen: dict = {}
    content = {"description": "does kanban things"}
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_llm(seen, content))

    r = client.post("/api/plugins/kanban/profiles/default/describe-auto", json={"overwrite": False})

    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True, r.json()
    assert seen["key"] == KEY_VALUE
    _assert_scope_restored()
