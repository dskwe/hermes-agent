"""A STANDALONE gateway's runtime record carries bare platform keys and ``served_profiles: []``
(``gateway.run_adapters`` publishes exactly that on a single-profile run). Routed through the
multiplexer rung of the liveness ladder (launchd-managed: no ``gateway.pid``), the record was
re-keyed as a multiplexer record and projected to ``{}``: ``/api/status`` reported
``gateway_platforms: {}`` and the Messaging card badged a connected platform "Restart needed"
forever (#123869). A record with no served roster already IS the profile's platform map — it must
pass through unchanged; only a record that actually serves profiles gets the namespaced re-key.
"""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture
def standalone_root(tmp_path, monkeypatch):
    """Default home of a launchd-managed STANDALONE gateway: live record, no ``gateway.pid``, and
    the rung-1/3 probes answer None exactly as on the reported host (no pid file, launcher argv
    proves nothing) — while the multiplexer rung still answers from the record's own live pid and
    empty roster, which is the routing that exposed the projection bug."""
    root = tmp_path / "hermes"
    root.mkdir(parents=True)
    (root / "gateway_state.json").write_text(json.dumps({
        "pid": os.getpid(), "hermes_home": str(root), "gateway_state": "running",
        "served_profiles": [],
        "platforms": {"feishu": {"state": "connected"}},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    import gateway.status as status
    import hermes_cli.gateway_multiplex_served as multiplex_served
    from hermes_cli.web_routers import messaging
    for mod in (status, messaging):
        monkeypatch.setattr(mod, "get_running_pid_cached", lambda *a, **k: None)
        monkeypatch.setattr(mod, "get_runtime_status_running_pid", lambda *a, **k: None)
    monkeypatch.setattr(multiplex_served, "live_default_gateway_pid", lambda *a, **k: os.getpid())
    return root


def test_standalone_record_projects_its_bare_platforms():
    from gateway.status import profile_platforms_from_multiplexer
    standalone = {"gateway_state": "running", "served_profiles": [],
                  "platforms": {"feishu": {"state": "connected"}}}
    assert profile_platforms_from_multiplexer(standalone, "default") == {"feishu": {"state": "connected"}}
    # A pre-multiplex record carries no roster key at all — same passthrough.
    legacy = {"platforms": {"telegram": {"state": "retrying"}}}
    assert profile_platforms_from_multiplexer(legacy, "default") == {"telegram": {"state": "retrying"}}
    # A record that actually serves profiles keeps the ``<profile>:`` re-key.
    mux = {"served_profiles": ["default", "alpha"],
           "platforms": {"alpha:telegram": {"state": "connected"}}}
    assert profile_platforms_from_multiplexer(mux, "alpha")["telegram"] == {"state": "connected"}


def test_messaging_card_for_a_standalone_default_gateway_not_restart_needed(standalone_root, monkeypatch):
    """The reported surface: the default home's own record reaches the multiplexer rung, the card
    must read the platform's live state — never ``pending_restart`` while the bot answers."""
    from hermes_cli.web_routers import messaging
    monkeypatch.setattr(messaging, "_platform_enablement", lambda *a, **k: (True, True, None))
    entry = {"id": "feishu", "name": "Feishu", "description": "", "docs_url": "", "env_vars": [],
             "required_env": []}
    [payload] = messaging._platform_payloads(None, [entry])
    assert payload["gateway_running"] is True
    assert payload["state"] == "connected", payload
