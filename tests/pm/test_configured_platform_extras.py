"""Configured gateway platforms join the dependency environment (#124228).

A home with platforms.telegram enabled must not update or first-generate into
a venv without python-telegram-bot. Covers: the explicit-install selection
union, the engine's first-generation selection, the legacy-venv migration's
$HERMES_HOME/venvs layout, and the registry hint agreeing with pm's command.

Network-free; the config seam (load_config_readonly) is stubbed at its module.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _stub_config(monkeypatch, config: dict) -> None:
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: config)


# ---- pm.platform_features: config -> extras ----


def test_enabled_platform_unions_into_explicit_selection(monkeypatch):
    import pm.platform_features as pf

    _stub_config(monkeypatch, {"platforms": {"telegram": {"enabled": True}}})
    assert pf.configured_platform_extras() == ["telegram"]
    assert pf.add_configured_platform_extras(["all"]) == ["all", "telegram"]
    # An explicit user request is kept, never replaced.
    assert pf.add_configured_platform_extras(["fal", "all"]) == ["all", "fal", "telegram"]


def test_gateway_nesting_and_underscore_ids_map(monkeypatch):
    import pm.platform_features as pf

    _stub_config(monkeypatch, {"gateway": {"platforms": {"google_chat": {"enabled": True}}}})
    assert pf.configured_platform_extras() == ["google-chat"]


def test_disabled_unmapped_and_undeclared_platforms_contribute_nothing(monkeypatch):
    import pm.platform_features as pf

    _stub_config(monkeypatch, {"platforms": {
        "telegram": {"enabled": False},   # configured off
        "email": {"enabled": True},       # stdlib/aiohttp: no extra to map
        "buzz": {"enabled": True},        # no matching pyproject extra
    }})
    assert pf.configured_platform_extras() == []


def test_no_configured_platforms_keeps_selection_semantics(monkeypatch):
    import pm.platform_features as pf

    _stub_config(monkeypatch, {})
    # None stays None: sync_venv() keeps its keep-what-is-recorded meaning.
    assert pf.add_configured_platform_extras(None) is None
    assert pf.add_configured_platform_extras(["all"]) == ["all"]


# ---- engine: first PM generation selection ----


def _fake_package():
    return SimpleNamespace(expected_stamp=lambda enabled, **kwargs: f"stamp:{','.join(enabled)}")


def test_first_generation_selection_unions_configured_platforms(monkeypatch):
    from pm import install as engine

    monkeypatch.setattr(engine, "_still_declared", lambda package, recorded: list(recorded))
    import pm.platform_features as pf
    monkeypatch.setattr(pf, "configured_platform_extras", lambda: ["telegram"])

    enabled, stamp, inputs = engine._target_selection(
        _fake_package(), {}, extras=None, inputs={}, repair=False, shipped=None, frozen=None)

    # Bundle install (shipped features = [all]) with Telegram configured:
    # the generation the update builds must carry the SDK.
    enabled, stamp, inputs = engine._target_selection(
        _fake_package(), {}, extras=None, inputs={}, repair=False, shipped=["all"], frozen=None)

    assert enabled == ["all", "telegram"]
    assert stamp == "stamp:all,telegram"
    assert inputs == {"configured_platform_extras": True}

    # No recorded baseline at all (crash before the first record): the
    # configured platform is still carried instead of lost with the baseline.
    enabled, _stamp, _inputs = engine._target_selection(
        _fake_package(), {}, extras=None, inputs={}, repair=False, shipped=None, frozen=None)
    assert enabled == ["telegram"]


def test_explicit_selection_still_wins_over_recorded_first_generation(monkeypatch):
    from pm import install as engine

    monkeypatch.setattr(engine, "_still_declared", lambda package, recorded: list(recorded))
    import pm.platform_features as pf
    monkeypatch.setattr(pf, "configured_platform_extras", lambda: ["telegram"])

    enabled, _stamp, _inputs = engine._target_selection(
        _fake_package(), {"extras": ["discord"]}, extras=["slack"], inputs={}, repair=False,
        shipped=None, frozen=None)

    assert enabled == ["discord", "slack", "telegram"]


# ---- install command path ----


def test_pm_install_unions_configured_platform_extras(monkeypatch, capsys):
    from pm import cli, runtime
    from pm import install as install_mod

    synced = []
    monkeypatch.setattr(runtime, "is_runtime", lambda: True)
    monkeypatch.setattr(install_mod, "sync_venv", lambda extras, **kwargs: synced.append(list(extras)))
    monkeypatch.setattr(install_mod, "activate", lambda **kwargs: [])
    monkeypatch.setattr(cli, "_install_names", lambda names, target=None, **kwargs: 0)
    import pm.platform_features as pf
    monkeypatch.setattr(pf, "configured_platform_extras", lambda: ["telegram"])

    failed = cli._install_python_environments([], sync=True, test_environment=None)

    assert failed == 0
    assert synced == [["all", "telegram"]]


# ---- migration: $HERMES_HOME/venvs/<name> layout ----


def test_legacy_selection_scans_home_venvs_layout(monkeypatch, tmp_path):
    import pm.extras as extras

    monkeypatch.setattr(extras, "_PLATFORM_GATES", {})
    home = tmp_path / "home"
    site = home / "venvs" / "agent" / "lib" / "python3.11" / "site-packages"
    (site / "discord").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    selection = extras.legacy_selection(tmp_path / "checkout")

    assert selection[0] == "all"
    assert "discord" in selection


def test_legacy_selection_without_any_venv_is_all(monkeypatch, tmp_path):
    import pm.extras as extras

    monkeypatch.setattr(extras, "_PLATFORM_GATES", {})
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))

    assert extras.legacy_selection(tmp_path / "checkout") == ["all"]


# ---- registry hint agrees with pm's command ----


def test_telegram_install_hint_names_the_pm_command(monkeypatch):
    from pm.extras import install_hint as pm_install_hint

    class Ctx:
        def register_platform(self, **kwargs):
            self.kwargs = kwargs

    from plugins.platforms.telegram import adapter

    ctx = Ctx()
    adapter.register(ctx)

    assert pm_install_hint("telegram") in ctx.kwargs["install_hint"]
