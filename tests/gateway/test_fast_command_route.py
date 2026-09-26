"""Regression: /fast (gateway and CLI) must only be offered where the route accepts
the fast/priority params — same gate the request builders use (``resolve_fast_mode_overrides``).

A fast-capable model id behind OpenRouter, a custom proxy, or a local llama.cpp server
sends no fast params on the wire, so the toggle must report "not supported" instead of
silently doing nothing (parity with session.info, fixed in fd602278c7).
"""

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    return runner


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1"),
        message_id="m1",
    )


@pytest.mark.asyncio
async def test_gateway_fast_refused_on_unbilled_route(monkeypatch, tmp_path):
    """gpt-5.4 served through OpenRouter: /fast must say not-supported, not open the picker."""
    _install_fake_agent(monkeypatch)
    runner = _make_runner()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda config_path=None: {
        "model": {"default": "gpt-5.4", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"}})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")

    response = await runner._handle_fast_command(_make_event("/fast"))

    assert "only available" in (response or "")  # gateway.fast.not_supported copy
    assert runner._service_tier is None  # nothing was applied


@pytest.mark.asyncio
async def test_gateway_fast_refused_on_local_route(monkeypatch, tmp_path):
    """A fast-capable id served by a local llama.cpp proxy must also refuse."""
    _install_fake_agent(monkeypatch)
    runner = _make_runner()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda config_path=None: {
        "model": {"default": "gpt-5.4", "provider": "llamacpp", "base_url": "http://127.0.0.1:18434/v1"}})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")

    response = await runner._handle_fast_command(_make_event("/fast"))

    assert "only available" in response


@pytest.mark.asyncio
async def test_gateway_fast_still_offered_on_first_party_route(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda config_path=None: {
        "model": {"default": "gpt-5.4", "provider": "openai", "base_url": "https://api.openai.com/v1"}})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")

    response = await runner._handle_fast_command(_make_event("/fast fast"))

    assert "FAST" in (response or "")
    assert runner._service_tier == "priority"


def test_cli_fast_unavailable_on_proxied_route():
    """CLI availability gate: fast-capable model behind OpenRouter must not offer /fast."""
    from hermes_cli.cli_info_mixin import CLIInfoMixin

    cli = SimpleNamespace(model="gpt-5.4", provider="openrouter", base_url="https://openrouter.ai/api/v1")
    assert CLIInfoMixin._fast_command_available(cli) is False


def test_cli_fast_available_on_first_party_route():
    from hermes_cli.cli_info_mixin import CLIInfoMixin

    cli = SimpleNamespace(model="gpt-5.4", provider="openai", base_url="https://api.openai.com/v1")
    assert CLIInfoMixin._fast_command_available(cli) is True
