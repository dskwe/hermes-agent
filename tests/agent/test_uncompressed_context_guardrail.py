"""Uncompressed context overflow guardrail (#89297).

When compression is explicitly disabled (``compression.enabled: false``),
sessions can grow past the model context window with nothing to shrink them.
The conversation loop's pre-API site warns (deduped, actionable); the
turn-context preflight re-arms the dedup once the session is back under the
window so a later re-overflow warns again.

The fake binds the PRODUCTION ``_warn_uncompressed_context_overflow`` /
``_clear_context_overflow_warn`` methods so their dedup logic is actually
under test (not a reimplementation).
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

from agent.turn_context import TurnContext, build_turn_context  # noqa: F401
from run_agent import AIAgent
from tests.agent.test_turn_context import _FakeAgent, _build

class _FakeUncompressedAgent(_FakeAgent):
    """Agent stub with compression disabled, bound to the REAL warn methods."""

    # Production methods under test — bound from AIAgent so the dedup key
    # handling and message text cannot silently drift from what ships.
    _warn_uncompressed_context_overflow = (
        AIAgent._warn_uncompressed_context_overflow
    )
    _clear_context_overflow_warn = AIAgent._clear_context_overflow_warn

    def __init__(self, model="deepseek-v4-flash", context_length=10_000,
                 config_enabled: bool = False):
        super().__init__()
        self.model = model
        self.provider = "deepseek"
        self.compression_enabled = False
        # What config.yaml said (agent/agent_init.py sets this alongside
        # ``compression_enabled``). False = the user disabled compression in
        # config; True = config left it on and something else (a host
        # integration) disabled it programmatically (#123500).
        self.compression_config_enabled = config_enabled
        self.context_compressor = types.SimpleNamespace(
            protect_first_n=2,
            protect_last_n=2,
            context_length=context_length,
            threshold_tokens=int(context_length * 0.75),
            last_prompt_tokens=-1,
        )

def _oversized_history(n_turns: int = 10) -> list:
    large_turn = "Large context content " * 500  # ~2,500 tokens each
    history = []
    for i in range(n_turns):
        history.append({"role": "user", "content": f"Turn {i}: {large_turn}"})
        history.append({"role": "assistant", "content": f"Reply {i}: {large_turn}"})
    return history

def test_production_warn_emits_once_and_dedups():
    """The real method warns once, then dedups identical overflows."""
    agent = _FakeUncompressedAgent(context_length=10_000)
    agent._emit_warning = MagicMock()

    agent._warn_uncompressed_context_overflow(15_000, 10_000)
    agent._warn_uncompressed_context_overflow(16_000, 10_000)

    agent._emit_warning.assert_called_once()
    msg = agent._emit_warning.call_args[0][0]
    assert "compression.enabled: false" in msg
    assert "10,000 tokens" in msg

def test_production_warn_config_disabled_copy_is_actionable():
    """Config-disabled (compression.enabled: false): the warning names the flag
    and points at config.yaml — the only place that can fix it."""
    agent = _FakeUncompressedAgent(context_length=10_000, config_enabled=False)
    agent._emit_warning = MagicMock()

    agent._warn_uncompressed_context_overflow(15_000, 10_000)

    msg = agent._emit_warning.call_args[0][0]
    assert "compression.enabled: false" in msg
    assert "enable compression in config.yaml" in msg

def test_production_warn_host_managed_copy_does_not_blame_config():
    """Host integrations disable compression programmatically after init
    (#123500) while config.yaml still says enabled: true. The warning must not
    claim ``compression.enabled: false`` nor send the user to config.yaml."""
    agent = _FakeUncompressedAgent(context_length=10_000, config_enabled=True)
    agent._emit_warning = MagicMock()

    agent._warn_uncompressed_context_overflow(15_000, 10_000)

    msg = agent._emit_warning.call_args[0][0]
    assert "compression.enabled: false" not in msg
    assert "config.yaml" not in msg
    assert "host application" in msg
    assert "10,000 tokens" in msg
    # The core's manual path is offered in both cases.
    assert "/compact" in msg

def test_production_warn_host_managed_dedups_too():
    """The host-managed variant dedups on the same key — no per-turn spam."""
    agent = _FakeUncompressedAgent(context_length=10_000, config_enabled=True)
    agent._emit_warning = MagicMock()

    agent._warn_uncompressed_context_overflow(15_000, 10_000)
    agent._warn_uncompressed_context_overflow(16_000, 10_000)

    agent._emit_warning.assert_called_once()

def test_clear_rearms_the_warning():
    """After _clear_context_overflow_warn (session back under the window),
    a later re-overflow warns again."""
    agent = _FakeUncompressedAgent(context_length=10_000)
    agent._emit_warning = MagicMock()

    agent._warn_uncompressed_context_overflow(15_000, 10_000)
    agent._clear_context_overflow_warn()
    agent._warn_uncompressed_context_overflow(15_500, 10_000)

    assert agent._emit_warning.call_count == 2

def test_uncompressed_session_within_limits_emits_no_warning():
    agent = _FakeUncompressedAgent(context_length=128_000)
    agent._emit_warning = MagicMock()
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    tctx = _build(agent, conversation_history=history)
    assert isinstance(tctx, TurnContext)
    agent._emit_warning.assert_not_called()

def test_preflight_rearm_clears_dedup_when_back_under_window():
    """The turn-context preflight re-arms the dedup once the session fits
    again (e.g. after a manual /compress), so growth past the window later
    warns a second time."""
    agent = _FakeUncompressedAgent(context_length=128_000)
    agent._emit_warning = MagicMock()
    # Simulate a previously fired warning.
    agent._last_ctx_overflow_warn = ("uncompressed_ctx_overflow", 128_000)

    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    tctx = _build(agent, conversation_history=history)
    assert isinstance(tctx, TurnContext)

    # Preflight cleared the dedup — the next overflow warns again.
    assert agent._last_ctx_overflow_warn is None
    agent._warn_uncompressed_context_overflow(200_000, 128_000)
    agent._emit_warning.assert_called_once()

def test_preflight_does_not_rearm_while_still_over_window():
    """While the session is still over the window, the dedup must survive
    the preflight (no per-turn warn spam)."""
    agent = _FakeUncompressedAgent(context_length=10_000)
    agent._emit_warning = MagicMock()
    agent._last_ctx_overflow_warn = ("uncompressed_ctx_overflow", 10_000)

    tctx = _build(agent, conversation_history=_oversized_history())
    assert isinstance(tctx, TurnContext)

    assert agent._last_ctx_overflow_warn == ("uncompressed_ctx_overflow", 10_000)
    agent._emit_warning.assert_not_called()

def test_multimodal_content_forces_real_estimate_in_rearm_gate():
    """List (multimodal) content defeats a char count; the pre-check must
    treat it as over-gate so the real estimator decides. A tiny multimodal
    session is still under the window, so the dedup is re-armed."""
    agent = _FakeUncompressedAgent(context_length=128_000)
    agent._emit_warning = MagicMock()
    agent._last_ctx_overflow_warn = ("uncompressed_ctx_overflow", 128_000)

    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
        {"role": "assistant", "content": "looking"},
    ]
    tctx = _build(agent, conversation_history=history)
    assert isinstance(tctx, TurnContext)
    assert agent._last_ctx_overflow_warn is None


def test_compression_disabled_copy_variants():
    """Contract between the two overflow-recovery copy variants: the config
    variant may blame settings; the host variant must not (#123500)."""
    from agent.turn_failure_copy import site_copy
    cfg = site_copy("compression_disabled", model="m")
    host = site_copy("compression_disabled_host", model="m")
    assert "compression.enabled" in cfg
    assert "your settings" in cfg
    assert "compression.enabled" not in host
    assert "your settings" not in host
    assert "host application" in host
    # Both offer the same three remedies.
    for msg in (cfg, host):
        assert "/compress" in msg and "/new" in msg and "bigger context window" in msg
