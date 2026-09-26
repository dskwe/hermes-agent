"""Configured gateway platforms → the extras their SDKs need.

``hermes pm install`` and the updater build the environment from the recorded
selection (or ``[all]``), which deliberately excludes the messaging extras —
so a home configured for Telegram could update into a venv without
python-telegram-bot and cold-start with "No adapter available". The fix is at
the source: an explicit sync unions the selection with the extras of every
platform the config actually enables. The platform id IS the extra name
(``platforms.telegram`` → ``telegram``, ``google_chat`` → ``google-chat``);
a platform with no matching extra (email, irc, ntfy — stdlib/aiohttp)
contributes nothing.
"""

from __future__ import annotations

import re

# Platforms whose config sections exist but whose dependencies are core
# (aiohttp/stdlib), so their ids are not pyproject extras. Filtering them here
# keeps the intent readable; anything else unknown is dropped by the
# declared-extras check anyway.
_UNMAPPED_PLATFORMS = frozenset({"email", "irc", "ntfy", "line", "sms", "mattermost", "simplex"})


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def configured_platform_extras(config: dict | None = None) -> list[str]:
    """Extras for platforms the config enables, that this pyproject declares.

    Reads ``platforms.<id>.enabled`` from both nestings (``gateway.platforms``
    included), the shape every config consumer uses. Never installs: callers
    union the result into an explicit sync, and the engine still refuses
    extras this platform's markers gate off.
    """
    if config is None:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    platforms: dict = {}
    gateway = config.get("gateway") if isinstance(config.get("gateway"), dict) else {}
    for section in (config.get("platforms"), gateway.get("platforms")):
        if isinstance(section, dict):
            platforms.update({k: v for k, v in section.items() if isinstance(v, dict)})
    enabled = sorted(pid for pid, block in platforms.items() if block.get("enabled"))
    if not enabled:
        return []
    from pm.features import declared_extras
    from pm.paths import repo_root

    declared = {_normalized(extra) for extra in declared_extras(repo_root())}
    return sorted({_normalized(pid) for pid in enabled
                   if pid not in _UNMAPPED_PLATFORMS and _normalized(pid) in declared})


def add_configured_platform_extras(extras: list[str] | None) -> list[str] | None:
    """Union an explicit selection with configured-platform extras (install path).

    None stays None when nothing is configured, so ``sync_venv()`` keeps its
    keep-what-is-recorded meaning; a recorded selection only ever grows.
    """
    wanted = configured_platform_extras()
    if not wanted:
        return extras
    return sorted(set(extras or []) | set(wanted))
