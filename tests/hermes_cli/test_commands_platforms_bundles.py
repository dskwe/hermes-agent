from hermes_cli.commands_platforms import discord_skill_commands_by_category, telegram_menu_commands


def _bundle_rows():
    return iter(
        [
            (
                "/research",
                {"description": "Load research skills", "skills": ["one", "two"]},
                (),
            )
        ]
    )


def _skill_rows(_platform):
    return iter(
        [
            (
                "/research",
                {"description": "A same-named skill", "name": "research", "skill_md_path": "/tmp/research"},
                (),
            )
        ]
    )


def test_discord_catalog_includes_bundles_before_same_named_skills(monkeypatch):
    monkeypatch.setattr("hermes_cli.commands_platforms._iter_gateway_skill_bundles", _bundle_rows)
    monkeypatch.setattr("hermes_cli.commands_platforms._iter_gateway_skills", _skill_rows)

    categories, uncategorized, hidden = discord_skill_commands_by_category(set())

    assert hidden == 1
    assert categories == {}
    assert uncategorized == [("research", "Load research skills", "/research")]


def test_telegram_menu_includes_bundle_and_gives_it_precedence(monkeypatch):
    monkeypatch.setattr("hermes_cli.commands_platforms._iter_gateway_skill_bundles", _bundle_rows)
    monkeypatch.setattr("hermes_cli.commands_platforms._iter_gateway_skills", _skill_rows)
    monkeypatch.setattr("hermes_cli.commands_platforms._iter_plugin_command_entries", lambda: iter([]))

    menu, _hidden = telegram_menu_commands(max_commands=100)

    assert ("research", "Load research skills") in menu
    assert sum(name == "research" for name, _description in menu) == 1
