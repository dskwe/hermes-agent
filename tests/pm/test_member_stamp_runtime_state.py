"""Runtime state inside a plugin member directory must not invalidate the stamp.

`members_stamp` hashes every file in each enabled plugin member directory to
decide whether the recorded dependency environment is current. A plugin that
writes runtime state (a SQLite DB, a watermark file, a queue log) into its own
directory is reasonable plugin behaviour — nothing in the plugin API says not
to — but every write changed the stamp, so the next launch re-synced and
re-exec'd forever (#122349). The stamp excludes conventional state suffixes at
the member root and the plugin's own ``.gitignore`` declarations; real build
inputs keep hashing.
"""

from pathlib import Path

from pm.workspace import members_stamp


def _member(tmp_path: Path, name: str = "stateful-plugin") -> Path:
    plugin = tmp_path / name
    plugin.mkdir()
    (plugin / "pyproject.toml").write_text(
        '[project]\nname="stateful-plugin"\nversion="1.0"\nrequires-python=">=3.11"\n',
        encoding="utf-8")
    (plugin / "plugin.yaml").write_text("name: stateful-plugin\n", encoding="utf-8")
    (plugin / "stateful_plugin").mkdir()
    (plugin / "stateful_plugin/__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    return plugin


def test_runtime_state_at_member_root_leaves_stamp_stable(tmp_path):
    # The #122349 report: cronalytics writes facts.db / pending.jsonl (and a
    # watermark.json it declares itself, covered below) at the member root;
    # conventional state files must not move the stamp.
    plugin = _member(tmp_path)
    before = members_stamp([plugin])

    (plugin / "facts.db").write_bytes(b"\x00sqlite")
    (plugin / "pending.jsonl").write_text("{}", encoding="utf-8")

    assert members_stamp([plugin]) == before


def test_changed_build_inputs_still_move_the_stamp(tmp_path):
    # State exclusion must not swallow real build inputs.
    plugin = _member(tmp_path)
    before = members_stamp([plugin])

    (plugin / "stateful_plugin/__init__.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert members_stamp([plugin]) != before


def test_pyproject_and_manifest_changes_still_move_the_stamp(tmp_path):
    plugin = _member(tmp_path)
    before = members_stamp([plugin])

    (plugin / "pyproject.toml").write_text(
        '[project]\nname="stateful-plugin"\nversion="1.1"\nrequires-python=">=3.11"\n',
        encoding="utf-8")

    assert members_stamp([plugin]) != before


def test_plugin_gitignore_excludes_declared_state_at_any_depth(tmp_path):
    # A plugin declares its own state (unconventional names, nested paths) the
    # way every other git tool would: .gitignore.
    plugin = _member(tmp_path)
    (plugin / ".gitignore").write_text(
        "cache/\ndata/liveness-*.json\nwatermark.json\n", encoding="utf-8")
    before = members_stamp([plugin])

    (plugin / "watermark.json").write_text("{}", encoding="utf-8")
    (plugin / "cache").mkdir()
    (plugin / "cache/derived.json").write_text("{}", encoding="utf-8")
    (plugin / "data").mkdir()
    (plugin / "data/liveness-2026-09-25.json").write_text("{}", encoding="utf-8")

    assert members_stamp([plugin]) == before


def test_nested_state_undeclared_by_suffix_still_moves_the_stamp(tmp_path):
    # Root-level suffix exclusion is deliberately conservative: a nested .db
    # the plugin did NOT declare is still a build input as far as the stamp is
    # concerned (it may be packaged data) — declare it in .gitignore instead.
    plugin = _member(tmp_path)
    before = members_stamp([plugin])

    (plugin / "stateful_plugin").mkdir(exist_ok=True)
    (plugin / "stateful_plugin/seed.db").write_bytes(b"\x00sqlite")

    assert members_stamp([plugin]) != before


def test_workspace_member_snapshot_skips_runtime_state(tmp_path):
    # The same rule for the copy into a generation: state must not ride along
    # into the workspace snapshot (it would break repair replay equality and
    # bloat generations).
    from pm.workspace import _workspace_member

    plugin = _member(tmp_path)
    (plugin / "facts.db").write_bytes(b"\x00sqlite")
    (plugin / ".gitignore").write_text("cache/\n", encoding="utf-8")
    (plugin / "cache").mkdir()
    (plugin / "cache/derived.json").write_text("{}", encoding="utf-8")

    root = tmp_path / "workspace"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)

    assert member.is_dir()
    assert not (member / "facts.db").exists()
    assert not (member / "cache").exists()
    assert (member / "pyproject.toml").is_file()
    assert (member / "stateful_plugin/__init__.py").is_file()
