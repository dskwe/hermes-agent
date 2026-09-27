"""Terminal-subshell PATH completion in ``tools/environments/local.py``.

A backend started by a non-interactive SSH session, systemd or a GUI launcher
inherits a PATH without ``~/.local/bin`` (only the login shell adds it), so CLIs
installed there were ``command not found`` from the terminal tool (#111778).
"""

import os
import sys

import pytest

from tools.environments import local as local_mod
from tools.environments.local import _append_missing_sane_path_entries, _make_run_env

pytestmark = pytest.mark.platforms("posix")  # POSIX PATH completion only


def test_existing_user_local_bin_appended_after_inherited_entries(monkeypatch, tmp_path):
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(local_mod, "_git_bash_bin_dirs", lambda: [])
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [])
    monkeypatch.setattr(local_mod, "_resolve_hermes_bin_dir", lambda: None)

    entries = _make_run_env({})["PATH"].split(os.pathsep)

    assert entries[:2] == ["/usr/bin", "/bin"]
    assert entries.count(str(local_bin)) == 1
    # Already on PATH: position kept, no duplicate appended.
    already = _append_missing_sane_path_entries(f"{local_bin}:/usr/bin").split(":")
    assert already[0] == str(local_bin) and already.count(str(local_bin)) == 1


def test_missing_user_local_bin_not_appended(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [])

    assert ".local" not in _append_missing_sane_path_entries("/usr/bin:/bin")


def test_dependency_venv_bin_precedes_managed_tool_dirs(monkeypatch, tmp_path):
    """#125040: pm store activation is store-first, so the process PATH reaches
    the terminal child as ``<store tool python>/bin:<dependency venv>/bin:...``.
    A bare ``python3`` then resolved the standalone tool Python — a bare
    interpreter without Hermes' installed dependencies — and skill scripts died
    on ``import yaml``. The child PATH must keep the dependency venv's bin
    ahead of every managed runtime dir."""
    store_bin = tmp_path / "tools" / "python-3.14" / "bin"
    venv_bin = tmp_path / "installs" / "env" / "venv" / "bin"
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [str(store_bin)])
    monkeypatch.setattr(local_mod, "_active_dependency_venv_bin", lambda: str(venv_bin))

    # Layout after bootstrap activate_dependencies + pm.activate() (store-first).
    child_path = _append_missing_sane_path_entries(f"{store_bin}:{venv_bin}:/usr/bin:/bin")

    entries = child_path.split(":")
    assert entries.index(str(venv_bin)) < entries.index(str(store_bin))
    assert entries.count(str(venv_bin)) == 1
    # User entries keep their relative order ahead of the appended sane tail.
    assert entries.index("/usr/bin") < entries.index("/bin")


def test_dependency_venv_bin_inserted_even_when_missing_from_path(monkeypatch, tmp_path):
    """A backend whose process PATH lost the venv bin still owes terminal
    children the dependency interpreter: the managed dirs would otherwise
    shadow python3 with the bare tool Python."""
    store_bin = tmp_path / "tools" / "python-3.14" / "bin"
    venv_bin = tmp_path / "installs" / "env" / "venv" / "bin"
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [str(store_bin)])
    monkeypatch.setattr(local_mod, "_active_dependency_venv_bin", lambda: str(venv_bin))

    entries = _append_missing_sane_path_entries("/usr/bin:/bin").split(":")

    assert entries.index(str(venv_bin)) < entries.index(str(store_bin))


def test_path_unchanged_without_managed_runtime_dirs(monkeypatch, tmp_path):
    """No store dirs on PATH → nothing shadows the interpreter; the completion
    must not reorder anything (venv relocation only matters vs managed dirs).
    The input survives as a prefix; only the missing sane tail is appended."""
    venv_bin = tmp_path / "installs" / "env" / "venv" / "bin"
    monkeypatch.setattr(local_mod, "_managed_runtime_path_entries", lambda: [])
    monkeypatch.setattr(local_mod, "_active_dependency_venv_bin", lambda: str(venv_bin))
    monkeypatch.setattr(local_mod, "_user_local_bin_entries", lambda: [])

    result = _append_missing_sane_path_entries(f"{venv_bin}:/usr/bin:/bin")

    assert result.startswith(f"{venv_bin}:/usr/bin:/bin:")
