"""Updater network git children are process-group isolated and reaped as a tree (#124794).

On a treeless partial clone with git < 2.44, a stalled fetch recurses: each git
level spawns a nested promisor `git fetch` under the timed-out child. Killing
only the direct child (what subprocess.run(timeout=...) does) leaves the chain
running — on the reporter's 8 GB board it grew to 2,922 git processes,
exhausted swap, and killed the gateway's platform adapter. The updater's
network git children therefore run in their own process group on POSIX
(Windows cleanup rides the Job-Object/tree-kill inside kill_process_tree) and a
fired timeout reaps the whole spawned tree.
"""

import os
import subprocess

import pytest

import hermes_cli.update_cmd as update_cmd
import hermes_cli.update_cmd_git as update_cmd_git


def _popen_seam_stub(monkeypatch, *, hung=False):
    """Replace subprocess.Popen with a recorder; returns (calls, procs-events).

    With hung=True the fake child's first communicate(timeout=...) raises
    TimeoutExpired and later calls record the drain, so cleanup is observable
    without any real process being spawned.
    """
    calls = []
    procs = []

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            self.args = cmd
            self.pid = 424242
            self.returncode = 0
            self.events = []
            calls.append((cmd, kwargs))
            procs.append(self.events)
            if hung:
                self.raised = False

        def communicate(self, timeout=None):
            if not hung:
                return ("", "")
            if not self.raised:
                self.raised = True
                self.events.append(("communicate", timeout))
                raise subprocess.TimeoutExpired(self.args, timeout)
            self.events.append(("drain", timeout))
            return ("", "")

        def kill(self):
            self.events.append(("kill", None))

    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    return calls, procs


@pytest.mark.platforms("posix")
def test_network_git_child_runs_in_its_own_process_group(monkeypatch, tmp_path):
    monkeypatch.setattr(update_cmd, "_m", lambda: type("M", (), {"PROJECT_ROOT": str(tmp_path)})())
    calls, _ = _popen_seam_stub(monkeypatch)
    update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True)
    kwargs = calls[0][1]
    assert kwargs["process_group"] == 0
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"


@pytest.mark.platforms("windows")
def test_network_git_child_windows_spawns_without_posix_group_kwarg(monkeypatch, tmp_path):
    monkeypatch.setattr(update_cmd, "_m", lambda: type("M", (), {"PROJECT_ROOT": str(tmp_path)})())
    calls, _ = _popen_seam_stub(monkeypatch)
    update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True)
    assert "process_group" not in calls[0][1]


@pytest.mark.platforms("posix")
def test_timed_out_fetch_reaps_the_whole_spawned_tree(monkeypatch, tmp_path):
    """Live red/green: the marker sleeps survive the timeout on base; none survive here."""
    pytest.importorskip("psutil")
    import psutil

    monkeypatch.setattr(update_cmd, "_m", lambda: type("M", (), {"PROJECT_ROOT": str(tmp_path)})())
    monkeypatch.setattr(update_cmd, "NETWORK_GIT_TIMEOUT_SECONDS", 1)
    marker = "hermes-pr-124794-tree-reap"
    result = update_cmd._git_run(["sh"], ["-c", f"sleep 30 & sleep 30; wait # {marker}"],
                                 cwd=tmp_path, network=True)
    assert result.returncode == 124
    assert "timed out" in (result.stderr or "")
    leftovers = [
        p for p in psutil.process_iter(["cmdline"])
        if p.info["cmdline"] and any(marker in str(part) for part in p.info["cmdline"])
    ]
    assert leftovers == [], f"spawned tree survived the timeout: {[p.pid for p in leftovers]}"


def test_upstream_sync_fetch_is_isolated_and_bounded(monkeypatch, tmp_path):
    calls, _ = _popen_seam_stub(monkeypatch)
    update_cmd_git._upstream_git_run(["git", "fetch", "upstream", "main", "--quiet"], tmp_path)
    kwargs = calls[0][1]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
    if os.name != "nt":
        assert kwargs["process_group"] == 0


def test_upstream_sync_timeout_reaps_then_reraises(monkeypatch, tmp_path):
    calls, procs = _popen_seam_stub(monkeypatch, hung=True)
    tree_kills = []
    monkeypatch.setattr("hermes_cli._subprocess_compat.kill_process_tree", lambda p: tree_kills.append(p))
    with pytest.raises(subprocess.TimeoutExpired):
        update_cmd_git._upstream_git_run(["git", "pull", "--ff-only", "upstream", "main"], tmp_path)
    assert procs[0][0] == ("communicate", update_cmd.NETWORK_GIT_TIMEOUT_SECONDS)
    assert len(tree_kills) == 1
    assert procs[0][-1] == ("drain", 1)
