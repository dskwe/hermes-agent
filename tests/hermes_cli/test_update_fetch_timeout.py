"""A dead-stalled network fetch ends the updater with an error, never a hang (#93759, #95777).

`_git_run(network=True)` bounds the wait; a `TimeoutExpired` becomes a failed
CompletedProcess whose stderr names the stall, so every caller's existing
fetch-failure path prints one clear line. Since #124794 the timed-out child is
also reaped as a whole process tree, and the child runs in its own process
group so that reap cannot leak the recursive promisor fetch chain. Local git
(network=False) is unbounded.
"""

import subprocess
from unittest.mock import MagicMock, patch

import hermes_cli.update_cmd as update_cmd


class _HungProc:
    """Fake child whose first communicate raises TimeoutExpired, then records cleanup."""

    def __init__(self):
        self.pid = 424242
        self.events = []
        self.raised = False

    def communicate(self, timeout=None):
        if not self.raised:
            self.raised = True
            self.events.append(("communicate", timeout))
            raise subprocess.TimeoutExpired(["git", "fetch"], timeout)
        self.events.append(("drain", timeout))
        return ("", "")

    def kill(self):
        self.events.append(("kill", None))


def _timeout(cmd, **kwargs):
    if "timeout" in kwargs:
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    return MagicMock(returncode=0, stdout="ok", stderr="")


def test_network_fetch_stall_becomes_a_failed_run_with_a_named_cause(monkeypatch):
    monkeypatch.setattr(update_cmd, "NETWORK_GIT_TIMEOUT_SECONDS", 30)
    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT="/repo"))
    proc = _HungProc()
    tree_kills = []
    monkeypatch.setattr("hermes_cli._subprocess_compat.kill_process_tree", lambda p: tree_kills.append(p))
    with patch.object(update_cmd.subprocess, "Popen", return_value=proc) as popen:
        result = update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True)

    assert result.returncode != 0
    assert "timed out" in result.stderr and "fetch" in result.stderr
    assert popen.call_args.kwargs["process_group"] == 0
    # The no-prompt guard still rides along with the bound.
    assert popen.call_args.kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
    # The bound fired at the configured NETWORK_GIT_TIMEOUT_SECONDS.
    assert proc.events[0] == ("communicate", 30)
    # The whole spawned tree is reaped, not just the direct child.
    assert tree_kills == [proc]
    assert proc.events[-1] == ("drain", 1)


def test_local_git_stays_unbounded_and_check_true_raises(monkeypatch):
    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT="/repo"))
    with patch.object(update_cmd.subprocess, "run", side_effect=_timeout) as run:
        assert update_cmd._git_run(["git"], ["rev-parse", "HEAD"]).returncode == 0
        assert "timeout" not in run.call_args.kwargs

        proc = _HungProc()
        monkeypatch.setattr("hermes_cli._subprocess_compat.kill_process_tree", lambda p: None)
        with patch.object(update_cmd.subprocess, "Popen", return_value=proc):
            try:
                update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True, check=True)
            except subprocess.CalledProcessError as exc:
                assert exc.returncode == 124
            else:
                raise AssertionError("check=True must raise on a timed-out fetch")
