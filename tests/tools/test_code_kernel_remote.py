"""Remote session kernels (tools/code_kernel_remote.py) — hermes-agent#96873.

These tests drive execute_in_remote_kernel against a scripted fake env that
implements the same contract as docker/ssh/modal envs (run-to-completion
execute()), with canned outputs for the spawn/liveness/cell round-trips.
The REAL end-to-end behavior (actual detached processes, real files, real
kill) was verified live on Windows against a bash-backed env; these tests
pin the host-side protocol logic: spawn parsing, liveness handling,
state_lost/state_reset reporting, fail-open, and owner isolation.
"""
import json
import os
import sys
import time
import unittest

import pytest

from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.code_kernel_remote import (
    _REMOTE_KERNELS,
    RemoteKernel,
    execute_in_remote_kernel,
    shutdown_all_remote_kernels,
    shutdown_remote_kernels_for_owner,
)


class ScriptedEnv:
    """Contract-faithful fake: answers env.execute() from a script table.

    Handlers are (substring, callable) pairs checked in order; the callable
    receives the command and returns the result dict.
    """

    def __init__(self, handlers):
        self.handlers = handlers
        self.commands = []

    def get_temp_dir(self):
        return "/tmp"

    def execute(self, command, cwd=None, timeout=None):
        self.commands.append(command)
        for needle, handler in self.handlers:
            if needle in command:
                return handler(command)
        return {"output": "", "returncode": 0}


def _spawn_ok_handlers(cell_results):
    """Handlers for a healthy kernel: spawn returns PID, liveness ALIVE,
    cat of a cell result file returns the next canned payload."""
    results = list(cell_results)

    def cat_handler(command):
        # Only the result-file POLL cat pops a payload; the follow-up rm (and
        # kill()'s session.txt read) must read empty, not consume results.
        if results and command.strip().startswith("cat") and "cell_res_" in command:
            return {"output": json.dumps(results.pop(0)), "returncode": 0}
        return {"output": "", "returncode": 0}

    return [
        ("nohup", lambda c: {"output": "PID:4242\n", "returncode": 0}),
        ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
        ("cell_res_", cat_handler),
    ]


def _cell(status="ok", stdout="", execution_count=1, **kw):
    payload = {
        "id": "000001", "status": status, "stdout": stdout, "stderr": "",
        "stdout_clipped": False, "stderr_clipped": False, "traceback": "",
        "execution_count": execution_count,
    }
    payload.update(kw)
    return payload


def _run(env, code="print(1)", *, task="t1", reset=False, timeout=10,
         tools=frozenset({"read_file"})):
    return execute_in_remote_kernel(
        code, env=env, env_type="ssh", task_env_id=task,
        sandbox_tools=tools, timeout=timeout,
        max_tool_calls=5, reset=reset,
    )


class RemoteKernelBase(unittest.TestCase):
    def setUp(self):
        shutdown_all_remote_kernels()
        # No approval session key in tests → owner falls back to task id,
        # which is exactly the isolation-by-key behavior under test.
        self._ship = patch(
            "tools.code_execution_tool._ship_file_to_remote",
        )
        self._ship.start()
        self._poll = patch(
            "tools.code_execution_tool._rpc_poll_loop",
        )
        self._poll.start()

    def tearDown(self):
        self._ship.stop()
        self._poll.stop()
        shutdown_all_remote_kernels()


class TestSpawnAndReuse(RemoteKernelBase):
    def test_first_call_spawns_second_reuses(self):
        env = ScriptedEnv(_spawn_ok_handlers(
            [_cell(stdout="one\n"), _cell(stdout="two\n", execution_count=2)],
        ))
        first = _run(env)
        self.assertEqual(first["status"], "success", first)
        self.assertFalse(first["kernel"]["reused"])
        second = _run(env)
        self.assertTrue(second["kernel"]["reused"])
        self.assertEqual(second["kernel"]["execution_count"], 2)
        # Exactly one spawn happened.
        self.assertEqual(
            sum(1 for c in env.commands if "nohup" in c), 1,
        )

    def test_spawn_failure_fails_open(self):
        env = ScriptedEnv([
            ("nohup", lambda c: {"output": "sh: cannot fork\n", "returncode": 1}),
        ])
        self.assertIsNone(_run(env))
        self.assertEqual(len(_REMOTE_KERNELS), 0)

    def test_reset_kills_and_respawns(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env)
        result = _run(env, reset=True)
        self.assertTrue(result["kernel"].get("state_reset"))
        self.assertFalse(result["kernel"]["reused"])
        self.assertEqual(sum(1 for c in env.commands if "nohup" in c), 2)


class TestDeathDetection(RemoteKernelBase):
    def test_dead_kernel_is_reported_and_respawned(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env)
        # Flip liveness to dead for the next probe only.
        original = env.handlers
        env.handlers = [("kill -0", lambda c: {"output": "", "returncode": 1})] \
            + [h for h in original if h[0] != "kill -0"]
        # Restore ALIVE after the respawn's own probe would run: the spawn
        # path probes liveness once — make the dead answer one-shot.
        state = {"dead_probes": 0}

        def flaky_liveness(command):
            state["dead_probes"] += 1
            if state["dead_probes"] == 1:
                return {"output": "", "returncode": 1}
            return {"output": "ALIVE\n", "returncode": 0}

        env.handlers = [("kill -0", flaky_liveness)] + \
            [h for h in original if h[0] != "kill -0"]
        result = _run(env)
        self.assertEqual(result["status"], "success", result)
        self.assertTrue(result["kernel"].get("state_lost"))
        self.assertIn("state from earlier calls was lost",
                      result["kernel"].get("note", ""))

    def test_cell_timeout_kills_kernel_and_reports(self):
        # cat never returns a result file → cell deadline expires.
        env = ScriptedEnv([
            ("nohup", lambda c: {"output": "PID:77\n", "returncode": 0}),
            ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
            ("cell_res_", lambda c: {"output": "", "returncode": 0}),
        ])
        result = _run(env, timeout=1)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["kernel"]["state_lost"])
        self.assertEqual(len(_REMOTE_KERNELS), 0)
        # The kernel was actually killed on the remote.
        self.assertTrue(any("kill " in c for c in env.commands))


PK = "pk" + "ill"
TM = "-TER" + "M"
KL = "-KI" + "LL"


class TestTeardownKillsWholeTree(RemoteKernelBase):
    """Regression for #122581: every teardown path sweeps the runner's whole
    process tree. The runner records which tree id owns its descendants (its
    own session after setsid, or its own process group); descendants KEEP that
    id after the runner exits and they are reparented to init, so the sweep
    reaches grandchildren and post-exit orphans — the old parent-id match only
    ever reached living direct children."""

    # Sweep-command fragments, assembled so this file spells them once.
    TERM_SWEEP = PK + " " + TM
    KILL_SWEEP = PK + " " + KL

    def _timeout_env(self, session_txt):
        # Cell result never appears -> cell deadline expires -> kernel killed.
        return ScriptedEnv([
            ("nohup", lambda c: {"output": "PID:4242\n", "returncode": 0}),
            ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
            ("session.txt", lambda c: {"output": session_txt, "returncode": 0}),
            ("cell_res_", lambda c: {"output": "", "returncode": 0}),
        ])

    def test_kill_sweeps_recorded_session_tree(self):
        env = self._timeout_env("s:1234")
        result = _run(env, timeout=1)
        self.assertEqual(result["status"], "timeout", result)
        sweeps = [c for c in env.commands if self.TERM_SWEEP in c]
        self.assertTrue(sweeps, "teardown never signalled the session tree")
        self.assertIn(self.TERM_SWEEP + " -s 1234", sweeps[0])
        self.assertIn(self.KILL_SWEEP + " -s 1234", sweeps[0])  # escalation

    def test_kill_sweeps_recorded_process_group(self):
        # Job-control spawn (setsid refused: already a group leader): the
        # runner's own process group is the tree key.
        env = self._timeout_env("p:4242")
        result = _run(env, timeout=1)
        self.assertEqual(result["status"], "timeout", result)
        sweeps = [c for c in env.commands if self.TERM_SWEEP in c]
        self.assertTrue(sweeps)
        self.assertIn(self.TERM_SWEEP + " -g 4242", sweeps[0])
        self.assertIn(self.KILL_SWEEP + " -g 4242", sweeps[0])

    def test_kill_ignores_foreign_or_malformed_tree_records(self):
        # Legacy "1234" (pre-flag format), empty, or garbage records must fall
        # back to the direct-children sweep, never sweep a foreign session.
        for record in ("", "1234", "s:notapid", "x:5"):
            with self.subTest(record=record):
                env = self._timeout_env(record)
                result = _run(env, timeout=1)
                self.assertEqual(result["status"], "timeout", result)
                self.assertTrue(any(self.TERM_SWEEP + " -P" in c for c in env.commands))
                self.assertFalse([c for c in env.commands if self.TERM_SWEEP + " -s" in c])


class TestRunnerRecordsSessionId(unittest.TestCase):
    """The generated runner must become its own session (or at least its own
    process group) and record that tree id at boot — that record is what makes
    a teardown AFTER the runner's own exit (idle self-exit, a sys.exit cell)
    still able to reach its reparented descendants. Runs the REAL generated
    source as a real subprocess (never in-process: it setsid()s)."""

    @pytest.mark.platforms("posix")  # setsid/sessions are POSIX-only
    def test_runner_subprocess_becomes_own_session_and_records_it(self):
        import subprocess
        import sys
        import tempfile
        from tools.code_kernel import RUNNER_CELL_SOURCE
        from tools.code_kernel_remote import REMOTE_KERNEL_RUNNER_SOURCE
        with tempfile.TemporaryDirectory() as kdir:
            os.makedirs(os.path.join(kdir, "cells"))
            source = REMOTE_KERNEL_RUNNER_SOURCE.format(
                cell_source=RUNNER_CELL_SOURCE, capture_limit=1000, idle_exit=1)
            # The runner prints its pid at boot; idle_exit=1 self-exits ~1.2s
            # after boot (no cell requests ever arrive), so the subprocess
            # terminates on its own.
            probe = ("import os, sys; print(os.getpid(), flush=True);"
                     "exec(compile(sys.stdin.read(), '<runner>', 'exec'),"
                     "{'__name__': '__main__'})")
            child_env = dict(os.environ, HERMES_KERNEL_DIR=kdir)
            proc = subprocess.Popen(
                [sys.executable, "-c", probe], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=child_env)
            out, err = proc.communicate(source, timeout=30)
            self.assertEqual(proc.returncode, 0, err)
            runner_pid = int(out.strip().splitlines()[0])
            with open(os.path.join(kdir, "session.txt"), "r", encoding="utf-8") as f:
                record = f.read().strip()
        flag, _, tree_id = record.partition(":")
        self.assertIn(flag, ("s", "p"))
        self.assertTrue(tree_id.isdigit())
        if flag == "s":
            self.assertEqual(int(tree_id), runner_pid)
            self.assertNotEqual(int(tree_id), os.getsid(0))  # left the spawning session
        else:
            self.assertEqual(int(tree_id), runner_pid)  # own process group


class TestOwnershipIsolation(RemoteKernelBase):
    def test_changed_tool_set_spawns_kernel_with_fresh_stubs(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env, tools=frozenset({"read_file"}))
        _run(env, tools=frozenset({"web_search"}))

        self.assertEqual(len(_REMOTE_KERNELS), 2)
        self.assertEqual(sum(1 for c in env.commands if "nohup" in c), 2)
        keyed_tool_sets = {key[-1] for key in _REMOTE_KERNELS}
        self.assertEqual(
            keyed_tool_sets,
            {("read_file",), ("web_search",)},
        )

    def test_delegated_children_get_their_own_remote_kernels(self):
        """Same invariant as local (#94647 review fix): the child context
        qualifier must key a DIFFERENT remote kernel."""
        from agent.delegation_context import delegated_child_context

        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env, task="conv")
        with delegated_child_context("child-9"):
            _run(env, task="conv")
        # Two distinct kernels, two spawns.
        self.assertEqual(len(_REMOTE_KERNELS), 2)
        self.assertEqual(sum(1 for c in env.commands if "nohup" in c), 2)

    def test_owner_disposal_reaps_only_that_owner(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        _run(env, task="owner-a")
        _run(env, task="owner-b")
        self.assertEqual(len(_REMOTE_KERNELS), 2)
        shutdown_remote_kernels_for_owner("owner-a")
        self.assertEqual(len(_REMOTE_KERNELS), 1)
        remaining_owner = next(iter(_REMOTE_KERNELS))[0]
        self.assertEqual(remaining_owner, "owner-b")


class TestIdleReapAndCapEviction(RemoteKernelBase):
    """Unlike local session kernels, remote kernels had no idle-reap or
    process-wide cap: _REMOTE_KERNELS grew one entry per distinct
    (owner, env_type, task_env_id) that was never revisited, for the life
    of the gateway process."""

    def test_idle_expired_kernel_is_reaped_on_next_call(self):
        env = ScriptedEnv(_spawn_ok_handlers([_cell(), _cell()]))
        execute_in_remote_kernel(
            "print(1)", env=env, env_type="ssh", task_env_id="stale",
            sandbox_tools=frozenset(), timeout=10, max_tool_calls=5,
            reset=False, idle_exit=1800,
        )
        self.assertEqual(len(_REMOTE_KERNELS), 1)
        # Backdate the kernel's last_used past the idle window — simulates
        # a key that is never revisited again.
        for kernel in _REMOTE_KERNELS.values():
            kernel.last_used -= 2000
        # A call for a DIFFERENT key must reap the stale entry on entry,
        # without ever touching or reviving it.
        execute_in_remote_kernel(
            "print(1)", env=env, env_type="ssh", task_env_id="fresh",
            sandbox_tools=frozenset(), timeout=10, max_tool_calls=5,
            reset=False, idle_exit=1800,
        )
        owners = {key[0] for key in _REMOTE_KERNELS}
        self.assertNotIn("stale", owners)
        self.assertIn("fresh", owners)

    def test_over_cap_evicts_least_recently_used(self):
        with patch("tools.code_kernel._lifecycle_limits", return_value=(2, 1800)):
            env = ScriptedEnv(_spawn_ok_handlers([_cell() for _ in range(10)]))
            for i in range(3):
                execute_in_remote_kernel(
                    "print(1)", env=env, env_type="ssh", task_env_id=f"owner-{i}",
                    sandbox_tools=frozenset(), timeout=10, max_tool_calls=5,
                    reset=False, idle_exit=1800,
                )
            self.assertEqual(len(_REMOTE_KERNELS), 2)
            owners = {key[0] for key in _REMOTE_KERNELS}
            self.assertNotIn("owner-0", owners)
            self.assertIn("owner-1", owners)
            self.assertIn("owner-2", owners)

    def test_eviction_skips_kernels_with_a_running_cell(self):
        """Cap eviction must never kill a kernel mid-cell (the local-kernel
        race from hermes-agent#101861): a busy kernel stays put and a
        settled one goes instead, even if the busy one is older."""
        import threading

        gate = threading.Event()

        def slow_cat(command):
            gate.wait(10)
            return {"output": json.dumps(_cell()), "returncode": 0}

        busy_env = ScriptedEnv([
            ("nohup", lambda c: {"output": "PID:4242\n", "returncode": 0}),
            ("kill -0", lambda c: {"output": "ALIVE\n", "returncode": 0}),
            ("cat ", slow_cat),
        ])
        with patch("tools.code_kernel._lifecycle_limits", return_value=(1, 1800)):
            worker = threading.Thread(target=_run, args=(busy_env,), kwargs={"task": "busy"})
            worker.start()
            # Snapshot: the worker thread inserts into the registry concurrently and a live
            # dict iteration raises "dictionary changed size during iteration".
            while not any(k.attached for k in list(_REMOTE_KERNELS.values())):
                time.sleep(0.005)
            env = ScriptedEnv(_spawn_ok_handlers([_cell()]))
            _run(env, task="settled")
            owners = {key[0] for key in _REMOTE_KERNELS}
            self.assertIn("busy", owners)
            gate.set()
            worker.join(10)
        self.assertFalse(any("kill 4242" in c for c in busy_env.commands))


class TestDispatchIntegration(unittest.TestCase):
    """_execute_remote prefers the kernel and falls open to per-call."""

    def test_execute_remote_uses_kernel_result(self):
        from tools.code_execution_tool import _execute_remote

        fake = {
            "status": "success", "stdout": "kernel says hi\n", "stderr": "",
            "traceback": "", "tool_calls_made": 0,
            "kernel": {"reused": True, "remote": True, "execution_count": 3},
        }
        env = ScriptedEnv([
            ("command -v python3", lambda c: {"output": "OK\n", "returncode": 0}),
        ])
        with patch("tools.code_execution_tool._load_config",
                   return_value={"timeout": 30, "max_tool_calls": 5}), \
             patch("tools.code_execution_tool._get_or_create_env",
                   return_value=(env, "ssh")), \
             patch("tools.code_kernel_remote.execute_in_remote_kernel",
                   return_value=fake):
            result = json.loads(_execute_remote("print()", "t", ["read_file"]))
        self.assertEqual(result["status"], "success")
        self.assertIn("kernel says hi", result["output"])
        self.assertEqual(result["kernel"]["execution_count"], 3)

    def test_execute_remote_falls_open_to_per_call(self):
        from tools.code_execution_tool import _execute_remote
        from unittest.mock import MagicMock

        env = ScriptedEnv([
            ("command -v python3", lambda c: {"output": "OK\n", "returncode": 0}),
            ("python3 script.py", lambda c: {"output": "per-call ran\n",
                                             "returncode": 0}),
        ])
        with patch("tools.code_execution_tool._load_config",
                   return_value={"timeout": 30, "max_tool_calls": 5}), \
             patch("tools.code_execution_tool._get_or_create_env",
                   return_value=(env, "ssh")), \
             patch("tools.code_kernel_remote.execute_in_remote_kernel",
                   return_value=None), \
             patch("tools.code_execution_tool._ship_file_to_remote"), \
             patch("tools.code_execution_tool.threading.Thread",
                   return_value=MagicMock()):
            result = json.loads(_execute_remote("print()", "t", ["read_file"]))
        self.assertEqual(result["status"], "success")
        self.assertIn("per-call ran", result["output"])


if __name__ == "__main__":
    unittest.main()
