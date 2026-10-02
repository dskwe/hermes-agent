from types import ModuleType, SimpleNamespace


def test_windows_pty_spawn_hides_winpty_console(monkeypatch):
    import tools.process_registry as process_registry

    spawned = {}

    class FakePtyProcess:
        pid = 1234

        @classmethod
        def spawn(cls, argv, *, cwd, env, dimensions):
            spawned.update(argv=argv, cwd=cwd, env=env, dimensions=dimensions)
            return cls()

    monkeypatch.setitem(__import__("sys").modules, "winpty", ModuleType("winpty"))
    monkeypatch.setattr(
        __import__("sys").modules["winpty"], "PtyProcess", FakePtyProcess, raising=False
    )
    monkeypatch.setattr(process_registry, "_IS_WINDOWS", True)

    registry = object.__new__(process_registry.ProcessRegistry)
    session = SimpleNamespace(
        id="session-1",
        cwd="C:/work",
        systemd_unit="",
        pid=None,
        host_start_time=None,
        _pty=None,
    )
    monkeypatch.setattr(registry, "_scope_argv", lambda *args: ["cmd.exe", "/c", "echo", "ok"])
    monkeypatch.setattr(registry, "_spawn_env", lambda _env_vars: {"WINPTY_SHOW_CONSOLE": "1"})
    monkeypatch.setattr(registry, "_safe_host_start_time", lambda _pid: 99)
    monkeypatch.setattr(registry, "_track_started", lambda *args: None)

    result = registry._spawn_local_pty(session, "echo ok", None)

    assert result is session
    assert spawned["env"]["WINPTY_SHOW_CONSOLE"] == "0"
    assert spawned["dimensions"] == (30, 120)
