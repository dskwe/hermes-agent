import sys

import pytest


def _run(monkeypatch, argv_tail, db):
    import hermes_cli.main as main_mod
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda *args, **kwargs: db)
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "archive", *argv_tail])
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    main_mod.main()


def test_archive_targets_session_id_and_requires_include_open(monkeypatch, capsys):
    class FakeDB:
        def __init__(self):
            self.archived = []

        def resolve_session_id(self, value):
            return "live-session" if value == "live" else None

        def get_session(self, session_id):
            return {"id": session_id, "ended_at": None}

        def set_session_archived(self, session_id, archived):
            self.archived.append((session_id, archived))
            return True

        def close(self):
            pass

    db = FakeDB()
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, ["--session-id", "live"], db)
    assert exc.value.code == 1
    assert db.archived == []
    assert "--include-open" in capsys.readouterr().out

    _run(monkeypatch, ["--session-id", "live", "--include-open"], db)
    assert db.archived == [("live-session", True)]
    assert "Archived session 'live-session'." in capsys.readouterr().out


def test_archive_include_open_reaches_bulk_selector(monkeypatch, capsys):
    seen = {}

    class FakeDB:
        def list_prune_candidates(self, **kwargs):
            seen.update(kwargs)
            return [{
                "id": "live-session", "source": "telegram", "title": "run",
                "started_at": 1.0, "last_active": 2.0, "ended_at": None,
                "message_count": 1, "archived": 0,
            }]

        def count_prune_matches(self, **kwargs):
            return 0

        def archive_sessions(self, **kwargs):
            seen["archive_call"] = kwargs
            return 1

        def close(self):
            pass

    _run(monkeypatch, ["--source", "telegram", "--include-open", "--yes"], FakeDB())
    assert seen["include_open"] is True
    assert seen["archive_call"]["include_open"] is True
    assert "Archived 1 session(s)." in capsys.readouterr().out
