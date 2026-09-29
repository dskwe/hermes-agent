from __future__ import annotations

import json


def test_recorded_venv_accepts_filesystem_canonical_path_with_different_case(tmp_path, monkeypatch):
    from pm import environments

    generations = tmp_path / "environments"
    environment = generations / "venv"
    environment.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"packages": {"venv": {"environment": str(environment)}}}), encoding="utf-8")

    monkeypatch.setattr(environments, "runtime_facts_path", lambda _: facts)
    monkeypatch.setattr(environments, "install_state_dir", lambda _: tmp_path)

    assert environments._recorded_venv(tmp_path) == environment.resolve()
