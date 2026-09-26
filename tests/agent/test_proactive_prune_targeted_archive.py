"""Regression tests for #124102: a proactive tool-result prune commit used to soft-archive
EVERY active row and re-insert the whole transcript as a new generation, so each prune
duplicated every user turn once per commit (54% of one live store's user rows were archive
generations) and reset every row id. When the prune's held coverage provably names every
active row, the commit must overlay changed payloads in place: unchanged rows keep id,
timestamp and position, nothing is archived, and ambiguous shapes still take the
full-rewrite path. Archive generations are visible only in RAW rows — the display
projection dedupes them by design — so the counts below read the table directly."""

import pytest

from unittest.mock import patch

from agent.context_compressor import ContextCompressor

LARGE_WINDOW = 1_000_000


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=LARGE_WINDOW):
        return ContextCompressor(
            model="test", quiet_mode=True, threshold_percent=0.50,
            protect_first_n=2, protect_last_n=4,
            proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000,
            proactive_prune_min_reclaim_tokens=4_096,
        )


def _history():
    history = [{"role": "user", "content": "start"}]
    for i in range(8):
        history.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{i}", "type": "function",
            "function": {"name": "terminal", "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": f"call_{i}",
                        "content": chr(65 + i) * 24_000 if i < 3 else "ok"})
    return history


def _raw_rows(db):
    return db._read_all(
        "SELECT id, role, content, timestamp, active, compacted FROM messages "
        "WHERE session_id = 's' ORDER BY id", ())


@pytest.fixture
def held_session(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s", source="cli")
    db.append_messages_batch("s", _history())
    cc = _compressor()
    cc._session_db, cc._session_id = db, "s"
    held = db.get_resume_conversations("s")[0]
    yield db, cc, held
    db.close()


def test_prune_commit_overlays_rows_in_place_without_archive_generation(held_session):
    """A committed prune must not create archive generations: the raw row set, ids and
    timestamps all survive; only the demoted tool payloads change (and stay searchable)."""
    db, cc, held = held_session
    before = _raw_rows(db)
    assert len(before) == 17 and all(r["active"] == 1 for r in before)  # 1 user + 8 pairs
    user_before = [r for r in before if r["role"] == "user"][0]

    result, pruned = cc.prune_tool_results_only(held, current_tokens=120_000)

    assert pruned >= 3 and result is not held  # a commit happened
    after = _raw_rows(db)
    assert len(after) == len(before)                     # no rewrite generation
    assert all(r["active"] == 1 for r in after)          # nothing archived
    assert [r["id"] for r in after] == [r["id"] for r in before]  # row ids survive
    users = [r for r in after if r["role"] == "user"]
    assert len(users) == 1                               # the user turn was not duplicated
    assert users[0]["id"] == user_before["id"]
    assert users[0]["timestamp"] == user_before["timestamp"]
    # The demoted payload landed in the row itself, in position, replacing the old body.
    call0 = [r for r in after if r["role"] == "tool" and r["id"] == [
        m for m in result if m.get("role") == "tool" and m.get("tool_call_id") == "call_0"
    ][0]["_row_id"]][0]
    assert len(call0["content"]) < 24_000
    assert sum(len(r["content"]) for r in after) < sum(len(r["content"]) for r in before)


def test_prune_is_idempotent_against_the_store(held_session):
    """Re-pruning the committed transcript must not rewrite the store again."""
    db, cc, held = held_session
    result, _ = cc.prune_tool_results_only(held, current_tokens=120_000)
    after_first = _raw_rows(db)
    second, n2 = cc.prune_tool_results_only(result, current_tokens=120_000)
    assert n2 == 0
    assert [(r["id"], r["content"]) for r in _raw_rows(db)] == [
        (r["id"], r["content"]) for r in after_first]


def test_leaseless_compaction_keeps_the_full_rewrite_path(tmp_path):
    """The issue's unit repro shape — archive_and_compact called with no coverage — is a
    lease-less compaction commit and must keep archiving the old generation (pinned so the
    new in-place path cannot silently swallow the fallback)."""
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", source="cli")
        msgs = [{"role": "user", "content": "hello dup", "timestamp": 1790433081.0},
                {"role": "assistant", "content": "ok", "timestamp": 1790433090.0}]
        db.append_messages_batch("s", msgs)
        db.archive_and_compact("s", msgs)
        db.archive_and_compact("s", msgs)
        rows = db._read_all(
            "SELECT active, compacted FROM messages WHERE session_id = 's'", ())
        assert sum(1 for r in rows if r["active"] == 1) == 2
        assert sum(1 for r in rows if r["active"] == 0 and r["compacted"] == 1) == 4  # 2 generations
    finally:
        db.close()


def test_foreign_tail_still_committing_via_full_rewrite_keeps_it_exactly_once(held_session):
    """A turn another surface appended after load is not covered by the held ids: the commit
    takes the full-rewrite path and must still keep the foreign message exactly once."""
    db, cc, held = held_session
    foreign = "[from Telegram] the vault code is 7741"
    db.append_message("s", "user", foreign)

    result, pruned = cc.prune_tool_results_only(held, current_tokens=120_000)

    assert pruned >= 3 and result is not held
    live = [str(m["content"]) for m in db.get_messages_as_conversation("s")]
    assert live[-1] == foreign
    assert sum(foreign in content for content in live) == 1
