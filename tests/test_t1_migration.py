"""T1 — the SQLite auto-migration that upgrades a database written by an older MeetQ."""

from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy import create_engine

from meetq.db.migrate import migrate
from meetq.db.models import Base, Question, Session
from meetq.db.session import db_session, dispose_db, init_db

pytestmark = pytest.mark.t1

# Exactly the shape a Phase 1 MeetQ left behind: no sessions.live / sessions.summary,
# no questions.round, and an obsolete NOT NULL questions.channels.
OLD_SCHEMA = """
CREATE TABLE sessions (
    id INTEGER NOT NULL PRIMARY KEY,
    title VARCHAR(300) NOT NULL,
    source_type VARCHAR(20) NOT NULL,
    status VARCHAR(30) NOT NULL,
    progress FLOAT NOT NULL,
    stage VARCHAR(60) NOT NULL,
    error TEXT NOT NULL,
    audio_path TEXT NOT NULL,
    duration_sec FLOAT NOT NULL,
    language VARCHAR(20) NOT NULL,
    glossary TEXT NOT NULL,
    mode VARCHAR(20) NOT NULL,
    channels TEXT NOT NULL,
    created_at DATETIME NOT NULL
);
CREATE TABLE questions (
    id INTEGER NOT NULL PRIMARY KEY,
    session_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    type VARCHAR(20) NOT NULL,
    priority INTEGER NOT NULL,
    evidence_quote TEXT NOT NULL,
    evidence_start_sec FLOAT NOT NULL,
    evidence_segment_ids TEXT NOT NULL,
    mode VARCHAR(20) NOT NULL,
    channels TEXT NOT NULL,
    created_at DATETIME NOT NULL
);
"""


def _old_database(path) -> None:
    con = sqlite3.connect(path)
    con.executescript(OLD_SCHEMA)
    con.execute(
        "INSERT INTO sessions VALUES (1, '既存の会議', 'upload', 'done', 1.0, '完了', '',"
        " 'C:/x.wav', 77.4, 'ja', '', 'followup', '[\"mixed\"]', '2026-09-20 00:00:00')"
    )
    con.execute(
        "INSERT INTO questions VALUES (1, 1, '既存の質問', 'clarify', 1, '根拠', 0.0, '[]',"
        " 'followup', '[\"mixed\"]', '2026-09-20 00:00:00')"
    )
    con.commit()
    con.close()


def _columns(path, table: str) -> list[str]:
    con = sqlite3.connect(path)
    try:
        return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]
    finally:
        con.close()


def test_old_database_is_unusable_before_migrating(tmp_path):
    """The failure this migration exists to fix."""
    path = tmp_path / "old.sqlite3"
    _old_database(path)
    assert "live" not in _columns(path, "sessions")
    engine = create_engine(f"sqlite:///{path}", future=True)
    with pytest.raises(Exception, match="no such column"):
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT live FROM sessions")
    engine.dispose()


def test_migrate_adds_missing_columns_and_keeps_the_rows(tmp_path):
    path = tmp_path / "old.sqlite3"
    _old_database(path)
    engine = create_engine(f"sqlite:///{path}", future=True)
    Base.metadata.create_all(engine)
    applied = migrate(engine)

    assert any("sessions" in s and "live" in s for s in applied)
    assert any("sessions" in s and "summary" in s for s in applied)
    assert any("questions" in s and "round" in s for s in applied)

    session_cols = _columns(path, "sessions")
    assert {"live", "summary"} <= set(session_cols)
    assert "round" in _columns(path, "questions")

    with engine.connect() as conn:
        row = conn.exec_driver_sql(
            "SELECT title, live, summary FROM sessions WHERE id = 1"
        ).fetchone()
    assert row[0] == "既存の会議"
    assert row[1] == 0  # the NOT NULL default was applied to the existing row
    assert row[2] == ""
    engine.dispose()


def test_migrate_drops_an_obsolete_not_null_column(tmp_path):
    """questions.channels would make every new INSERT fail, so it must be removed."""
    path = tmp_path / "old.sqlite3"
    _old_database(path)
    engine = create_engine(f"sqlite:///{path}", future=True)
    Base.metadata.create_all(engine)
    applied = migrate(engine)

    assert any("DROP COLUMN" in s and "channels" in s for s in applied)
    assert "channels" not in _columns(path, "questions")
    assert "channels" in _columns(path, "sessions")  # still a real model column
    engine.dispose()


def test_migrate_is_idempotent(tmp_path):
    path = tmp_path / "old.sqlite3"
    _old_database(path)
    engine = create_engine(f"sqlite:///{path}", future=True)
    Base.metadata.create_all(engine)
    assert migrate(engine)
    assert migrate(engine) == []
    engine.dispose()


def test_init_db_migrates_and_the_app_can_write_again(tmp_path):
    """End to end: open an old database through init_db, then insert new rows."""
    path = tmp_path / "old.sqlite3"
    _old_database(path)
    try:
        init_db(path)
        with db_session() as db:
            existing = db.get(Session, 1)
            assert existing is not None
            assert existing.title == "既存の会議"
            assert existing.live is False
            assert existing.summary == ""

            fresh = Session(title="新規", source_type="record")
            db.add(fresh)
            db.flush()
            question = Question(
                session_id=fresh.id,
                text="新しい質問",
                type="risk",
                priority=2,
                evidence_quote="根拠",
                round=3,
            )
            db.add(question)
            db.flush()
            assert question.round == 3
    finally:
        dispose_db()
