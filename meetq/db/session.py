"""SQLite engine / session factory."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from meetq.config import get_settings
from meetq.db.migrate import migrate
from meetq.db.models import Base

_engine = None
_SessionLocal: sessionmaker | None = None


def init_db(db_path: Path | None = None) -> None:
    global _engine, _SessionLocal
    path = db_path or get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(
        f"sqlite:///{path}", future=True, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(_engine)
    migrate(_engine)  # bring a database written by an older MeetQ up to date
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


def dispose_db() -> None:
    """Close all pooled connections (Windows keeps the SQLite file locked otherwise)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


@contextmanager
def db_session() -> Iterator[OrmSession]:
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
