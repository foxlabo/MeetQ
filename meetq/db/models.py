"""SQLAlchemy models: sessions / segments / questions / usage."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(300), default="")
    source_type: Mapped[str] = mapped_column(String(20), default="upload")  # upload | record
    status: Mapped[str] = mapped_column(String(30), default="created")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str] = mapped_column(String(60), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    audio_path: Mapped[str] = mapped_column(Text, default="")
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    language: Mapped[str] = mapped_column(String(20), default="ja")
    glossary: Mapped[str] = mapped_column(Text, default="")  # newline separated terms
    mode: Mapped[str] = mapped_column(String(20), default="followup")
    channels: Mapped[str] = mapped_column(Text, default='["mixed"]')  # JSON list, WAV order
    live: Mapped[bool] = mapped_column(Boolean, default=False)  # F8: live transcription on
    summary: Mapped[str] = mapped_column(Text, default="")  # F9: cumulative summary
    corrections: Mapped[str] = mapped_column(Text, default="")  # term fixes, `wrong => right`
    created_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)

    segments: Mapped[list[Segment]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Segment.start_sec"
    )
    questions: Mapped[list[Question]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    usage: Mapped[Usage | None] = relationship(
        back_populates="session", cascade="all, delete-orphan", uselist=False
    )


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    idx: Mapped[int] = mapped_column(Integer, default=0)
    start_sec: Mapped[float] = mapped_column(Float, default=0.0)
    end_sec: Mapped[float] = mapped_column(Float, default=0.0)
    text: Mapped[str] = mapped_column(Text, default="")  # shown text, corrections applied
    raw_text: Mapped[str] = mapped_column(Text, default="")  # STT output, before corrections
    channel: Mapped[str] = mapped_column(String(10), default="mixed")  # self | others | mixed

    session: Mapped[Session] = relationship(back_populates="segments")


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    text: Mapped[str] = mapped_column(Text, default="")
    type: Mapped[str] = mapped_column(String(20), default="clarify")
    priority: Mapped[int] = mapped_column(Integer, default=2)
    evidence_quote: Mapped[str] = mapped_column(Text, default="")
    evidence_start_sec: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_segment_ids: Mapped[str] = mapped_column(Text, default="[]")  # JSON list of int
    mode: Mapped[str] = mapped_column(String(20), default="followup")
    round: Mapped[int] = mapped_column(Integer, default=0)  # F9: rolling generation round
    created_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)

    session: Mapped[Session] = relationship(back_populates="questions")


class Usage(Base):
    """F7: per-session transcribed audio seconds and LLM token counts."""

    __tablename__ = "usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    transcribed_sec: Mapped[float] = mapped_column(Float, default=0.0)
    stt_requests: Mapped[int] = mapped_column(Integer, default=0)
    llm_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_requests: Mapped[int] = mapped_column(Integer, default=0)

    session: Mapped[Session] = relationship(back_populates="usage")

    @classmethod
    def blank(cls, session_id: int) -> Usage:
        """Column defaults only apply at INSERT, so zero the counters explicitly."""
        return cls(
            session_id=session_id,
            transcribed_sec=0.0,
            stt_requests=0,
            llm_input_tokens=0,
            llm_output_tokens=0,
            llm_requests=0,
        )


class AppSetting(Base):
    """Key/value store for UI-editable settings (F6). Never holds the API key."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(60), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")
