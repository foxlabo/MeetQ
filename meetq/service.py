"""Session processing: decode -> chunk -> transcribe -> generate questions."""

from __future__ import annotations

import json
import logging
import threading
import traceback
from pathlib import Path

from meetq import corrections
from meetq.api.progress import hub
from meetq.audio.base import SAMPLE_RATE, AudioBuffer, Chunk
from meetq.audio.chunker import VadChunker
from meetq.audio.file_source import decode_to_pcm
from meetq.audio.wavio import read_wav
from meetq.config import get_settings
from meetq.db.models import Question, Segment, Session, Usage
from meetq.db.session import db_session
from meetq.qgen.base import LLMProvider
from meetq.qgen.openai_provider import build_provider
from meetq.qgen.pipeline import generate_questions
from meetq.qgen.validate import TranscriptIndex
from meetq.stt.base import Transcriber, TranscriptionContext
from meetq.stt.openai_file import OpenAIFileTranscriber

log = logging.getLogger(__name__)


def _set(session_id: int, **fields) -> None:
    with db_session() as db:
        sess = db.get(Session, session_id)
        if sess is None:
            return
        for key, value in fields.items():
            setattr(sess, key, value)
        payload = {
            "session_id": session_id,
            "status": sess.status,
            "stage": sess.stage,
            "progress": round(sess.progress, 4),
            "error": sess.error,
        }
    hub.publish(session_id, payload)


def load_buffers(path: Path, channels: list[str]) -> list[AudioBuffer]:
    """Load audio as one buffer per logical channel."""
    if path.suffix.lower() == ".wav" and len(channels) > 1:
        tracks, rate = read_wav(path)
        if rate == SAMPLE_RATE and len(tracks) == len(channels):
            return [
                AudioBuffer(pcm=track, channel=name, sample_rate=SAMPLE_RATE)
                for name, track in zip(channels, tracks, strict=False)
            ]
    pcm = decode_to_pcm(path)
    channel = channels[0] if len(channels) == 1 else "mixed"
    return [AudioBuffer(pcm=pcm, channel=channel, sample_rate=SAMPLE_RATE)]


def build_chunks(buffers: list[AudioBuffer]) -> list[Chunk]:
    settings = get_settings()
    chunker = VadChunker(
        target_sec=settings.chunk_target_sec,
        min_sec=settings.chunk_min_sec,
        max_sec=settings.chunk_max_sec,
    )
    chunks: list[Chunk] = []
    for buf in buffers:
        chunks.extend(chunker.chunk(buf.pcm, channel=buf.channel))
    chunks.sort(key=lambda c: (c.start_sec, c.channel))
    for i, chunk in enumerate(chunks):
        chunk.index = i
    return chunks


def run_question_generation(
    session_id: int, mode: str, provider: LLMProvider | None = None
) -> int:
    """Generate (or regenerate) questions for an already transcribed session."""
    provider = provider or build_provider()
    with db_session() as db:
        sess = db.get(Session, session_id)
        if sess is None:
            raise ValueError(f"session {session_id} not found")
        segments = [
            {
                "id": s.id,
                "start_sec": s.start_sec,
                "end_sec": s.end_sec,
                "text": s.text,
                "channel": s.channel,
            }
            for s in sorted(sess.segments, key=lambda s: s.start_sec)
        ]
    _set(session_id, status="generating", stage="質問生成中", progress=0.85, mode=mode)
    questions, usage = generate_questions(provider, segments, mode=mode)
    with db_session() as db:
        sess = db.get(Session, session_id)
        assert sess is not None
        for old in list(sess.questions):
            db.delete(old)
        db.flush()
        for q in questions:
            db.add(
                Question(
                    session_id=session_id,
                    text=q.text,
                    type=q.type,
                    priority=q.priority,
                    evidence_quote=q.evidence_quote,
                    evidence_start_sec=q.evidence_start_sec,
                    evidence_segment_ids=json.dumps(q.evidence_segment_ids),
                    mode=mode,
                )
            )
        row = sess.usage or Usage.blank(session_id)
        row.llm_input_tokens += usage.input_tokens
        row.llm_output_tokens += usage.output_tokens
        row.llm_requests += usage.requests
        db.add(row)
    return len(questions)


def process_session(
    session_id: int,
    transcriber: Transcriber | None = None,
    provider: LLMProvider | None = None,
) -> None:
    """Full pipeline for one session. Runs in a worker thread."""
    try:
        with db_session() as db:
            sess = db.get(Session, session_id)
            if sess is None:
                raise ValueError(f"session {session_id} not found")
            audio_path = Path(sess.audio_path)
            language = sess.language or get_settings().language
            glossary = [t.strip() for t in (sess.glossary or "").splitlines() if t.strip()]
            rules_text = sess.corrections or ""
            # the corrected terms double as STT hints so they are heard right to begin with
            for term in corrections.targets(rules_text):
                if term not in glossary:
                    glossary.append(term)
            mode = sess.mode or "followup"
            channels = json.loads(sess.channels or '["mixed"]')

        _set(session_id, status="processing", stage="音声をデコード中", progress=0.05, error="")
        buffers = load_buffers(audio_path, channels)
        duration = max((b.duration_sec for b in buffers), default=0.0)
        _set(session_id, stage="無音区切りでチャンク化中", progress=0.1, duration_sec=duration)
        chunks = build_chunks(buffers)
        speech_chunks = [c for c in chunks if c.meta.get("has_speech", True)]
        if not speech_chunks:
            speech_chunks = chunks

        transcriber = transcriber or OpenAIFileTranscriber()

        def on_progress(done: int, total: int) -> None:
            _set(
                session_id,
                stage=f"文字起こし中 {done}/{total}",
                progress=0.1 + 0.7 * (done / max(1, total)),
            )

        _set(session_id, status="transcribing", stage="文字起こし中", progress=0.12)
        result = transcriber.transcribe_chunks(
            speech_chunks,
            TranscriptionContext(language=language, keywords=glossary),
            on_progress=on_progress,
        )

        with db_session() as db:
            sess = db.get(Session, session_id)
            assert sess is not None
            for old in list(sess.segments):
                db.delete(old)
            db.flush()
            for seg in result.segments:
                db.add(
                    Segment(
                        session_id=session_id,
                        idx=seg.index,
                        start_sec=seg.start_sec,
                        end_sec=seg.end_sec,
                        text=corrections.apply_text(seg.text, rules_text),
                        raw_text=seg.text,
                        channel=seg.channel,
                    )
                )
            row = sess.usage or Usage.blank(session_id)
            row.transcribed_sec += result.transcribed_sec
            row.stt_requests += result.requests
            db.add(row)

        run_question_generation(session_id, mode, provider=provider)
        _set(session_id, status="done", stage="完了", progress=1.0)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        log.error("session %s failed: %s\n%s", session_id, exc, traceback.format_exc())
        _set(session_id, status="error", stage="エラー", error=f"{type(exc).__name__}: {exc}")


def apply_session_corrections(session_id: int, rules_text: str) -> dict:
    """Re-derive every segment from its raw STT text with `rules_text`.

    Question texts get the same fixes, and each evidence quote is re-anchored so it
    stays a substring of the transcript (F4).
    """
    rules, errors = corrections.parse_rules(rules_text)
    if errors:
        raise ValueError("; ".join(errors))
    stats = {"segments_changed": 0, "replacements": 0, "questions_changed": 0, "reanchored": 0}
    with db_session() as db:
        sess = db.get(Session, session_id)
        if sess is None:
            raise ValueError(f"session {session_id} not found")
        segments = sorted(sess.segments, key=lambda s: s.start_sec)
        for seg in segments:
            if not seg.raw_text:
                seg.raw_text = seg.text  # sessions from before corrections existed
            new_text, count = corrections.apply_rules(seg.raw_text, rules)
            stats["replacements"] += count
            if new_text != seg.text:
                seg.text = new_text
                stats["segments_changed"] += 1
        db.flush()

        index = TranscriptIndex(
            [{"id": s.id, "start_sec": s.start_sec, "text": s.text} for s in segments]
        )
        by_id = {s.id: s for s in segments}
        for q in sess.questions:
            new_q_text, _ = corrections.apply_rules(q.text, rules)
            changed = new_q_text != q.text
            q.text = new_q_text
            fixed_quote, _ = corrections.apply_rules(q.evidence_quote, rules)
            located = None
            quote = q.evidence_quote
            for candidate in (q.evidence_quote, fixed_quote):
                located = index.locate(candidate)
                if located is not None:
                    quote = candidate
                    break
            if located is None:
                # a rule cut across the quote boundary: fall back to the whole evidence
                # segment, which is a substring by construction
                ids = [i for i in json.loads(q.evidence_segment_ids or "[]") if i in by_id]
                if ids:
                    quote = by_id[ids[0]].text
                    located = index.locate(quote)
                    stats["reanchored"] += 1
            if located is not None:
                if quote != q.evidence_quote:
                    changed = True
                q.evidence_quote = quote
                q.evidence_segment_ids = json.dumps(located[0])
                q.evidence_start_sec = located[1]
            if changed:
                stats["questions_changed"] += 1

        sess.summary = corrections.apply_rules(sess.summary or "", rules)[0]
        sess.corrections = rules_text
    return stats


def start_processing(session_id: int) -> threading.Thread:
    thread = threading.Thread(
        target=process_session,
        args=(session_id,),
        name=f"meetq-session-{session_id}",
        daemon=True,
    )
    thread.start()
    return thread


def start_manual_round(session_id: int, controller) -> threading.Thread:
    """Run one rolling round in the background (F9's 今すぐ生成).

    A round can take a minute or more on a long meeting, which is past the timeout of
    an HTTP proxy or tunnel in front of MeetQ, so the request returns at once
    and the UI picks the questions up from the WebSocket.
    """

    def _run() -> None:
        try:
            controller.generate_round(trigger="manual")
        except Exception as exc:  # noqa: BLE001 - surfaced through the live status
            log.error("manual round failed: %s", exc)

    thread = threading.Thread(target=_run, name=f"meetq-manual-{session_id}", daemon=True)
    thread.start()
    return thread


def start_regeneration(session_id: int, mode: str) -> threading.Thread:
    def _run() -> None:
        try:
            run_question_generation(session_id, mode)
            _set(session_id, status="done", stage="完了", progress=1.0)
        except Exception as exc:  # noqa: BLE001
            log.error("regeneration failed: %s", exc)
            _set(session_id, status="error", stage="エラー", error=str(exc))

    thread = threading.Thread(target=_run, name=f"meetq-regen-{session_id}", daemon=True)
    thread.start()
    return thread
