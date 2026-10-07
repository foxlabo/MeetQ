"""F8/F9 — live recording session: stream to the Realtime API, append confirmed text
to the transcript, and generate questions on a rolling timer."""

from __future__ import annotations

import json
import logging
import threading
import time

import numpy as np

from meetq import corrections
from meetq.api.progress import hub
from meetq.audio.base import AudioBuffer
from meetq.audio.loopback import LoopbackSource
from meetq.config import get_settings
from meetq.db.models import Question, Segment, Session, Usage
from meetq.db.session import db_session
from meetq.qgen.base import LLMProvider
from meetq.qgen.openai_provider import build_provider
from meetq.qgen.rolling import generate_rolling
from meetq.stt.openai_live import LiveFinal, OpenAILiveTranscriber

log = logging.getLogger(__name__)


class LiveController:
    """Owns the capture, one live transcriber per channel, and the rolling timer."""

    def __init__(
        self,
        session_id: int,
        language: str,
        keywords: list[str],
        mode: str = "followup",
        provider: LLMProvider | None = None,
    ):
        self.settings = get_settings()
        self.session_id = session_id
        self.language = language
        self.keywords = keywords
        self.mode = mode
        self._provider = provider
        self._transcribers: dict[str, OpenAILiveTranscriber] = {}
        self._source: LoopbackSource | None = None
        self._timer: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.rounds = 0  # completed rounds (their questions are committed)
        self._round_seq = 0  # numbering, taken before a round writes
        self.started_at = 0.0

    # ------------------------------------------------------------- lifecycle
    def start(self, loopback_index: int | None = None, mic_index: int | None = None) -> list[str]:
        channels = ["others"] + (["self"] if mic_index is not None else [])
        for channel in channels:
            self._transcribers[channel] = OpenAILiveTranscriber(
                channel=channel,
                language=self.language,
                keywords=self.keywords,
                on_final=self._on_final,
                on_delta=self._on_delta,
            )
        self._source = LoopbackSource(
            loopback_index=loopback_index, mic_index=mic_index, on_block=self._on_block
        )
        for transcriber in self._transcribers.values():
            transcriber.start()
        self._source.start()
        self.started_at = time.time()
        self._timer = threading.Thread(
            target=self._rolling_loop, name=f"meetq-rolling-{self.session_id}", daemon=True
        )
        self._timer.start()
        return channels

    def stop(self) -> list[AudioBuffer]:
        self._stop_event.set()
        buffers: list[AudioBuffer] = []
        if self._source is not None:
            buffers = self._source.stop()
        for transcriber in self._transcribers.values():
            transcriber.stop()
        if self._timer is not None:
            self._timer.join(timeout=5)
        return buffers

    @property
    def reconnects(self) -> int:
        return sum(t.reconnects for t in self._transcribers.values())

    @property
    def finals(self) -> int:
        return sum(t.finals for t in self._transcribers.values())

    # -------------------------------------------------------------- capture
    def _on_block(self, channel: str, block: np.ndarray) -> None:
        transcriber = self._transcribers.get(channel)
        if transcriber is not None:
            transcriber.feed(block)

    def _on_delta(self, channel: str, delta: str) -> None:
        if delta:
            hub.publish(
                self.session_id,
                {"session_id": self.session_id, "type": "delta", "channel": channel, "text": delta},
            )

    def _on_final(self, final: LiveFinal) -> None:
        """Persist a confirmed turn and push it to the UI (F8)."""
        with self._lock, db_session() as db:
            sess = db.get(Session, self.session_id)
            if sess is None:
                return
            idx = len(sess.segments)
            segment = Segment(
                session_id=self.session_id,
                idx=idx,
                start_sec=final.start_sec,
                end_sec=final.end_sec,
                text=corrections.apply_text(final.text, sess.corrections or ""),
                raw_text=final.text,
                channel=final.channel,
            )
            db.add(segment)
            row = sess.usage or Usage.blank(self.session_id)
            row.transcribed_sec += max(0.0, final.end_sec - final.start_sec)
            row.stt_requests += 1
            db.add(row)
            sess.duration_sec = max(sess.duration_sec, final.end_sec)
            db.flush()
            payload = {
                "session_id": self.session_id,
                "type": "segment",
                "segment": {
                    "id": segment.id,
                    "idx": segment.idx,
                    "start_sec": round(segment.start_sec, 2),
                    "end_sec": round(segment.end_sec, 2),
                    "text": segment.text,
                    "channel": segment.channel,
                },
            }
        hub.publish(self.session_id, payload)

    # -------------------------------------------------------------- rolling
    def _rolling_loop(self) -> None:
        interval = max(5.0, float(self.settings.rolling_interval_sec))
        next_run = time.time() + interval
        while not self._stop_event.is_set():
            if time.time() >= next_run:
                started = time.time()
                try:
                    self.generate_round(trigger="timer")
                except Exception as exc:  # noqa: BLE001 - one bad round must not stop the loop
                    log.warning("rolling generation failed: %s", exc)
                # cadence is measured from the start of a round, so a slow round
                # does not push the next one a full interval further out
                next_run = max(time.time(), started + interval)
            self._stop_event.wait(0.5)

    def generate_round(self, trigger: str = "manual") -> int:
        """One rolling round (F9). Also used by the UI's 今すぐ生成 button."""
        provider = self._provider or build_provider()
        with db_session() as db:
            sess = db.get(Session, self.session_id)
            if sess is None:
                return 0
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
            existing = [q.text for q in sess.questions]
            previous_summary = sess.summary or ""
            mode = sess.mode or self.mode
        if not segments:
            return 0

        result = generate_rolling(
            provider,
            segments,
            existing,
            previous_summary,
            mode=mode,
            window_sec=self.settings.rolling_window_sec,
        )
        with self._lock, db_session() as db:
            sess = db.get(Session, self.session_id)
            if sess is None:
                return 0
            self._round_seq += 1
            round_no = self._round_seq
            sess.summary = result.summary
            for q in result.questions:
                db.add(
                    Question(
                        session_id=self.session_id,
                        text=q.text,
                        type=q.type,
                        priority=q.priority,
                        evidence_quote=q.evidence_quote,
                        evidence_start_sec=q.evidence_start_sec,
                        evidence_segment_ids=json.dumps(q.evidence_segment_ids),
                        mode=mode,
                        round=round_no,
                    )
                )
            row = sess.usage or Usage.blank(self.session_id)
            row.llm_input_tokens += result.usage.input_tokens
            row.llm_output_tokens += result.usage.output_tokens
            row.llm_requests += result.usage.requests
            db.add(row)
        # only count the round once its questions are committed, so anyone who sees
        # `rounds` advance can already read that round's questions
        with self._lock:
            self.rounds += 1
        log.info(
            "rolling round %s (%s): +%s questions, -%s dup, -%s answered",
            round_no,
            trigger,
            len(result.questions),
            result.dropped_duplicates,
            result.dropped_answered,
        )
        hub.publish(
            self.session_id,
            {
                "session_id": self.session_id,
                "type": "questions",
                "round": round_no,
                "trigger": trigger,
                "added": len(result.questions),
                "dropped_duplicates": result.dropped_duplicates,
                "dropped_answered": result.dropped_answered,
            },
        )
        return len(result.questions)


class LiveRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._controllers: dict[int, LiveController] = {}

    def add(self, session_id: int, controller: LiveController) -> None:
        with self._lock:
            self._controllers[session_id] = controller

    def get(self, session_id: int) -> LiveController | None:
        with self._lock:
            return self._controllers.get(session_id)

    def pop(self, session_id: int) -> LiveController | None:
        with self._lock:
            return self._controllers.pop(session_id, None)

    def active(self) -> list[int]:
        with self._lock:
            return sorted(self._controllers)


live_registry = LiveRegistry()
