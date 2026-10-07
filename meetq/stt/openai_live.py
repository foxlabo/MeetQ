"""OpenAILiveTranscriber — one Realtime WebSocket transcription session per channel.

Protocol details (URL intent, 24 kHz minimum, turn_detection=null, manual commit)
were verified against the live API.
"""

from __future__ import annotations

import base64
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from meetq.audio.base import SAMPLE_RATE
from meetq.audio.chunker import FRAME_MS, VadBackend, default_vad
from meetq.config import Settings, get_settings

log = logging.getLogger(__name__)

DELTA_EVENT = "conversation.item.input_audio_transcription.delta"
COMPLETED_EVENT = "conversation.item.input_audio_transcription.completed"


@dataclass
class LiveFinal:
    """A confirmed transcript for one committed turn."""

    start_sec: float
    end_sec: float
    text: str
    channel: str


def upsample(pcm: np.ndarray, src_rate: int, dst_rate: int, carry: float | None = None):
    """Linear resample of one block, carrying the previous block's last sample."""
    pcm = np.asarray(pcm, dtype=np.float32)
    if src_rate == dst_rate or len(pcm) == 0:
        return pcm.astype(np.int16), (float(pcm[-1]) if len(pcm) else carry)
    source = pcm if carry is None else np.concatenate(([carry], pcm))
    offset = 0.0 if carry is None else 1.0
    out_len = int(round(len(pcm) * dst_rate / src_rate))
    positions = offset + np.arange(out_len) * (src_rate / dst_rate)
    resampled = np.interp(positions, np.arange(len(source)), source)
    return np.clip(resampled, -32768, 32767).astype(np.int16), float(pcm[-1])


class OpenAILiveTranscriber:
    """Streams 16 kHz mono PCM to the Realtime API and reports confirmed text.

    `on_final` is called from the reader thread for every committed turn.
    `on_delta` (optional) receives partial text as it arrives.
    """

    def __init__(
        self,
        channel: str = "mixed",
        language: str = "ja",
        keywords: list[str] | None = None,
        on_final: Callable[[LiveFinal], None] | None = None,
        on_delta: Callable[[str, str], None] | None = None,
        client=None,
        settings: Settings | None = None,
        model: str | None = None,
        vad: VadBackend | None = None,
    ):
        self.settings = settings or get_settings()
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.settings.openai_api_key, timeout=self.settings.openai_timeout
            )
        self.client = client
        if model is None:
            from meetq import appsettings

            model = appsettings.all_settings()["stt_live_model"]
        self.model = model
        self.channel = channel
        self.language = language
        self.keywords = keywords or []
        self.on_final = on_final
        self.on_delta = on_delta
        self.vad = vad or default_vad()

        self._queue: queue.Queue = queue.Queue(maxsize=512)
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._pending_turns: deque[tuple[float, float]] = deque()
        self._fed_sec = 0.0
        self._turn_start_sec = 0.0
        self._carry: float | None = None
        self._residual = np.zeros(0, dtype=np.int16)
        self._silence_run = 0.0
        self.reconnects = 0
        self.errors: list[str] = []
        self.finals = 0
        self._conn = None

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._worker = threading.Thread(
            target=self._run, name=f"meetq-live-{self.channel}", daemon=True
        )
        self._worker.start()

    def feed(self, pcm: np.ndarray) -> None:
        """Queue a block of 16 kHz mono int16 audio."""
        if self._stop_event.is_set():
            return
        try:
            self._queue.put_nowait(np.asarray(pcm, dtype=np.int16).copy())
        except queue.Full:  # pragma: no cover - only under sustained overload
            log.warning("live audio queue full on channel %s; dropping a block", self.channel)

    def stop(self, timeout: float = 15.0) -> None:
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)

    def drop_connection(self) -> bool:
        """Force the current WebSocket closed so the session reconnects (F8).

        Useful to recover a wedged session without losing the recording, and it is
        how T6 exercises the reconnect path against the real API.
        """
        conn = self._conn
        if conn is None:
            return False
        try:
            conn.close()
        except Exception:  # pragma: no cover - already gone
            return False
        return True

    # ------------------------------------------------------------- internals
    def _session_update(self) -> dict:
        transcription: dict = {"model": self.model}
        if self.language:
            transcription["languages"] = [self.language]
        if self.keywords:
            transcription["keywords"] = list(self.keywords)
        if self.settings.live_delay:
            transcription["delay"] = self.settings.live_delay
        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": int(self.settings.live_sample_rate),
                        },
                        "transcription": transcription,
                        "turn_detection": None,
                    }
                },
            },
        }

    def _run(self) -> None:
        delay = 0.5
        while not self._stop_event.is_set():
            try:
                with self.client.realtime.connect(
                    extra_query={"intent": "transcription"}, max_retries=0
                ) as conn:
                    self._conn = conn
                    conn.send(self._session_update())
                    reader = threading.Thread(
                        target=self._read_events,
                        args=(conn,),
                        name=f"meetq-live-read-{self.channel}",
                        daemon=True,
                    )
                    reader.start()
                    self._pump(conn)
                    reader.join(timeout=5)
                self._conn = None
                delay = 0.5
                if self._stop_event.is_set():
                    return
            except Exception as exc:  # noqa: BLE001 - reconnect on any transport failure
                if self._stop_event.is_set():
                    return
                self.reconnects += 1
                self.errors.append(f"{type(exc).__name__}: {exc}")
                log.warning(
                    "live session %s dropped (%s); reconnecting in %.1fs",
                    self.channel,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
                # a reconnected session starts a fresh turn
                with self._lock:
                    self._turn_start_sec = self._fed_sec
                    self._pending_turns.clear()

    def _read_events(self, conn) -> None:
        try:
            for event in conn:
                event_type = getattr(event, "type", "")
                if event_type == DELTA_EVENT:
                    if self.on_delta is not None:
                        self.on_delta(self.channel, getattr(event, "delta", "") or "")
                elif event_type == COMPLETED_EVENT:
                    self._emit_final(getattr(event, "transcript", "") or "")
                elif event_type == "error":
                    detail = getattr(event, "error", None)
                    message = getattr(detail, "message", str(detail))
                    self.errors.append(message)
                    log.error("live session %s error: %s", self.channel, message)
                if self._stop_event.is_set():
                    return
        except Exception as exc:  # noqa: BLE001 - surfaces as a reconnect in _run
            if not self._stop_event.is_set():
                log.warning("live reader %s stopped: %s", self.channel, type(exc).__name__)

    def _emit_final(self, text: str) -> None:
        with self._lock:
            bounds = self._pending_turns.popleft() if self._pending_turns else (0.0, self._fed_sec)
        text = text.strip()
        if not text:
            return
        self.finals += 1
        if self.on_final is not None:
            self.on_final(
                LiveFinal(
                    start_sec=round(bounds[0], 3),
                    end_sec=round(bounds[1], 3),
                    text=text,
                    channel=self.channel,
                )
            )

    def _commit(self, conn) -> None:
        with self._lock:
            start, end = self._turn_start_sec, self._fed_sec
            if end - start <= 0.05:
                return
            self._pending_turns.append((start, end))
            self._turn_start_sec = end
        self._silence_run = 0.0
        conn.send({"type": "input_audio_buffer.commit"})

    def _should_commit(self, block: np.ndarray) -> bool:
        """Local VAD decides turn boundaries (the live model has no server-side VAD)."""
        frame_len = int(SAMPLE_RATE * FRAME_MS / 1000)
        data = np.concatenate([self._residual, block])
        frames = len(data) // frame_len
        for i in range(frames):
            frame = data[i * frame_len : (i + 1) * frame_len]
            if self.vad.is_speech(frame, SAMPLE_RATE):
                self._silence_run = 0.0
            else:
                self._silence_run += FRAME_MS / 1000.0
        self._residual = data[frames * frame_len :]
        with self._lock:
            turn_len = self._fed_sec - self._turn_start_sec
        if turn_len >= self.settings.live_turn_max_sec:
            return True
        return turn_len >= self.settings.live_turn_min_sec and self._silence_run >= 0.4

    def _pump(self, conn) -> None:
        rate = int(self.settings.live_sample_rate)
        while not self._stop_event.is_set():
            try:
                block = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            resampled, self._carry = upsample(block, SAMPLE_RATE, rate, self._carry)
            conn.send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(resampled.tobytes()).decode("ascii"),
                }
            )
            with self._lock:
                self._fed_sec += len(block) / SAMPLE_RATE
            if self._should_commit(block):
                self._commit(conn)
        # flush the tail so the last words are still confirmed
        try:
            while True:
                block = self._queue.get_nowait()
                resampled, self._carry = upsample(block, SAMPLE_RATE, rate, self._carry)
                conn.send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(resampled.tobytes()).decode("ascii"),
                    }
                )
                with self._lock:
                    self._fed_sec += len(block) / SAMPLE_RATE
        except queue.Empty:
            pass
        try:
            self._commit(conn)
            time.sleep(1.5)  # let the final `completed` event arrive
        except Exception:  # pragma: no cover - connection already gone
            pass
