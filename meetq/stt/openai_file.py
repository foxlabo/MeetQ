"""OpenAIFileTranscriber — chunked file transcription via POST /v1/audio/transcriptions.

Timestamps come from the local VAD chunks, not from the model (only whisper-1 has them).
Chunks run in bounded-size batches so that the previous batch's tail text can be fed
forward as `prompt` while still transcribing in parallel.
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor

from meetq.audio.base import Chunk
from meetq.audio.wavio import pcm_to_wav_bytes
from meetq.config import Settings, get_settings
from meetq.stt.base import (
    Transcriber,
    TranscriptionContext,
    TranscriptionResult,
    TranscriptSegment,
)

log = logging.getLogger(__name__)

PROMPT_TAIL_CHARS = 400


def _is_retryable(exc: Exception) -> bool:
    # A 429 for an exhausted credit balance is permanent: retrying only wastes time.
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and body.get("error", {}).get("type") == "insufficient_quota":
        return False
    if "insufficient_quota" in str(getattr(exc, "message", "")) or "no credits remaining" in str(
        exc
    ):
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


class OpenAIFileTranscriber(Transcriber):
    def __init__(self, client=None, settings: Settings | None = None, model: str | None = None):
        self.settings = settings or get_settings()
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.settings.openai_api_key, timeout=self.settings.openai_timeout
            )
        self.client = client
        from meetq import appsettings

        self.model = model or appsettings.all_settings()["stt_model"]

    # -- single chunk ---------------------------------------------------
    def _transcribe_one(self, chunk: Chunk, context: TranscriptionContext, prompt: str) -> str:
        wav = pcm_to_wav_bytes(chunk.pcm, chunk.sample_rate)
        kwargs: dict = {
            "model": self.model,
            "file": (f"chunk_{chunk.index:04d}.wav", wav, "audio/wav"),
        }
        if prompt:
            kwargs["prompt"] = prompt[-PROMPT_TAIL_CHARS:]
        if context.keywords:
            kwargs["keywords"] = list(context.keywords)
        if context.language:
            kwargs["languages"] = [context.language]

        delay = 1.0
        last: Exception | None = None
        for attempt in range(self.settings.stt_max_retries):
            try:
                resp = self.client.audio.transcriptions.create(**kwargs)
                return (getattr(resp, "text", "") or "").strip()
            except Exception as exc:  # noqa: BLE001 - re-raised below when not retryable
                last = exc
                if not _is_retryable(exc) or attempt == self.settings.stt_max_retries - 1:
                    raise
                sleep_for = delay + random.random() * 0.3
                log.warning(
                    "transcription retry %s/%s in %.1fs (%s)",
                    attempt + 1,
                    self.settings.stt_max_retries,
                    sleep_for,
                    type(exc).__name__,
                )
                time.sleep(sleep_for)
                delay *= 2
        raise last  # pragma: no cover - loop always returns or raises

    # -- batch ----------------------------------------------------------
    def transcribe_chunks(
        self,
        chunks: list[Chunk],
        context: TranscriptionContext,
        on_progress=None,
    ) -> TranscriptionResult:
        result = TranscriptionResult()
        if not chunks:
            return result
        batch_size = max(1, int(self.settings.stt_concurrency))
        carry = context.prompt
        done = 0
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                prompt = carry
                texts = list(
                    pool.map(lambda c, p=prompt: self._transcribe_one(c, context, p), batch)
                )
                for chunk, text in zip(batch, texts, strict=True):
                    result.requests += 1
                    result.transcribed_sec += chunk.duration_sec
                    if text:
                        result.segments.append(
                            TranscriptSegment(
                                index=len(result.segments),
                                start_sec=chunk.start_sec,
                                end_sec=chunk.end_sec,
                                text=text,
                                channel=chunk.channel,
                            )
                        )
                    done += 1
                    if on_progress is not None:
                        on_progress(done, len(chunks))
                joined = " ".join(t for t in texts if t)
                if joined:
                    carry = joined[-PROMPT_TAIL_CHARS:]
        result.segments.sort(key=lambda s: (s.start_sec, s.index))
        for i, seg in enumerate(result.segments):
            seg.index = i
        result.transcribed_sec = round(result.transcribed_sec, 3)
        return result
