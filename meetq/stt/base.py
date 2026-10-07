"""Transcriber interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from meetq.audio.base import Chunk


@dataclass
class TranscriptSegment:
    index: int
    start_sec: float
    end_sec: float
    text: str
    channel: str = "mixed"


@dataclass
class TranscriptionContext:
    language: str = "ja"
    keywords: list[str] = field(default_factory=list)
    prompt: str = ""


@dataclass
class TranscriptionResult:
    segments: list[TranscriptSegment] = field(default_factory=list)
    transcribed_sec: float = 0.0
    requests: int = 0


class Transcriber(abc.ABC):
    @abc.abstractmethod
    def transcribe_chunks(
        self,
        chunks: list[Chunk],
        context: TranscriptionContext,
        on_progress=None,
    ) -> TranscriptionResult: ...
