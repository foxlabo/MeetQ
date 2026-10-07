"""Audio source interface shared by file input and live capture."""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000  # everything in MeetQ is normalised to 16 kHz mono int16

Channel = str  # "self" | "others" | "mixed"


@dataclass
class AudioBuffer:
    """A contiguous mono int16 PCM buffer for one channel."""

    pcm: np.ndarray
    channel: Channel = "mixed"
    sample_rate: int = SAMPLE_RATE
    start_sec: float = 0.0

    @property
    def duration_sec(self) -> float:
        return len(self.pcm) / float(self.sample_rate)


@dataclass
class Chunk:
    """A VAD-delimited slice of audio. start_sec is owned by MeetQ, not by the STT model."""

    index: int
    start_sec: float
    end_sec: float
    pcm: np.ndarray
    channel: Channel = "mixed"
    sample_rate: int = SAMPLE_RATE
    meta: dict = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


class AudioSource(abc.ABC):
    """Produces mono int16 PCM at SAMPLE_RATE."""

    sample_rate: int = SAMPLE_RATE

    @abc.abstractmethod
    def buffers(self) -> Iterator[AudioBuffer]:
        """Yield one buffer per channel (files) or streamed blocks (live capture)."""

    def read_all(self) -> list[AudioBuffer]:
        return list(self.buffers())
