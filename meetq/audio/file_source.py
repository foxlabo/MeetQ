"""FileSource: decode mp4 / mp3 / wav / m4a to 16 kHz mono int16 with PyAV."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from meetq.audio.base import SAMPLE_RATE, AudioBuffer, AudioSource

SUPPORTED_SUFFIXES = {".mp4", ".mp3", ".wav", ".m4a"}


def decode_to_pcm(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any container PyAV understands into mono int16 PCM."""
    import av  # imported lazily so unit tests that never decode stay fast

    parts: list[np.ndarray] = []
    with av.open(str(path)) as container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            raise ValueError(f"no audio stream in {path.name}")
        stream = streams[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                parts.append(resampled.to_ndarray().reshape(-1).astype(np.int16))
        for resampled in resampler.resample(None):
            parts.append(resampled.to_ndarray().reshape(-1).astype(np.int16))
    if not parts:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(parts).astype(np.int16)


class FileSource(AudioSource):
    def __init__(self, path: Path, channel: str = "mixed", sample_rate: int = SAMPLE_RATE):
        self.path = Path(path)
        self.channel = channel
        self.sample_rate = sample_rate

    def buffers(self) -> Iterator[AudioBuffer]:
        pcm = decode_to_pcm(self.path, self.sample_rate)
        yield AudioBuffer(pcm=pcm, channel=self.channel, sample_rate=self.sample_rate)
