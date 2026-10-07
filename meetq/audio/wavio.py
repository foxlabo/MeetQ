"""Minimal WAV helpers (stdlib `wave`, no ffmpeg dependency)."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from meetq.audio.base import SAMPLE_RATE


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
    return buf.getvalue()


def write_wav(path: Path, channels: list[np.ndarray], sample_rate: int = SAMPLE_RATE) -> None:
    """Write one or more mono int16 tracks as an interleaved WAV file."""
    if not channels:
        raise ValueError("no channels to write")
    length = max(len(c) for c in channels)
    padded = [
        np.pad(np.asarray(c, dtype=np.int16), (0, length - len(c)), mode="constant")
        for c in channels
    ]
    interleaved = np.stack(padded, axis=1).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(len(padded))
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.astype(np.int16).tobytes())


def read_wav(path: Path) -> tuple[list[np.ndarray], int]:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        data = data.reshape(-1, n_channels)
        return [np.ascontiguousarray(data[:, i]) for i in range(n_channels)], sample_rate
    return [data], sample_rate


def rms(pcm: np.ndarray) -> float:
    if len(pcm) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(pcm, dtype=np.float64) ** 2)))
