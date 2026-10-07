"""VAD-based chunker.

Chunks are delimited at silence. Each chunk carries its own start second, which is
what MeetQ uses for segment timestamps. Model-side timestamps exist only on whisper-1,
and sending a whole file for them would hit the 25 MB upload limit.
"""

from __future__ import annotations

import abc

import numpy as np

from meetq.audio.base import SAMPLE_RATE, Chunk

FRAME_MS = 30


class VadBackend(abc.ABC):
    """Returns True when a 30 ms frame contains speech."""

    name = "base"

    @abc.abstractmethod
    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool: ...


class WebRtcVad(VadBackend):
    name = "webrtcvad"

    def __init__(self, aggressiveness: int = 2):
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        return bool(self._vad.is_speech(frame.astype(np.int16).tobytes(), sample_rate))


class EnergyVad(VadBackend):
    """Fallback backend with no native dependency, for when webrtcvad is unavailable."""

    name = "energy"

    def __init__(self, threshold: float = 300.0):
        self.threshold = threshold

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        if len(frame) == 0:
            return False
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) > self.threshold


def default_vad() -> VadBackend:
    try:
        return WebRtcVad()
    except Exception:  # pragma: no cover - only on platforms without the wheel
        return EnergyVad()


class VadChunker:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        target_sec: float = 20.0,
        min_sec: float = 10.0,
        max_sec: float = 60.0,
        silence_sec: float = 0.4,
        vad: VadBackend | None = None,
    ):
        if not (0 < min_sec <= target_sec <= max_sec):
            raise ValueError("expected 0 < min_sec <= target_sec <= max_sec")
        self.sample_rate = sample_rate
        self.target_sec = target_sec
        self.min_sec = min_sec
        self.max_sec = max_sec
        self.silence_sec = silence_sec
        self.vad = vad or default_vad()

    @property
    def frame_len(self) -> int:
        return int(self.sample_rate * FRAME_MS / 1000)

    def chunk(self, pcm: np.ndarray, channel: str = "mixed") -> list[Chunk]:
        pcm = np.asarray(pcm, dtype=np.int16)
        if len(pcm) == 0:
            return []
        flen = self.frame_len
        n_frames = len(pcm) // flen
        speech: list[bool] = []
        for i in range(n_frames):
            frame = pcm[i * flen : (i + 1) * flen]
            speech.append(self.vad.is_speech(frame, self.sample_rate))

        frame_sec = flen / self.sample_rate
        silence_frames = max(1, int(round(self.silence_sec / frame_sec)))
        max_frames = int(self.max_sec / frame_sec)
        min_frames = int(self.min_sec / frame_sec)
        target_frames = int(self.target_sec / frame_sec)

        boundaries: list[tuple[int, int]] = []  # [start_frame, end_frame)
        start = 0
        run = 0
        for i, is_speech in enumerate(speech):
            length = i - start + 1
            run = run + 1 if not is_speech else 0
            if length >= max_frames:
                boundaries.append((start, i + 1))
                start = i + 1
                run = 0
                continue
            if run >= silence_frames and length >= min_frames:
                # cut inside the silence so speech is never clipped
                boundaries.append((start, i + 1))
                start = i + 1
                run = 0
                continue
            if length >= target_frames and run >= 1:
                boundaries.append((start, i + 1))
                start = i + 1
                run = 0
        if start < n_frames or not boundaries:
            boundaries.append((start, max(n_frames, start + 1)))

        chunks: list[Chunk] = []
        total_samples = len(pcm)
        for idx, (fs, fe) in enumerate(boundaries):
            s0 = fs * flen
            s1 = min(fe * flen, total_samples)
            if idx == len(boundaries) - 1:
                s1 = total_samples  # never drop the tail remainder
            if s1 <= s0:
                continue
            piece = pcm[s0:s1]
            has_speech = any(speech[fs:fe]) if speech[fs:fe] else False
            chunks.append(
                Chunk(
                    index=len(chunks),
                    start_sec=round(s0 / self.sample_rate, 3),
                    end_sec=round(s1 / self.sample_rate, 3),
                    pcm=piece,
                    channel=channel,
                    sample_rate=self.sample_rate,
                    meta={"has_speech": has_speech},
                )
            )
        return chunks
