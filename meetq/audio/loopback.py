"""LoopbackSource: WASAPI loopback (speakers, "others") + microphone ("self").

The two are captured as separate channels so the transcript can attribute each
segment (F2 / F3).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from meetq.audio.base import SAMPLE_RATE, AudioBuffer, AudioSource

log = logging.getLogger(__name__)

# PortAudio's initialisation and stream setup are not thread safe: opening the
# loopback and the microphone at the same time crashes the process with an access
# violation. Serialise every PortAudio entry point that allocates.
_PORTAUDIO_LOCK = threading.Lock()


@dataclass
class DeviceInfo:
    index: int
    name: str
    kind: str  # "loopback" | "input"
    channels: int
    default_sample_rate: float
    is_default: bool = False

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "kind": self.kind,
            "channels": self.channels,
            "default_sample_rate": self.default_sample_rate,
            "is_default": self.is_default,
        }


class AudioBackendUnavailable(RuntimeError):
    pass


def _pyaudio():
    try:
        import pyaudiowpatch as pyaudio  # type: ignore
    except Exception as exc:  # pragma: no cover - non-Windows / missing package
        raise AudioBackendUnavailable(f"PyAudioWPatch unavailable: {exc}") from exc
    return pyaudio


def list_devices() -> dict:
    """Enumerate WASAPI loopback devices and microphones (F2 device list API)."""
    pyaudio = _pyaudio()
    loopbacks: list[DeviceInfo] = []
    inputs: list[DeviceInfo] = []
    default_loopback_index: int | None = None
    default_input_index: int | None = None
    with _PORTAUDIO_LOCK:
        p = pyaudio.PyAudio()
    try:
        try:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_output = p.get_device_info_by_index(int(wasapi["defaultOutputDevice"]))
        except Exception:
            default_output = None
        for info in p.get_loopback_device_info_generator():
            is_default = bool(
                default_output and str(default_output["name"]) in str(info["name"])
            )
            if is_default and default_loopback_index is None:
                default_loopback_index = int(info["index"])
            loopbacks.append(
                DeviceInfo(
                    index=int(info["index"]),
                    name=str(info["name"]),
                    kind="loopback",
                    channels=int(info["maxInputChannels"]),
                    default_sample_rate=float(info["defaultSampleRate"]),
                    is_default=is_default,
                )
            )
        try:
            default_input_index = int(p.get_default_input_device_info()["index"])
        except Exception:
            default_input_index = None
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) <= 0:
                continue
            if info.get("isLoopbackDevice"):
                continue
            inputs.append(
                DeviceInfo(
                    index=int(info["index"]),
                    name=str(info["name"]),
                    kind="input",
                    channels=int(info["maxInputChannels"]),
                    default_sample_rate=float(info["defaultSampleRate"]),
                    is_default=int(info["index"]) == default_input_index,
                )
            )
    finally:
        with _PORTAUDIO_LOCK:
            p.terminate()
    if default_loopback_index is None and loopbacks:
        default_loopback_index = loopbacks[0].index
        for d in loopbacks:
            d.is_default = d.index == default_loopback_index
    return {
        "loopback": [d.as_dict() for d in loopbacks],
        "input": [d.as_dict() for d in inputs],
        "default_loopback_index": default_loopback_index,
        "default_input_index": default_input_index,
    }


def _to_mono_16k(raw: bytes, channels: int, rate: int) -> np.ndarray:
    data = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        usable = (len(data) // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    data = data.astype(np.float32)
    if rate != SAMPLE_RATE and len(data) > 1:
        target_len = max(1, int(round(len(data) * SAMPLE_RATE / rate)))
        src = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        dst = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        data = np.interp(dst, src, data)
    return np.clip(data, -32768, 32767).astype(np.int16)


class _Capture(threading.Thread):
    def __init__(
        self,
        device_index: int,
        channel: str,
        frames_per_buffer: int = 1024,
        on_block: Callable[[str, np.ndarray], None] | None = None,
    ):
        super().__init__(daemon=True)
        self.device_index = device_index
        self.channel = channel
        self.frames_per_buffer = frames_per_buffer
        self.on_block = on_block
        self._stop_event = threading.Event()
        self._parts: list[np.ndarray] = []
        self.error: Exception | None = None

    def run(self) -> None:
        pyaudio = _pyaudio()
        stream = None
        p = None
        try:
            with _PORTAUDIO_LOCK:
                p = pyaudio.PyAudio()
                info = p.get_device_info_by_index(self.device_index)
                channels = max(1, int(info["maxInputChannels"]))
                rate = int(info["defaultSampleRate"])
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.frames_per_buffer,
                )
            while not self._stop_event.is_set():
                raw = stream.read(self.frames_per_buffer, exception_on_overflow=False)
                block = _to_mono_16k(raw, channels, rate)
                self._parts.append(block)
                if self.on_block is not None:
                    try:
                        self.on_block(self.channel, block)
                    except Exception:  # pragma: no cover - never kill the capture
                        log.exception("on_block callback failed")
        except Exception as exc:  # pragma: no cover - device dependent
            self.error = exc
        finally:
            with _PORTAUDIO_LOCK:
                if stream is not None:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass
                if p is not None:
                    p.terminate()

    def stop(self) -> np.ndarray:
        self._stop_event.set()
        self.join(timeout=5)
        if not self._parts:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._parts).astype(np.int16)


class LoopbackSource(AudioSource):
    """Captures speaker output and (optionally) the microphone as separate channels."""

    def __init__(
        self,
        loopback_index: int | None = None,
        mic_index: int | None = None,
        on_block: Callable[[str, np.ndarray], None] | None = None,
    ):
        devices = list_devices()
        self.loopback_index = (
            loopback_index if loopback_index is not None else devices["default_loopback_index"]
        )
        self.mic_index = mic_index
        if self.loopback_index is None:
            raise AudioBackendUnavailable("no WASAPI loopback device found")
        self.on_block = on_block
        self._captures: list[_Capture] = []
        self._result: list[AudioBuffer] = []

    def start(self) -> None:
        self._captures = [_Capture(int(self.loopback_index), "others", on_block=self.on_block)]
        if self.mic_index is not None:
            self._captures.append(_Capture(int(self.mic_index), "self", on_block=self.on_block))
        for c in self._captures:
            c.start()

    def stop(self) -> list[AudioBuffer]:
        self._result = []
        for c in self._captures:
            pcm = c.stop()
            if c.error is not None and len(pcm) == 0:
                raise c.error
            self._result.append(AudioBuffer(pcm=pcm, channel=c.channel))
        self._captures = []
        return self._result

    def buffers(self) -> Iterator[AudioBuffer]:
        yield from self._result
