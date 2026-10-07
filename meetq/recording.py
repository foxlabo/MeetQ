"""Active recording registry (F2), with optional live transcription (F8)."""

from __future__ import annotations

import threading

from meetq.audio.base import AudioBuffer
from meetq.audio.loopback import LoopbackSource
from meetq.live import LiveController, live_registry


class Recorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[int, LoopbackSource] = {}

    def start(
        self,
        session_id: int,
        loopback_index: int | None = None,
        mic_index: int | None = None,
        live: bool = False,
        language: str = "ja",
        keywords: list[str] | None = None,
        mode: str = "followup",
    ) -> list[str]:
        if live:
            controller = LiveController(
                session_id=session_id,
                language=language,
                keywords=keywords or [],
                mode=mode,
            )
            channels = controller.start(loopback_index, mic_index)
            live_registry.add(session_id, controller)
            return channels

        source = LoopbackSource(loopback_index=loopback_index, mic_index=mic_index)
        source.start()
        with self._lock:
            self._active[session_id] = source
        return ["others"] + (["self"] if mic_index is not None else [])

    def stop(self, session_id: int) -> list[AudioBuffer]:
        controller = live_registry.pop(session_id)
        if controller is not None:
            return controller.stop()
        with self._lock:
            source = self._active.pop(session_id)
        return source.stop()

    def active(self) -> list[int]:
        with self._lock:
            return sorted(set(self._active) | set(live_registry.active()))


recorder = Recorder()
