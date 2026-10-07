"""Progress fan-out from worker threads to WebSocket clients."""

from __future__ import annotations

import asyncio
from collections import defaultdict


class ProgressHub:
    def __init__(self) -> None:
        self._subscribers: dict[int, set[asyncio.Queue]] = defaultdict(set)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last: dict[int, dict] = {}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, session_id: int) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers[session_id].add(queue)
        last = self._last.get(session_id)
        if last is not None:
            queue.put_nowait(last)
        return queue

    def unsubscribe(self, session_id: int, queue: asyncio.Queue) -> None:
        self._subscribers[session_id].discard(queue)

    def publish(self, session_id: int, payload: dict) -> None:
        """Safe to call from any thread."""
        self._last[session_id] = payload
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._deliver, session_id, payload)
        except RuntimeError:  # pragma: no cover - loop shutting down
            pass

    def _deliver(self, session_id: int, payload: dict) -> None:
        for queue in list(self._subscribers.get(session_id, ())):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:  # pragma: no cover - slow client
                pass


hub = ProgressHub()
