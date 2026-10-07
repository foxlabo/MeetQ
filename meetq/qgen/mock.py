"""Mock provider — unit tests (T1) only. Never used by T2..T6."""

from __future__ import annotations

from collections.abc import Callable

from meetq.qgen.base import LLMProvider, TokenUsage


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, responder: Callable[[str, str], dict] | list[dict]):
        self._responder = responder
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def generate_json(
        self, system: str, user: str, schema: dict, schema_name: str
    ) -> tuple[dict, TokenUsage]:
        idx = self._calls
        self._calls += 1
        if callable(self._responder):
            payload = self._responder(system, user)
        else:
            payload = self._responder[min(idx, len(self._responder) - 1)]
        return payload, TokenUsage(input_tokens=len(user) // 4, output_tokens=32, requests=1)
