"""OpenAI provider using the Responses API with Structured Outputs."""

from __future__ import annotations

import json
import logging
import random
import time

from meetq.config import Settings, get_settings
from meetq.qgen.base import LLMProvider, TokenUsage

log = logging.getLogger(__name__)


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
    return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}


def _extract_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            piece = getattr(content, "text", None)
            if piece:
                parts.append(piece)
    return "".join(parts)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, client=None, settings: Settings | None = None, model: str | None = None):
        self.settings = settings or get_settings()
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.settings.openai_api_key, timeout=self.settings.openai_timeout
            )
        self.client = client
        from meetq import appsettings

        self.model = model or appsettings.all_settings()["llm_model"]

    def generate_json(
        self, system: str, user: str, schema: dict, schema_name: str
    ) -> tuple[dict, TokenUsage]:
        delay = 1.0
        last: Exception | None = None
        for attempt in range(4):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "schema": schema,
                            "strict": True,
                        }
                    },
                )
                usage = getattr(response, "usage", None)
                tokens = TokenUsage(
                    input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                    requests=1,
                )
                return json.loads(_extract_text(response) or "{}"), tokens
            except Exception as exc:  # noqa: BLE001
                last = exc
                if not _is_retryable(exc) or attempt == 3:
                    raise
                time.sleep(delay + random.random() * 0.3)
                delay *= 2
        raise last  # pragma: no cover


def build_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    if settings.llm_provider == "mock":  # pragma: no cover - opt-in escape hatch
        from meetq.qgen.mock import MockProvider

        return MockProvider(lambda s, u: {"questions": []})
    return OpenAIProvider(settings=settings)
