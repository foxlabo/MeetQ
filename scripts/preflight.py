"""Preflight: OPENAI_API_KEY presence + a minimal real API round trip.

Exits 0 on success. On failure prints the reason (never the key) and exits 1.
A 200 from GET /v1/models is not enough: an account with an exhausted credit
balance still answers 200 there while every inference call fails, so this also
makes the smallest possible inference request.
"""

from __future__ import annotations

import sys

from meetq.config import get_settings


def main() -> int:
    settings = get_settings()
    if not settings.key_configured:
        print("PREFLIGHT FAILED: OPENAI_API_KEY is not set (.env)")
        return 1

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, timeout=60.0)

    try:
        client.models.list()
    except Exception as exc:  # noqa: BLE001
        print(f"PREFLIGHT FAILED: cannot reach the OpenAI API ({type(exc).__name__}: {exc})")
        return 1

    probe_model = sys.argv[1] if len(sys.argv) > 1 else "gpt-5.4-nano"
    try:
        client.responses.create(model=probe_model, input="ping", max_output_tokens=16)
    except Exception as exc:  # noqa: BLE001
        print(f"PREFLIGHT FAILED: inference call rejected ({type(exc).__name__}: {exc})")
        return 1

    print("PREFLIGHT OK: API key present, models reachable, inference call succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
