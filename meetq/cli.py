"""`poetry run meetq` entry point."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from meetq.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="meetq", description="MeetQ local web UI")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())
    if not settings.key_configured:
        logging.getLogger("meetq").warning(
            "OPENAI_API_KEY is not set; transcription and question generation will fail."
        )
    print(f"MeetQ -> http://{args.host}:{args.port}")
    uvicorn.run("meetq.api.app:app", host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":  # pragma: no cover
    main()
