"""FastAPI application: REST/WebSocket API plus the built frontend."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from meetq.api.progress import hub
from meetq.api.routes import router
from meetq.db.session import init_db

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="MeetQ", version="0.1.0")
    app.include_router(router)

    @app.on_event("startup")
    async def _startup() -> None:
        init_db()
        hub.bind_loop(asyncio.get_running_loop())

    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{full_path:path}")
        async def spa(full_path: str):
            candidate = STATIC_DIR / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            index = STATIC_DIR / "index.html"
            if index.is_file():
                return FileResponse(index)
            return JSONResponse({"detail": "frontend not built"}, status_code=404)
    else:  # pragma: no cover - dev without a frontend build

        @app.get("/")
        async def root():
            return JSONResponse(
                {"detail": "frontend not built; run `npm run build` in frontend/"},
                status_code=200,
            )

    return app


app = create_app()
