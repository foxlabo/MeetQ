"""REST + WebSocket API."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from meetq import appsettings, corrections, service, transcript
from meetq.audio.file_source import SUPPORTED_SUFFIXES
from meetq.audio.wavio import write_wav
from meetq.config import get_settings
from meetq.db.models import Session as SessionRow
from meetq.db.models import Usage
from meetq.db.session import db_session
from meetq.live import live_registry
from meetq.recording import recorder

router = APIRouter(prefix="/api")


# --------------------------------------------------------------------- models
class SettingsPayload(BaseModel):
    stt_model: str | None = None
    stt_live_model: str | None = None
    llm_model: str | None = None
    language: str | None = None
    glossary: str | None = None
    corrections: str | None = None


class SessionPatch(BaseModel):
    title: str | None = None
    glossary: str | None = None
    language: str | None = None
    mode: str | None = None


class RegeneratePayload(BaseModel):
    mode: str = "followup"


class CorrectionsPayload(BaseModel):
    rules: str = ""
    save_global: bool = False  # also store as the default for future transcripts


class StartRecordingPayload(BaseModel):
    title: str = ""
    loopback_index: int | None = None
    mic_index: int | None = None
    language: str | None = None
    glossary: str | None = None
    mode: str = "followup"
    live: bool = True  # F8: stream to the live transcription API while recording


# --------------------------------------------------------------------- helpers
def _session_dict(sess: SessionRow) -> dict:
    usage = sess.usage
    return {
        "id": sess.id,
        "title": sess.title,
        "source_type": sess.source_type,
        "status": sess.status,
        "stage": sess.stage,
        "progress": round(sess.progress, 4),
        "error": sess.error,
        "duration_sec": round(sess.duration_sec, 2),
        "language": sess.language,
        "glossary": sess.glossary,
        "mode": sess.mode,
        "live": bool(sess.live),
        "summary": sess.summary,
        "corrections": sess.corrections,
        "created_at": sess.created_at.isoformat(),
        "segment_count": len(sess.segments),
        "question_count": len(sess.questions),
        "usage": {
            "transcribed_sec": round(usage.transcribed_sec, 2) if usage else 0.0,
            "stt_requests": usage.stt_requests if usage else 0,
            "llm_input_tokens": usage.llm_input_tokens if usage else 0,
            "llm_output_tokens": usage.llm_output_tokens if usage else 0,
            "llm_requests": usage.llm_requests if usage else 0,
        },
    }


def _segment_dict(seg) -> dict:
    return {
        "id": seg.id,
        "idx": seg.idx,
        "start_sec": round(seg.start_sec, 2),
        "end_sec": round(seg.end_sec, 2),
        "text": seg.text,
        "channel": seg.channel,
    }


def _question_dict(q) -> dict:
    return {
        "id": q.id,
        "text": q.text,
        "type": q.type,
        "priority": q.priority,
        "evidence_quote": q.evidence_quote,
        "evidence_start_sec": round(q.evidence_start_sec, 2),
        "evidence_segment_ids": json.loads(q.evidence_segment_ids or "[]"),
        "mode": q.mode,
        "round": q.round,
    }


def _clock(seconds: float) -> str:
    """mm:ss, or h:mm:ss once a recording runs past an hour."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _get_session(db, session_id: int) -> SessionRow:
    sess = db.get(SessionRow, session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    return sess


# --------------------------------------------------------------------- routes
@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/settings")
def read_settings() -> dict:
    values = appsettings.all_settings()
    values["api_key_configured"] = get_settings().key_configured  # never the key itself
    return values


@router.put("/settings")
def write_settings(payload: SettingsPayload) -> dict:
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if "corrections" in updates:
        _, errors = corrections.parse_rules(updates["corrections"])
        if errors:
            raise HTTPException(status_code=400, detail=" / ".join(errors))
    values = appsettings.save(updates)
    values["api_key_configured"] = get_settings().key_configured
    return values


@router.get("/devices")
def devices() -> dict:
    from meetq.audio.loopback import AudioBackendUnavailable, list_devices

    try:
        return {"available": True, **list_devices()}
    except AudioBackendUnavailable as exc:
        return {
            "available": False,
            "reason": str(exc),
            "loopback": [],
            "input": [],
            "default_loopback_index": None,
            "default_input_index": None,
        }


@router.get("/sessions")
def list_sessions() -> list[dict]:
    with db_session() as db:
        rows = db.query(SessionRow).order_by(SessionRow.id.desc()).all()
        return [_session_dict(s) for s in rows]


@router.get("/sessions/{session_id}")
def get_session(session_id: int) -> dict:
    with db_session() as db:
        return _session_dict(_get_session(db, session_id))


@router.patch("/sessions/{session_id}")
def patch_session(session_id: int, payload: SessionPatch) -> dict:
    with db_session() as db:
        sess = _get_session(db, session_id)
        for key, value in payload.model_dump().items():
            if value is not None:
                setattr(sess, key, value)
        db.flush()
        return _session_dict(sess)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: int) -> dict:
    with db_session() as db:
        sess = _get_session(db, session_id)
        db.delete(sess)
    return {"deleted": session_id}


@router.post("/sessions/upload")
async def upload(
    file: UploadFile = File(...),
    title: str = Form(""),
    mode: str = Form("followup"),
    language: str | None = Form(None),
    glossary: str | None = Form(None),
) -> dict:
    """F1: mp4 / mp3 / wav / m4a upload."""
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type {suffix}; allowed: "
            + ", ".join(sorted(SUPPORTED_SUFFIXES)),
        )
    defaults = appsettings.all_settings()
    settings = get_settings()
    with db_session() as db:
        sess = SessionRow(
            title=title or Path(file.filename or "session").stem,
            source_type="upload",
            status="queued",
            stage="待機中",
            language=language or defaults["language"],
            glossary=glossary if glossary is not None else defaults["glossary"],
            mode=mode,
            channels=json.dumps(["mixed"]),
            corrections=defaults["corrections"],
        )
        db.add(sess)
        db.flush()
        session_id = sess.id
        dest = settings.media_dir / f"session_{session_id}{suffix}"
        sess.audio_path = str(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    await file.close()

    service.start_processing(session_id)
    with db_session() as db:
        return _session_dict(_get_session(db, session_id))


@router.get("/sessions/{session_id}/segments")
def segments(session_id: int) -> list[dict]:
    with db_session() as db:
        sess = _get_session(db, session_id)
        return [_segment_dict(s) for s in sorted(sess.segments, key=lambda s: s.start_sec)]


@router.get("/sessions/{session_id}/questions")
def questions(session_id: int) -> list[dict]:
    with db_session() as db:
        sess = _get_session(db, session_id)
        rows = sorted(sess.questions, key=lambda q: (q.priority, q.evidence_start_sec))
        return [_question_dict(q) for q in rows]


@router.get("/sessions/{session_id}/usage")
def usage(session_id: int) -> dict:
    with db_session() as db:
        sess = _get_session(db, session_id)
        row: Usage | None = sess.usage
        return {
            "transcribed_sec": round(row.transcribed_sec, 2) if row else 0.0,
            "stt_requests": row.stt_requests if row else 0,
            "llm_input_tokens": row.llm_input_tokens if row else 0,
            "llm_output_tokens": row.llm_output_tokens if row else 0,
            "llm_requests": row.llm_requests if row else 0,
        }


@router.post("/sessions/{session_id}/regenerate")
def regenerate(session_id: int, payload: RegeneratePayload) -> dict:
    with db_session() as db:
        sess = _get_session(db, session_id)
        if not sess.segments:
            raise HTTPException(status_code=400, detail="no transcript yet")
    service.start_regeneration(session_id, payload.mode)
    return {"started": True, "mode": payload.mode}


@router.post("/sessions/{session_id}/corrections/preview")
def corrections_preview(session_id: int, payload: CorrectionsPayload) -> dict:
    """Show where each rule would fire, before anything is changed."""
    rules, errors = corrections.parse_rules(payload.rules)
    with db_session() as db:
        sess = _get_session(db, session_id)
        segs = [
            {
                "id": s.id,
                "start_sec": s.start_sec,
                "text": s.text,
                "raw_text": s.raw_text,
            }
            for s in sorted(sess.segments, key=lambda s: s.start_sec)
        ]
    items = corrections.preview(segs, rules)
    return {
        "errors": errors,
        "rules": [
            {
                "wrong": item.wrong,
                "right": item.right,
                "count": item.count,
                "examples": item.examples,
            }
            for item in items
        ],
        "total": sum(item.count for item in items),
    }


@router.post("/sessions/{session_id}/corrections")
def corrections_apply(session_id: int, payload: CorrectionsPayload) -> dict:
    """Apply term corrections to this session (and optionally keep them as defaults)."""
    with db_session() as db:
        sess = _get_session(db, session_id)
        if sess.status not in {"done", "error"}:
            raise HTTPException(
                status_code=409,
                detail="録音中・処理中は修正できません。完了してから適用してください",
            )
    try:
        stats = service.apply_session_corrections(session_id, payload.rules)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.save_global:
        appsettings.save({"corrections": payload.rules})
    return {"applied": True, **stats}


@router.get("/sessions/{session_id}/export")
def export(session_id: int, format: str = "md"):
    with db_session() as db:
        sess = _get_session(db, session_id)
        data = {
            "session": _session_dict(sess),
            "segments": [
                _segment_dict(s) for s in sorted(sess.segments, key=lambda s: s.start_sec)
            ],
            "questions": [
                _question_dict(q)
                for q in sorted(sess.questions, key=lambda q: (q.priority, q.evidence_start_sec))
            ],
        }
    # the text downloads are re-flowed to one sentence per line; JSON keeps the
    # segments as they are, since questions point at segment ids
    sentences = transcript.reflow(data["segments"])
    if format == "txt":
        lines = [f"[{_clock(s.start_sec)}] {s.text}" for s in sentences]
        return PlainTextResponse(
            "\n".join(lines) + "\n",
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="meetq_session_{session_id}_transcript.txt"'
                )
            },
        )
    if format == "json":
        return JSONResponse(
            data,
            headers={
                "Content-Disposition": f'attachment; filename="meetq_session_{session_id}.json"'
            },
        )
    lines = [f"# {data['session']['title'] or f'Session {session_id}'}", ""]
    lines.append(f"- 作成: {data['session']['created_at']}")
    lines.append(f"- 長さ: {data['session']['duration_sec']} 秒")
    u = data["session"]["usage"]
    lines.append(
        f"- 利用量: 転写 {u['transcribed_sec']} 秒 / LLM "
        f"{u['llm_input_tokens']}+{u['llm_output_tokens']} tokens"
    )
    lines += ["", "## 質問", ""]
    for q in data["questions"]:
        ts = q["evidence_start_sec"]
        lines.append(
            f"- **[P{q['priority']}][{q['type']}]** {q['text']}  \n"
            f"  根拠 ({int(ts // 60):02d}:{int(ts % 60):02d}): 「{q['evidence_quote']}」"
        )
    lines += ["", "## 文字起こし", ""]
    for s in sentences:
        lines.append(f"- `{_clock(s.start_sec)}` [{s.channel}] {s.text}")
    return PlainTextResponse(
        "\n".join(lines) + "\n",
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="meetq_session_{session_id}.md"'},
    )


# --------------------------------------------------------------- recording (F2)
@router.post("/recording/start")
def recording_start(payload: StartRecordingPayload) -> dict:
    defaults = appsettings.all_settings()
    with db_session() as db:
        sess = SessionRow(
            title=payload.title or "録音セッション",
            source_type="record",
            status="recording",
            stage="録音中",
            language=payload.language or defaults["language"],
            glossary=payload.glossary if payload.glossary is not None else defaults["glossary"],
            mode=payload.mode,
            corrections=defaults["corrections"],
        )
        db.add(sess)
        db.flush()
        session_id = sess.id
    raw_glossary = payload.glossary if payload.glossary is not None else defaults["glossary"]
    glossary_terms = [t.strip() for t in (raw_glossary or "").splitlines() if t.strip()]
    for term in corrections.targets(defaults["corrections"]):
        if term not in glossary_terms:
            glossary_terms.append(term)
    try:
        channels = recorder.start(
            session_id,
            payload.loopback_index,
            payload.mic_index,
            live=payload.live,
            language=payload.language or defaults["language"],
            keywords=glossary_terms,
            mode=payload.mode,
        )
    except Exception as exc:  # noqa: BLE001
        with db_session() as db:
            sess = _get_session(db, session_id)
            sess.status = "error"
            sess.error = str(exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with db_session() as db:
        sess = _get_session(db, session_id)
        sess.channels = json.dumps(channels)
        sess.live = payload.live
        sess.stage = "録音中（ライブ転写）" if payload.live else "録音中"
        db.flush()
        return _session_dict(sess)


@router.post("/recording/{session_id}/stop")
def recording_stop(session_id: int) -> dict:
    """Stop recording, save the WAV, then build the final full-text transcript.

    Live sessions (F8) also go through the file pipeline: the live transcript is
    committed every few seconds for speed, which clips words at the cut points, so
    the saved recording is re-transcribed at real silences and the questions are
    regenerated from that full text.
    """
    settings = get_settings()
    try:
        buffers = recorder.stop(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="not recording") from exc
    dest = settings.media_dir / f"session_{session_id}.wav"
    write_wav(dest, [b.pcm for b in buffers])
    with db_session() as db:
        sess = _get_session(db, session_id)
        sess.audio_path = str(dest)
        sess.channels = json.dumps([b.channel for b in buffers])
        sess.duration_sec = max(
            sess.duration_sec, max((b.duration_sec for b in buffers), default=0.0)
        )
        sess.status = "queued"
        sess.stage = "全文を文字起こし中" if sess.live else "待機中"
        sess.progress = 0.0

    service.start_processing(session_id)
    with db_session() as db:
        return _session_dict(_get_session(db, session_id))


@router.post("/sessions/{session_id}/generate-now")
def generate_now(session_id: int) -> dict:
    """F9: the UI's 今すぐ生成 button."""
    controller = live_registry.get(session_id)
    if controller is None:
        raise HTTPException(status_code=400, detail="no live session is running")
    service.start_manual_round(session_id, controller)
    return {"started": True, "rounds": controller.rounds}


@router.get("/sessions/{session_id}/live")
def live_status(session_id: int) -> dict:
    controller = live_registry.get(session_id)
    if controller is None:
        return {"running": False, "rounds": 0, "reconnects": 0, "finals": 0}
    return {
        "running": True,
        "rounds": controller.rounds,
        "reconnects": controller.reconnects,
        "finals": controller.finals,
        "interval_sec": get_settings().rolling_interval_sec,
    }


@router.get("/recording/active")
def recording_active() -> dict:
    return {"session_ids": recorder.active()}


# ------------------------------------------------------------------ websocket
@router.websocket("/ws/sessions/{session_id}")
async def session_ws(websocket: WebSocket, session_id: int) -> None:
    from meetq.api.progress import hub

    await websocket.accept()
    hub.bind_loop(asyncio.get_running_loop())
    queue = hub.subscribe(session_id)
    try:
        with db_session() as db:
            sess = db.get(SessionRow, session_id)
            if sess is not None:
                await websocket.send_json(
                    {
                        "session_id": session_id,
                        "status": sess.status,
                        "stage": sess.stage,
                        "progress": round(sess.progress, 4),
                        "error": sess.error,
                    }
                )
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception:  # pragma: no cover - client vanished
        pass
    finally:
        hub.unsubscribe(session_id, queue)
