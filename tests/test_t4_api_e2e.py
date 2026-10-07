"""T4 — API E2E: upload the fixture, wait for the pipeline, then check the questions
produced by the real LLM. Nothing is mocked."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from meetq.api.app import create_app
from meetq.qgen.base import QUESTION_TYPES
from meetq.qgen.validate import TranscriptIndex, normalize

pytestmark = pytest.mark.t4

TIMEOUT_SEC = 300


@pytest.fixture(scope="module")
def processed(request):
    wav = request.getfixturevalue("fixture_wav")
    manifest = request.getfixturevalue("manifest")
    client = TestClient(create_app())
    with wav.open("rb") as fh:
        response = client.post(
            "/api/sessions/upload",
            files={"file": (wav.name, fh, "audio/wav")},
            data={
                "title": "T4 fixture",
                "mode": "followup",
                "glossary": "\n".join(manifest["keywords"]),
            },
        )
    assert response.status_code == 200, response.text
    session_id = response.json()["id"]

    deadline = time.time() + TIMEOUT_SEC
    session = response.json()
    while time.time() < deadline:
        session = client.get(f"/api/sessions/{session_id}").json()
        if session["status"] in {"done", "error"}:
            break
        time.sleep(2)
    print(f"T4 final status={session['status']} stage={session['stage']} error={session['error']}")
    assert session["status"] == "done", f"pipeline did not finish: {session}"
    return {
        "client": client,
        "session_id": session_id,
        "session": session,
        "segments": client.get(f"/api/sessions/{session_id}/segments").json(),
        "questions": client.get(f"/api/sessions/{session_id}/questions").json(),
        "usage": client.get(f"/api/sessions/{session_id}/usage").json(),
    }


def test_upload_rejects_unsupported_types():
    client = TestClient(create_app())
    response = client.post(
        "/api/sessions/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_transcript_was_produced(processed):
    segments = processed["segments"]
    assert len(segments) >= 3
    assert all(s["text"].strip() for s in segments)


def test_at_least_three_questions_with_valid_schema(processed):
    questions = processed["questions"]
    print(f"T4 generated {len(questions)} questions")
    for q in questions:
        print(f"  [P{q['priority']}][{q['type']}] {q['text']} / 根拠: {q['evidence_quote']}")
    assert len(questions) >= 3, f"only {len(questions)} questions"
    for q in questions:
        assert set(q) >= {
            "text",
            "type",
            "priority",
            "evidence_quote",
            "evidence_start_sec",
            "evidence_segment_ids",
        }
        assert q["text"].strip()
        assert q["type"] in QUESTION_TYPES
        assert q["priority"] in (1, 2, 3)
        assert isinstance(q["evidence_segment_ids"], list) and q["evidence_segment_ids"]
        assert isinstance(q["evidence_start_sec"], (int, float))


def test_every_evidence_quote_is_a_real_substring(processed):
    index = TranscriptIndex(processed["segments"])
    normalized = index.normalized
    for q in processed["questions"]:
        assert normalize(q["evidence_quote"]) in normalized, (
            f"evidence not found in transcript: {q['evidence_quote']}"
        )
        located = index.locate(q["evidence_quote"])
        assert located is not None
        assert sorted(located[0]) == sorted(q["evidence_segment_ids"])
        assert located[1] == pytest.approx(q["evidence_start_sec"], abs=0.01)


def test_usage_is_recorded(processed):
    usage = processed["usage"]
    print(f"T4 usage={usage}")
    assert usage["transcribed_sec"] > 60.0
    assert usage["stt_requests"] >= 1
    assert usage["llm_input_tokens"] > 0
    assert usage["llm_output_tokens"] > 0
    assert usage["llm_requests"] >= 1
    assert processed["session"]["usage"]["transcribed_sec"] == usage["transcribed_sec"]


def test_exports_contain_questions_and_transcript(processed):
    client = processed["client"]
    session_id = processed["session_id"]
    md = client.get(f"/api/sessions/{session_id}/export?format=md")
    assert md.status_code == 200
    assert "attachment" in md.headers["content-disposition"]
    body = md.text
    assert "## 質問" in body and "## 文字起こし" in body
    first = processed["questions"][0]["text"]
    assert first in body

    js = client.get(f"/api/sessions/{session_id}/export?format=json")
    assert js.status_code == 200
    data = js.json()
    assert len(data["questions"]) == len(processed["questions"])
    assert len(data["segments"]) == len(processed["segments"])


def test_regeneration_produces_questions_again(processed):
    client = processed["client"]
    session_id = processed["session_id"]
    response = client.post(f"/api/sessions/{session_id}/regenerate", json={"mode": "comprehension"})
    assert response.status_code == 200
    deadline = time.time() + TIMEOUT_SEC
    while time.time() < deadline:
        session = client.get(f"/api/sessions/{session_id}").json()
        if session["status"] in {"done", "error"}:
            break
        time.sleep(2)
    assert session["status"] == "done", session
    questions = client.get(f"/api/sessions/{session_id}/questions").json()
    print(f"T4 regenerated {len(questions)} comprehension questions")
    assert questions
    assert all(q["mode"] == "comprehension" for q in questions)
    index = TranscriptIndex(processed["segments"])
    for q in questions:
        assert index.locate(q["evidence_quote"]) is not None
