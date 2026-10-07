"""T5 (UI) — term corrections and the transcript download, driven through the browser.

The session is written straight into the server's database, so this test makes no
OpenAI calls: corrections and the text export are purely local features.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as OrmSession

from meetq.db.migrate import migrate
from meetq.db.models import Base, Question, Segment, Session

pytestmark = pytest.mark.t5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MISHEARD = [
    "では以降の使用変更、使用策定についてお話しします。",
    "仕様書のレビューは大切です。",
    "使用書を見たときにはここを確認します。",
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _seed(db_path: Path) -> int:
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    Base.metadata.create_all(engine)
    migrate(engine)
    with OrmSession(engine) as db:
        sess = Session(title="定例ミーティング（誤認識あり）", source_type="upload", status="done")
        db.add(sess)
        db.flush()
        seg_ids = []
        for i, text in enumerate(MISHEARD):
            seg = Segment(
                session_id=sess.id,
                idx=i,
                start_sec=i * 20.0,
                end_sec=i * 20.0 + 20.0,
                text=text,
                raw_text=text,
            )
            db.add(seg)
            db.flush()
            seg_ids.append(seg.id)
        db.add(
            Question(
                session_id=sess.id,
                text="使用策定とは何ですか?",
                type="clarify",
                priority=1,
                evidence_quote="使用変更、使用策定",
                evidence_segment_ids=json.dumps([seg_ids[0]]),
            )
        )
        db.commit()
        session_id = sess.id
    engine.dispose()
    return session_id


@pytest.fixture(scope="module")
def seeded_server():
    assert (PROJECT_ROOT / "meetq" / "static" / "index.html").is_file(), "build the frontend"
    data_dir = Path(tempfile.mkdtemp(prefix="meetq-corr-ui-"))
    session_id = _seed(data_dir / "meetq.sqlite3")
    port = _free_port()
    env = {**os.environ, "MEETQ_DATA_DIR": str(data_dir), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "meetq.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=2) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(0.3)
    else:
        proc.terminate()
        raise RuntimeError("server did not become ready")
    try:
        yield base, session_id
    finally:
        proc.terminate()
        proc.wait(timeout=20)


def _transcript_text(page) -> str:
    return page.get_by_test_id("transcript").inner_text()


def _wait_for_transcript(page, needle: str) -> None:
    page.wait_for_function(
        "n => document.querySelector('[data-testid=\"transcript\"]').innerText.includes(n)",
        arg=needle,
    )


def test_preview_apply_download_and_revert(seeded_server, page, tmp_path):
    base, session_id = seeded_server
    page.set_default_timeout(15_000)
    page.goto(f"{base}/sessions/{session_id}")
    page.wait_for_selector('[data-testid="transcript"]')
    assert "使用策定" in _transcript_text(page)

    # preview first: counts and the matched context, nothing changed yet
    page.get_by_test_id("corrections-open").click()
    page.get_by_test_id("corrections-input").fill("使用 => 仕様\n以降 => 移行")
    assert page.get_by_test_id("corrections-apply").is_disabled(), "apply needs a preview"
    page.get_by_test_id("corrections-preview").click()
    page.wait_for_selector('[data-testid="corrections-preview-result"]')
    assert page.get_by_test_id("corrections-count-使用").inner_text() == "3"
    assert page.get_by_test_id("corrections-count-以降").inner_text() == "1"
    assert "使用策定" in _transcript_text(page)

    # apply: the transcript and the question are fixed in place
    page.get_by_test_id("corrections-apply").click()
    page.wait_for_selector('[data-testid="corrections-result"]')
    print(f"T5 corrections: {page.get_by_test_id('corrections-result').inner_text()}")
    _wait_for_transcript(page, "仕様策定")
    transcript = _transcript_text(page)
    assert "移行の仕様変更、仕様策定" in transcript
    assert "使用" not in transcript and "以降" not in transcript
    question = page.locator('.question[data-testid^="question-"]').first
    assert "仕様策定とは何ですか" in question.inner_text()

    # clicking the question still lands on its (re-anchored) evidence
    question.click()
    page.wait_for_selector('[data-highlighted="true"]')
    assert "仕様策定" in page.locator('[data-highlighted="true"]').first.inner_text()

    # the transcript download carries the corrected text
    with page.expect_download() as info:
        page.get_by_test_id("export-txt").click()
    download = info.value
    target = tmp_path / download.suggested_filename
    download.save_as(target)
    body = target.read_text(encoding="utf-8")
    print(f"T5 transcript download: {download.suggested_filename} ({len(body)} chars)")
    assert download.suggested_filename.endswith("_transcript.txt")
    assert body.splitlines()[0] == "[00:00] では移行の仕様変更、仕様策定についてお話しします。"
    assert "使用" not in body

    # clearing the rules and applying again restores the original
    page.get_by_test_id("corrections-input").fill("")
    page.get_by_test_id("corrections-preview").click()
    page.get_by_test_id("corrections-apply").click()
    _wait_for_transcript(page, "使用策定")
    assert "以降の使用変更" in _transcript_text(page)
