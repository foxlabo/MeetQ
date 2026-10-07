"""T5 — UI E2E with Playwright against a real uvicorn server and the real API.

Flow: upload -> questions appear -> click a question highlights its evidence
segment -> export downloads.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.t5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
READY_TIMEOUT = 60
PIPELINE_TIMEOUT_MS = 300_000


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def server():
    static = PROJECT_ROOT / "meetq" / "static" / "index.html"
    assert static.is_file(), "frontend build missing: run `npm run build` in frontend/"

    port = _free_port()
    data_dir = tempfile.mkdtemp(prefix="meetq-ui-")
    env = {**os.environ, "MEETQ_DATA_DIR": data_dir, "PYTHONIOENCODING": "utf-8"}
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
    deadline = time.time() + READY_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=2) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(0.4)
    else:
        proc.terminate()
        raise RuntimeError("server did not become ready")
    try:
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=20)


def test_upload_through_ui_then_highlight_and_export(server, page, fixture_wav, tmp_path):
    page.set_default_timeout(20_000)

    page.goto(f"{server}/new")
    page.get_by_test_id("title-input").fill("T5 fixture")
    page.get_by_test_id("file-input").set_input_files(str(fixture_wav))
    page.get_by_test_id("upload-button").click()

    page.wait_for_url("**/sessions/*", timeout=30_000)
    page.wait_for_selector('[data-testid="session-status"]')

    # Progress is shown while the pipeline runs, then the session reaches "done".
    started = time.time()
    page.wait_for_function(
        "() => document.querySelector('[data-testid=\"session-status\"]')?.textContent === 'done'",
        timeout=PIPELINE_TIMEOUT_MS,
    )
    print(f"T5 pipeline finished in {time.time() - started:.1f}s")

    page.wait_for_function(
        "() => document.querySelectorAll('.question[data-testid^=\"question-\"]').length > 0",
        timeout=60_000,
    )
    questions = page.locator('.question[data-testid^="question-"]')
    count = questions.count()
    print(f"T5 questions rendered: {count}")
    assert count >= 1

    # type filter (F5): narrowing to the first question's type keeps at least that one
    first_type = questions.first.locator(".badge").nth(1).inner_text()
    page.get_by_test_id("type-filter").select_option(first_type)
    page.wait_for_function(
        "() => document.querySelectorAll('.question[data-testid^=\"question-\"]').length > 0"
    )
    filtered = page.locator('.question[data-testid^="question-"]').count()
    print(f"T5 filter={first_type} -> {filtered} question(s)")
    assert filtered >= 1
    page.get_by_test_id("type-filter").select_option("all")
    page.wait_for_function(
        "() => document.querySelectorAll('.question[data-testid^=\"question-\"]').length === "
        + str(count)
    )

    segments = page.locator('[data-testid^="segment-"]')
    assert segments.count() >= 1
    assert page.locator('[data-highlighted="true"]').count() == 0

    questions.first.click()
    page.wait_for_selector('[data-highlighted="true"]')
    highlighted = page.locator('[data-highlighted="true"]')
    assert highlighted.count() >= 1
    print(f"T5 highlighted segments after click: {highlighted.count()}")

    # the highlighted segment must contain the question's evidence quote
    quote = questions.first.locator(".muted").inner_text()
    evidence = quote.split("「", 1)[1].rsplit("」", 1)[0]
    highlighted_text = " ".join(
        highlighted.nth(i).inner_text() for i in range(highlighted.count())
    )
    from meetq.qgen.validate import normalize

    assert normalize(evidence) in normalize(highlighted_text)

    with page.expect_download() as download_info:
        page.get_by_test_id("export-md").click()
    download = download_info.value
    target = tmp_path / "export.md"
    download.save_as(target)
    body = target.read_text(encoding="utf-8")
    print(f"T5 downloaded {download.suggested_filename} ({len(body)} chars)")
    assert "## 質問" in body and "## 文字起こし" in body
    assert len(body) > 200


def test_settings_page_never_exposes_the_key(server, page):
    page.set_default_timeout(20_000)
    page.goto(f"{server}/")
    page.wait_for_selector('[data-testid="session-list"]')

    page.get_by_test_id("nav-settings").click()
    page.wait_for_selector('[data-testid="settings-page"]')
    assert page.get_by_test_id("stt-model").input_value()
    assert page.get_by_test_id("llm-model").input_value()
    status = page.get_by_test_id("api-key-status").inner_text()
    assert "設定済み" in status
    assert "sk-" not in page.content()  # the key itself is never sent to the UI
