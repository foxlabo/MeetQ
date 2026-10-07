"""T6 — Phase 2 E2E: the fixture is played at real-time pace into the live
transcription pipeline (WASAPI loopback -> Realtime API -> UI), and rolling
question generation runs twice.

Nothing is mocked. Running this makes the PC audibly play the fixture script.
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
import winsound
from pathlib import Path

import pytest

from meetq.audio.loopback import AudioBackendUnavailable, list_devices
from meetq.qgen.rolling import DUPLICATE_THRESHOLD, similarity
from meetq.qgen.validate import TranscriptIndex, normalize

pytestmark = pytest.mark.t6

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROLLING_INTERVAL_SEC = 20
FIRST_TEXT_BUDGET_SEC = 10.0
ROUNDS_WANTED = 2
READY_TIMEOUT = 60


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _api(base: str, path: str) -> dict | list:
    with urllib.request.urlopen(f"{base}{path}", timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


@pytest.fixture(scope="module")
def live_server():
    static = PROJECT_ROOT / "meetq" / "static" / "index.html"
    assert static.is_file(), "frontend build missing: run `npm run build` in frontend/"

    port = _free_port()
    data_dir = tempfile.mkdtemp(prefix="meetq-live-ui-")
    env = {
        **os.environ,
        "MEETQ_DATA_DIR": data_dir,
        "PYTHONIOENCODING": "utf-8",
        "MEETQ_ROLLING_INTERVAL_SEC": str(ROLLING_INTERVAL_SEC),
    }
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
        proc.wait(timeout=25)


def test_live_transcription_and_rolling_generation(live_server, page, fixture_wav, manifest):
    try:
        devices = list_devices()
    except AudioBackendUnavailable as exc:
        pytest.fail(f"MEETQ_BLOCKED: audio backend unavailable ({exc})")
    if not devices["loopback"] or devices["default_loopback_index"] is None:
        pytest.fail(
            "MEETQ_BLOCKED: no WASAPI output/loopback device present; "
            f"enumeration={json.dumps(devices, ensure_ascii=False)}"
        )

    page.set_default_timeout(30_000)
    page.goto(f"{live_server}/new")
    page.get_by_test_id("title-input").fill("T6 live fixture")
    page.wait_for_selector('[data-testid="live-toggle"]')
    assert page.get_by_test_id("live-toggle").is_checked(), "live transcription must default on"
    page.get_by_test_id("start-recording").click()
    page.wait_for_url("**/sessions/*", timeout=30_000)
    session_id = int(page.url.rstrip("/").rsplit("/", 1)[1])

    # ---- F8: first confirmed text must reach the UI within 10 s of audio starting
    t0 = time.time()
    winsound.PlaySound(str(fixture_wav), winsound.SND_FILENAME | winsound.SND_ASYNC)
    try:
        page.wait_for_function(
            "() => document.querySelectorAll('[data-testid^=\"segment-\"]').length > 0",
            timeout=int(FIRST_TEXT_BUDGET_SEC * 3 * 1000),
        )
        first_text_sec = time.time() - t0
        first_segment = page.locator('[data-testid^="segment-"]').first.inner_text()
        print(f"T6 first confirmed text in the UI after {first_text_sec:.2f}s: {first_segment!r}")
        assert first_text_sec <= FIRST_TEXT_BUDGET_SEC, (
            f"first confirmed text took {first_text_sec:.2f}s (budget {FIRST_TEXT_BUDGET_SEC}s)"
        )

        # ---- F9: wait for two rolling rounds fired by the timer
        rounds_deadline = time.time() + ROLLING_INTERVAL_SEC * ROUNDS_WANTED + 120
        live = {"rounds": 0}
        while time.time() < rounds_deadline:
            live = _api(live_server, f"/api/sessions/{session_id}/live")  # type: ignore[assignment]
            if int(live.get("rounds", 0)) >= ROUNDS_WANTED:
                break
            time.sleep(1.0)
        print(
            f"T6 live status after {time.time() - t0:.1f}s: rounds={live.get('rounds')} "
            f"finals={live.get('finals')} reconnects={live.get('reconnects')}"
        )
        assert int(live.get("rounds", 0)) >= ROUNDS_WANTED, (
            f"only {live.get('rounds')} rolling rounds ran"
        )

        # ---- F9/F10 are checked while still live: after 録音停止 the questions are
        # regenerated from the full-text pass, so this is the moment
        # the rolling-round questions exist.
        live_segments = _api(live_server, f"/api/sessions/{session_id}/segments")
        rolling = _api(live_server, f"/api/sessions/{session_id}/questions")
        live_session = _api(live_server, f"/api/sessions/{session_id}")
        assert isinstance(live_segments, list) and isinstance(rolling, list)

        joined = normalize("".join(s["text"] for s in live_segments))
        found = [kw for kw in manifest["keywords"] if normalize(kw) in joined]
        print(f"T6 live transcript: {len(live_segments)} live segments, keywords found={found}")
        assert found, f"none of {manifest['keywords']} appeared in the live transcript"

        by_round: dict[int, list[dict]] = {}
        for q in rolling:
            by_round.setdefault(int(q["round"]), []).append(q)
        for round_no in sorted(by_round):
            print(f"T6 rolling round {round_no}: {len(by_round[round_no])} question(s)")
            for q in by_round[round_no]:
                print(f"    [P{q['priority']}][{q['type']}] {q['text']}")
                print(f"      根拠 {q['evidence_start_sec']}s: 「{q['evidence_quote']}」")
        print(f"T6 cumulative summary: {live_session['summary'][:160]}")

        assert len(by_round) >= ROUNDS_WANTED, f"questions came from rounds {sorted(by_round)}"
        assert len(rolling) >= 2, f"only {len(rolling)} questions in total"

        duplicates = []
        for i, a in enumerate(rolling):
            for b in rolling[i + 1 :]:
                score = similarity(normalize(a["text"]), normalize(b["text"]))
                if score >= DUPLICATE_THRESHOLD:
                    duplicates.append((a["text"], b["text"], score))
        print(f"T6 duplicate pairs among {len(rolling)} questions: {len(duplicates)}")
        assert not duplicates, f"duplicate questions: {duplicates}"

        live_index = TranscriptIndex(live_segments)
        for q in rolling:
            assert live_index.locate(q["evidence_quote"]) is not None, (
                f"evidence not in the live transcript: {q['evidence_quote']}"
            )
    finally:
        winsound.PlaySound(None, winsound.SND_PURGE)
        page.get_by_test_id("stop-recording").click()

    # ---- after 録音停止: the saved recording is re-transcribed as full text and the
    # questions are regenerated from it
    stopped_at = time.time()
    page.wait_for_function(
        "() => ['done','error'].includes("
        "document.querySelector('[data-testid=\"session-status\"]')?.textContent)",
        timeout=240_000,
    )
    assert page.get_by_test_id("session-status").inner_text() == "done"
    print(f"T6 full-text pass finished {time.time() - stopped_at:.1f}s after stopping")

    final_segments = _api(live_server, f"/api/sessions/{session_id}/segments")
    final_questions = _api(live_server, f"/api/sessions/{session_id}/questions")
    assert isinstance(final_segments, list) and isinstance(final_questions, list)

    # SQLite hands out row ids again after the old rows are deleted, so compare content
    live_lines = [(s["start_sec"], s["text"]) for s in live_segments]
    final_lines = [(s["start_sec"], s["text"]) for s in final_segments]
    assert final_lines != live_lines, "the live transcript was not replaced"
    # file chunks are cut at real silences (10-30 s), so there are fewer, longer lines
    assert len(final_segments) < len(live_segments), (
        f"{len(final_segments)} full-text segments vs {len(live_segments)} live ones"
    )
    starts = [s["start_sec"] for s in final_segments]
    assert starts == sorted(starts), "full-text timestamps must increase"

    final_joined = normalize("".join(s["text"] for s in final_segments))
    final_found = [kw for kw in manifest["keywords"] if normalize(kw) in final_joined]
    print(
        f"T6 full text: {len(final_segments)} segments (live had {len(live_segments)}), "
        f"keywords found={final_found}"
    )
    assert final_found, "the full-text transcript lost the fixture keywords"

    print(f"T6 final questions from the full text: {len(final_questions)}")
    for q in final_questions:
        print(f"    [P{q['priority']}][{q['type']}] {q['text']}")
    assert len(final_questions) >= 2
    assert all(int(q["round"]) == 0 for q in final_questions)
    final_index = TranscriptIndex(final_segments)
    for q in final_questions:
        assert final_index.locate(q["evidence_quote"]) is not None, (
            f"evidence not in the full-text transcript: {q['evidence_quote']}"
        )

    usage = _api(live_server, f"/api/sessions/{session_id}/usage")
    print(f"T6 usage={usage}")
    assert usage["transcribed_sec"] > 0
    assert usage["llm_requests"] >= ROUNDS_WANTED
