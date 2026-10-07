"""Shared test fixtures.

The audio fixture (tests/fixtures/meeting_ja.wav) is the ONLY audio ever sent to the
OpenAI API by the test suite, to keep spend bounded.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"
FIXTURE_WAV = FIXTURE_DIR / "meeting_ja.wav"
FIXTURE_MANIFEST = FIXTURE_DIR / "meeting_ja.json"


def _ensure_fixture() -> None:
    if FIXTURE_WAV.exists() and FIXTURE_MANIFEST.exists():
        return
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PROJECT_ROOT / "scripts" / "make_fixture.ps1"),
        ],
        check=True,
    )


@pytest.fixture(scope="session", autouse=True)
def isolated_data_dir():
    """Give every test run its own SQLite DB and media directory."""
    tmp = tempfile.mkdtemp(prefix="meetq-test-")
    os.environ["MEETQ_DATA_DIR"] = tmp
    from meetq.config import get_settings

    get_settings.cache_clear()
    get_settings()
    from meetq.db.session import dispose_db, init_db

    init_db()
    try:
        yield Path(tmp)
    finally:
        dispose_db()
        get_settings.cache_clear()
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="session")
def manifest() -> dict:
    _ensure_fixture()
    return json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8-sig"))


@pytest.fixture(scope="session")
def fixture_wav() -> Path:
    _ensure_fixture()
    return FIXTURE_WAV


def normalize_for_match(text: str) -> str:
    from meetq.qgen.validate import normalize

    return normalize(text)
