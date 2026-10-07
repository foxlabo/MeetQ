"""T3 — loopback E2E: play the fixture on the default output device and capture it
back through WASAPI loopback, then transcribe the capture with the real API.

Running this makes the PC audibly play the fixture script.
"""

from __future__ import annotations

import json
import time
import winsound
from pathlib import Path

import pytest

from meetq.audio.chunker import VadChunker
from meetq.audio.loopback import AudioBackendUnavailable, LoopbackSource, list_devices
from meetq.audio.wavio import rms
from meetq.qgen.validate import normalize
from meetq.stt.base import TranscriptionContext
from meetq.stt.openai_file import OpenAIFileTranscriber

pytestmark = pytest.mark.t3

RECORD_SEC = 26.0
RMS_THRESHOLD = 150.0  # silence on this path sits near 0


@pytest.fixture(scope="module")
def capture(request):
    wav: Path = request.getfixturevalue("fixture_wav")
    try:
        devices = list_devices()
    except AudioBackendUnavailable as exc:
        pytest.fail(f"MEETQ_BLOCKED: audio backend unavailable ({exc})")

    print("T3 device enumeration:")
    print(json.dumps(devices, ensure_ascii=False, indent=1))
    if not devices["loopback"] or devices["default_loopback_index"] is None:
        pytest.fail(
            "MEETQ_BLOCKED: no WASAPI output/loopback device present; "
            f"enumeration={json.dumps(devices, ensure_ascii=False)}"
        )

    source = LoopbackSource()
    source.start()
    time.sleep(0.5)
    winsound.PlaySound(str(wav), winsound.SND_FILENAME | winsound.SND_ASYNC)
    time.sleep(RECORD_SEC)
    winsound.PlaySound(None, winsound.SND_PURGE)
    buffers = source.stop()
    assert buffers, "loopback capture produced no buffer"
    return {"buffers": buffers, "devices": devices}


def test_loopback_capture_has_signal(capture):
    buffer = capture["buffers"][0]
    level = rms(buffer.pcm)
    print(f"T3 captured {buffer.duration_sec:.1f}s on channel={buffer.channel} rms={level:.1f}")
    assert buffer.duration_sec >= RECORD_SEC - 2.0
    assert level > RMS_THRESHOLD, f"captured RMS {level:.1f} <= {RMS_THRESHOLD}"


def test_loopback_capture_transcribes_to_a_known_keyword(capture, manifest):
    buffer = capture["buffers"][0]
    chunks = VadChunker().chunk(buffer.pcm, channel=buffer.channel)
    result = OpenAIFileTranscriber().transcribe_chunks(
        chunks,
        TranscriptionContext(language=manifest["language"], keywords=manifest["keywords"]),
    )
    full = normalize(" ".join(s.text for s in result.segments))
    found = [kw for kw in manifest["keywords"] if normalize(kw) in full]
    print(f"T3 transcript segments={len(result.segments)} keywords found={found}")
    assert found, f"none of {manifest['keywords']} appeared in the loopback transcript"
