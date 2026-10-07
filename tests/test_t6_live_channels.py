"""T6 (Phase 2) — the two live-session paths that the UI test does not reach:

* mic + loopback captured as two concurrent Realtime sessions (F8)
* automatic reconnection after the WebSocket drops mid-stream (F8)

Both run against the real API; nothing is mocked.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from meetq.audio.base import SAMPLE_RATE
from meetq.audio.file_source import decode_to_pcm
from meetq.audio.loopback import AudioBackendUnavailable, list_devices
from meetq.db.models import Session as SessionRow
from meetq.db.session import db_session
from meetq.live import LiveController
from meetq.stt.openai_live import LiveFinal, OpenAILiveTranscriber

pytestmark = pytest.mark.t6

DUAL_CHANNEL_PLAY_SEC = 16.0
RECONNECT_FEED_SEC = 30.0


def _devices_or_blocked() -> dict:
    try:
        devices = list_devices()
    except AudioBackendUnavailable as exc:
        pytest.fail(f"MEETQ_BLOCKED: audio backend unavailable ({exc})")
    if not devices["loopback"] or devices["default_loopback_index"] is None:
        pytest.fail(
            "MEETQ_BLOCKED: no WASAPI output/loopback device present; "
            f"enumeration={json.dumps(devices, ensure_ascii=False)}"
        )
    return devices


def test_mic_and_loopback_run_as_two_live_sessions(fixture_wav, manifest):
    """F8: with a microphone selected, MeetQ opens one Realtime session per channel."""
    import winsound

    devices = _devices_or_blocked()
    mic_index = devices["default_input_index"]
    if mic_index is None or not devices["input"]:
        pytest.fail(
            "MEETQ_BLOCKED: no input (microphone) device present; "
            f"enumeration={json.dumps(devices, ensure_ascii=False)}"
        )

    with db_session() as db:
        row = SessionRow(
            title="T6 dual channel",
            source_type="record",
            status="recording",
            language=manifest["language"],
            mode="followup",
        )
        db.add(row)
        db.flush()
        session_id = row.id

    controller = LiveController(
        session_id=session_id,
        language=manifest["language"],
        keywords=manifest["keywords"],
        mode="followup",
    )
    channels = controller.start(loopback_index=None, mic_index=int(mic_index))
    try:
        assert channels == ["others", "self"], channels
        assert set(controller._transcribers) == {"others", "self"}
        time.sleep(0.5)
        winsound.PlaySound(str(fixture_wav), winsound.SND_FILENAME | winsound.SND_ASYNC)
        time.sleep(DUAL_CHANNEL_PLAY_SEC)
    finally:
        winsound.PlaySound(None, winsound.SND_PURGE)
        buffers = controller.stop()

    by_channel = {b.channel: b for b in buffers}
    print(
        "T6 dual-channel capture: "
        + ", ".join(f"{name}={buf.duration_sec:.1f}s" for name, buf in by_channel.items())
    )
    assert set(by_channel) == {"others", "self"}
    for name, buf in by_channel.items():
        assert buf.duration_sec > DUAL_CHANNEL_PLAY_SEC * 0.5, f"{name} captured too little"

    for channel, transcriber in controller._transcribers.items():
        print(
            f"T6 live session {channel}: finals={transcriber.finals} "
            f"reconnects={transcriber.reconnects} errors={transcriber.errors[:1]}"
        )
        assert not transcriber.errors, f"{channel} session reported {transcriber.errors}"

    # the speaker channel carried the fixture, so it must have produced confirmed text
    assert controller._transcribers["others"].finals >= 1

    with db_session() as db:
        row = db.get(SessionRow, session_id)
        assert row is not None
        channels_seen = {s.channel for s in row.segments}
        texts = " ".join(s.text for s in row.segments)
    print(f"T6 dual-channel segments: {len(channels_seen)} channel(s) {channels_seen}")
    assert "others" in channels_seen
    assert texts.strip()


def test_live_session_reconnects_after_the_socket_drops(fixture_wav, manifest):
    """F8: dropping the WebSocket mid-stream must not end the session."""
    pcm = decode_to_pcm(fixture_wav)
    finals: list[LiveFinal] = []
    lock = threading.Lock()

    def on_final(final: LiveFinal) -> None:
        with lock:
            finals.append(final)

    transcriber = OpenAILiveTranscriber(
        channel="others",
        language=manifest["language"],
        keywords=manifest["keywords"],
        on_final=on_final,
    )
    transcriber.start()

    stop_feeding = threading.Event()

    def pump() -> None:
        """Feed the fixture at real-time pace, looping if it runs out."""
        block = int(SAMPLE_RATE * 0.1)
        position = 0
        while not stop_feeding.is_set():
            chunk = pcm[position : position + block]
            if len(chunk) == 0:
                position = 0
                continue
            transcriber.feed(np.asarray(chunk))
            position += block
            time.sleep(0.1)

    feeder = threading.Thread(target=pump, daemon=True)
    feeder.start()

    try:
        started = time.time()
        while not finals and time.time() - started < 25:
            time.sleep(0.2)
        assert finals, "no confirmed text before the forced disconnect"
        before = len(finals)
        print(
            f"T6 reconnect: {before} final(s) after {time.time() - started:.1f}s, "
            "dropping socket"
        )

        assert transcriber.drop_connection(), "no live connection to drop"
        drop_at = time.time()

        while transcriber.reconnects < 1 and time.time() - drop_at < 20:
            time.sleep(0.2)
        print(
            f"T6 reconnect: reconnects={transcriber.reconnects} after "
            f"{time.time() - drop_at:.1f}s (errors={transcriber.errors[:1]})"
        )
        assert transcriber.reconnects >= 1, "the session did not notice the dropped socket"

        while len(finals) <= before and time.time() - drop_at < RECONNECT_FEED_SEC:
            time.sleep(0.2)
        after = len(finals)
        print(
            f"T6 reconnect: confirmed text resumed after {time.time() - drop_at:.1f}s "
            f"({before} -> {after} finals): {finals[-1].text[:40]!r}"
        )
        assert after > before, "no confirmed text arrived after reconnecting"
    finally:
        stop_feeding.set()
        feeder.join(timeout=5)
        transcriber.stop()

    assert all(f.start_sec < f.end_sec for f in finals)
    starts = [f.start_sec for f in finals]
    assert starts == sorted(starts), "timestamps must keep increasing across a reconnect"
