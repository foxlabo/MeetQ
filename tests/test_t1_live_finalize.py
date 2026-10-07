"""T1 — after a live recording stops, the saved audio is re-transcribed at real
silences and the questions are regenerated from that full text.

OpenAI is mocked here; T6 runs the same path against the real API.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from meetq import service
from meetq.api.app import create_app
from meetq.audio.base import AudioBuffer, Chunk
from meetq.db.models import Question, Segment, Session, Usage
from meetq.db.session import db_session
from meetq.qgen.mock import MockProvider
from meetq.qgen.validate import TranscriptIndex, normalize
from meetq.stt.base import Transcriber, TranscriptionContext, TranscriptionResult, TranscriptSegment

pytestmark = pytest.mark.t1

# What the file model "hears" per chunk: clean, complete sentences.
FULL_TEXT = [
    "先日は打ち合わせありがとうございました。",
    "イベントを行うっていう話だったと思うんですけれども、",
    "今の状況的に難しくなってしまったので、それに代わるイベントを考えています。",
    "予算と担当者は来週までに決めたいと思います。",
]


class FileModelStandIn(Transcriber):
    """Returns FULL_TEXT, one sentence per chunk, with the chunk's own timestamps."""

    def __init__(self) -> None:
        self.calls = 0
        self.keywords: list[str] = []

    def transcribe_chunks(
        self, chunks: list[Chunk], context: TranscriptionContext, on_progress=None
    ) -> TranscriptionResult:
        self.calls += 1
        self.keywords = list(context.keywords)
        result = TranscriptionResult()
        for i, chunk in enumerate(chunks):
            result.requests += 1
            result.transcribed_sec += chunk.duration_sec
            result.segments.append(
                TranscriptSegment(
                    index=i,
                    start_sec=chunk.start_sec,
                    end_sec=chunk.end_sec,
                    text=FULL_TEXT[i % len(FULL_TEXT)],
                    channel=chunk.channel,
                )
            )
        return result


def _live_session(fixture_wav) -> int:
    """A session exactly as a live recording leaves it right after 録音停止."""
    with db_session() as db:
        sess = Session(
            title="live finalize",
            source_type="record",
            status="queued",
            live=True,
            language="ja",
            glossary="予算\n担当者",
            mode="followup",
            audio_path=str(fixture_wav),
            channels=json.dumps(["others"]),
            summary="会議中に作られた累積サマリ",
        )
        db.add(sess)
        db.flush()
        # live text clipped at the forced 5 s commits
        for i, text in enumerate(["先日は", "じゃあありがとうございました", "今の象徴的に難しく"]):
            db.add(
                Segment(
                    session_id=sess.id,
                    idx=i,
                    start_sec=i * 5.0,
                    end_sec=i * 5.0 + 5.0,
                    text=text,
                    channel="others",
                )
            )
        for round_no in (1, 2):
            db.add(
                Question(
                    session_id=sess.id,
                    text=f"会議中の質問 round {round_no}",
                    type="clarify",
                    priority=1,
                    evidence_quote="象徴的に難しく",
                    round=round_no,
                )
            )
        usage = Usage.blank(sess.id)
        usage.transcribed_sec = 15.0
        usage.stt_requests = 3
        usage.llm_input_tokens = 100
        usage.llm_output_tokens = 50
        usage.llm_requests = 2
        db.add(usage)
        return sess.id


def test_live_session_is_rebuilt_from_the_full_text(fixture_wav):
    session_id = _live_session(fixture_wav)
    transcriber = FileModelStandIn()
    provider = MockProvider(
        [
            {
                "questions": [
                    {
                        "text": "代わりのイベントは何を検討していますか?",
                        "type": "unresolved",
                        "priority": 1,
                        "evidence_quote": "それに代わるイベントを考えています",
                    },
                    {
                        "text": "予算と担当者は誰が決めますか?",
                        "type": "clarify",
                        "priority": 2,
                        "evidence_quote": "予算と担当者は来週までに決めたい",
                    },
                ]
            }
        ]
    )

    service.process_session(session_id, transcriber=transcriber, provider=provider)

    with db_session() as db:
        sess = db.get(Session, session_id)
        assert sess is not None
        assert sess.status == "done", sess.error
        segments = sorted(sess.segments, key=lambda s: s.start_sec)
        questions = list(sess.questions)
        usage = sess.usage

        # the clipped live transcript is gone, replaced by the full text
        texts = [s.text for s in segments]
        assert "じゃあありがとうございました" not in texts
        assert "今の象徴的に難しく" not in texts
        assert texts[0] == FULL_TEXT[0]
        assert all(s.channel == "others" for s in segments)
        starts = [s.start_sec for s in segments]
        assert starts == sorted(starts) and starts[0] == 0.0

        # questions were regenerated from that text; the in-meeting ones are gone
        assert {q.text for q in questions} == {
            "代わりのイベントは何を検討していますか?",
            "予算と担当者は誰が決めますか?",
        }
        assert all(q.round == 0 for q in questions)

        # F4: every evidence quote is a substring of the transcript now shown
        index = TranscriptIndex(
            [{"id": s.id, "start_sec": s.start_sec, "text": s.text} for s in segments]
        )
        for q in questions:
            assert normalize(q.evidence_quote) in index.normalized
            ids = json.loads(q.evidence_segment_ids)
            assert ids and set(ids) <= {s.id for s in segments}

        # F7: usage keeps the live spend and adds the file pass on top
        assert usage.transcribed_sec > 15.0
        assert usage.stt_requests == 3 + transcriber.calls * len(segments)
        assert usage.llm_requests == 2 + provider.calls

        # the glossary still reaches the file model as keywords
        assert transcriber.keywords == ["予算", "担当者"]
        # the cumulative summary from the meeting is kept
        assert sess.summary == "会議中に作られた累積サマリ"


def test_stopping_a_live_recording_queues_the_full_text_pass(monkeypatch, fixture_wav):
    """The stop endpoint must send live sessions through the file pipeline too."""
    from meetq.api import routes

    with db_session() as db:
        sess = Session(
            title="live stop",
            source_type="record",
            status="recording",
            live=True,
            channels=json.dumps(["others"]),
        )
        db.add(sess)
        db.flush()
        session_id = sess.id

    pcm = (np.sin(np.linspace(0, 400, 16000 * 2)) * 8000).astype(np.int16)
    monkeypatch.setattr(
        routes.recorder, "stop", lambda sid: [AudioBuffer(pcm=pcm, channel="others")]
    )
    started: list[int] = []
    monkeypatch.setattr(routes.service, "start_processing", lambda sid: started.append(sid))

    response = TestClient(create_app()).post(f"/api/recording/{session_id}/stop")
    assert response.status_code == 200, response.text
    body = response.json()

    assert started == [session_id], "live sessions must be re-transcribed after stopping"
    assert body["status"] == "queued"
    assert body["stage"] == "全文を文字起こし中"
    assert body["duration_sec"] == pytest.approx(2.0, abs=0.05)


def test_generate_now_returns_before_the_round_finishes(monkeypatch):
    """F9's button must not hold the HTTP request open (proxies time out)."""
    import threading

    from meetq.live import live_registry

    started = threading.Event()
    release = threading.Event()

    class SlowController:
        rounds = 0

        def generate_round(self, trigger: str = "manual") -> int:
            started.set()
            release.wait(timeout=10)
            SlowController.rounds += 1
            return 1

    with db_session() as db:
        sess = Session(title="manual round", source_type="record", status="recording", live=True)
        db.add(sess)
        db.flush()
        session_id = sess.id

    controller = SlowController()
    live_registry.add(session_id, controller)
    try:
        response = TestClient(create_app()).post(f"/api/sessions/{session_id}/generate-now")
        assert response.status_code == 200, response.text
        assert response.json() == {"started": True, "rounds": 0}
        assert started.wait(timeout=5), "the round never started in the background"
        assert controller.rounds == 0, "the request waited for the round to finish"
    finally:
        release.set()
        live_registry.pop(session_id)


def test_generate_now_needs_a_live_session():
    with db_session() as db:
        sess = Session(title="not live", source_type="upload", status="done")
        db.add(sess)
        db.flush()
        session_id = sess.id
    response = TestClient(create_app()).post(f"/api/sessions/{session_id}/generate-now")
    assert response.status_code == 400
