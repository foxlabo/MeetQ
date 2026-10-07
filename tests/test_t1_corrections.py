"""T1 — term corrections and the transcript text download.

No OpenAI calls here: corrections are local string rules, and the pipeline test uses a
stand-in transcriber.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from meetq import appsettings, service
from meetq.api.app import create_app
from meetq.audio.base import Chunk
from meetq.corrections import apply_rules, parse_rules, preview, targets
from meetq.db.models import Question, Segment, Session
from meetq.db.session import db_session
from meetq.live import LiveController
from meetq.qgen.mock import MockProvider
from meetq.qgen.validate import TranscriptIndex, normalize
from meetq.stt.base import (
    Transcriber,
    TranscriptionContext,
    TranscriptionResult,
    TranscriptSegment,
)
from meetq.stt.openai_live import LiveFinal

pytestmark = pytest.mark.t1

RULES = "使用 => 仕様\n以降 => 移行"


# ---------------------------------------------------------------- parsing
def test_parse_rules_accepts_every_separator_and_skips_comments():
    rules, errors = parse_rules("# 開発用語\n使用 => 仕様\n\n以降 → 移行\n昨日 -> 機能\n")
    assert errors == []
    assert [(r.wrong, r.right) for r in rules] == [
        ("使用", "仕様"),
        ("以降", "移行"),
        ("昨日", "機能"),
    ]


@pytest.mark.parametrize(
    "line",
    ["使用 仕様", "=> 仕様", "使用 =>"],
)
def test_parse_rules_reports_malformed_lines(line):
    rules, errors = parse_rules(line)
    assert rules == []
    assert len(errors) == 1 and "1 行目" in errors[0]


def test_parse_rules_rejects_a_duplicate_wrong_term():
    rules, errors = parse_rules("使用 => 仕様\n使用 => 私用")
    assert len(rules) == 1
    assert "重複" in errors[0]


# --------------------------------------------------------------- applying
def test_apply_rules_prefers_the_longest_match():
    rules, _ = parse_rules("使用 => 仕様\n使用策定 => 仕様策定")
    assert apply_rules("使用策定", rules) == ("仕様策定", 1)


def test_an_identity_rule_protects_a_term_from_a_shorter_rule():
    rules, errors = parse_rules("使用 => 仕様\n使用料 => 使用料")
    assert errors == []
    text, count = apply_rules("使用策定と使用料", rules)
    assert text == "仕様策定と使用料"
    assert count == 1  # the protected term is not counted as a change


def test_apply_rules_is_single_pass_without_cascades():
    rules, _ = parse_rules("A => B\nB => C")
    assert apply_rules("AB", rules) == ("BC", 2)


def test_targets_are_the_corrected_terms_once_each():
    assert targets("使用 => 仕様\n使用書 => 仕様書\n以降 => 移行") == [
        "仕様",
        "仕様書",
        "移行",
    ]


def test_preview_counts_on_raw_text_with_context():
    rules, _ = parse_rules(RULES)
    segs = [
        {"id": 1, "start_sec": 14.2, "text": "x", "raw_text": "以降の使用変更、使用策定"},
        {"id": 2, "start_sec": 30.0, "text": "仕様書", "raw_text": ""},
    ]
    items = {p.wrong: p for p in preview(segs, rules)}
    assert items["使用"].count == 2
    assert items["以降"].count == 1
    example = items["使用"].examples[0]
    assert example["match"] == "使用" and example["start_sec"] == 14.2
    assert example["before"].endswith("以降の")


# ------------------------------------------------- applying to a session
def _session_with_mishearing(corrections: str = "") -> int:
    with db_session() as db:
        sess = Session(
            title="定例ミーティング",
            source_type="upload",
            status="done",
            summary="以降の使用変更について",
            corrections=corrections,
        )
        db.add(sess)
        db.flush()
        texts = [
            "では以降の使用変更、使用策定についてお話しします。",
            "仕様書のレビューは大切です。",
            "使用書を見たときにはここを確認します。",
        ]
        segs = []
        for i, text in enumerate(texts):
            seg = Segment(
                session_id=sess.id,
                idx=i,
                start_sec=i * 20.0,
                end_sec=i * 20.0 + 20.0,
                text=text,
                raw_text=text,
            )
            db.add(seg)
            segs.append(seg)
        db.flush()
        db.add(
            Question(
                session_id=sess.id,
                text="使用策定とは具体的に何ですか?",
                type="clarify",
                priority=1,
                evidence_quote="使用変更、使用策定",
                evidence_start_sec=0.0,
                evidence_segment_ids=json.dumps([segs[0].id]),
            )
        )
        db.add(
            # the quote ends in the middle of a term a rule will replace: 「使」|「用書」
            Question(
                session_id=sess.id,
                text="確認する項目は何ですか?",
                type="clarify",
                priority=2,
                evidence_quote="を見たときにはここを確認します",
                evidence_start_sec=40.0,
                evidence_segment_ids=json.dumps([segs[2].id]),
            )
        )
        db.add(
            Question(
                session_id=sess.id,
                text="はみ出す引用",
                type="risk",
                priority=2,
                evidence_quote="使用書を見た",
                evidence_start_sec=40.0,
                evidence_segment_ids=json.dumps([segs[2].id]),
            )
        )
        return sess.id


def _state(session_id: int) -> tuple[list[Segment], list[Question], Session]:
    with db_session() as db:
        sess = db.get(Session, session_id)
        assert sess is not None
        segs = sorted(sess.segments, key=lambda s: s.start_sec)
        qs = sorted(sess.questions, key=lambda q: q.id)
        return segs, qs, sess


def test_applying_rules_fixes_transcript_questions_and_summary():
    session_id = _session_with_mishearing()
    stats = service.apply_session_corrections(session_id, RULES)
    segs, qs, sess = _state(session_id)

    assert stats["replacements"] == 4
    assert stats["segments_changed"] == 2
    assert segs[0].text == "では移行の仕様変更、仕様策定についてお話しします。"
    assert segs[2].text == "仕様書を見たときにはここを確認します。"
    assert segs[0].raw_text.startswith("では以降の使用変更")  # the original is kept
    assert qs[0].text == "仕様策定とは具体的に何ですか?"
    assert sess.summary == "移行の仕様変更について"
    assert sess.corrections == RULES

    # every evidence quote is still a substring of the transcript after the fix (F4)
    index = TranscriptIndex([{"id": s.id, "start_sec": s.start_sec, "text": s.text} for s in segs])
    for q in qs:
        assert normalize(q.evidence_quote) in index.normalized, q.evidence_quote
        assert json.loads(q.evidence_segment_ids)
    assert qs[0].evidence_quote == "仕様変更、仕様策定"
    assert qs[2].evidence_quote == "仕様書を見た"


def test_clearing_the_rules_restores_the_original_transcript():
    session_id = _session_with_mishearing()
    service.apply_session_corrections(session_id, RULES)
    service.apply_session_corrections(session_id, "")
    segs, qs, sess = _state(session_id)
    assert segs[0].text == "では以降の使用変更、使用策定についてお話しします。"
    assert sess.corrections == ""
    index = TranscriptIndex([{"id": s.id, "start_sec": s.start_sec, "text": s.text} for s in segs])
    for q in qs:
        assert normalize(q.evidence_quote) in index.normalized


def test_segments_from_before_corrections_existed_use_their_text_as_raw():
    session_id = _session_with_mishearing()
    with db_session() as db:
        for seg in db.get(Session, session_id).segments:
            seg.raw_text = ""
    service.apply_session_corrections(session_id, RULES)
    segs, _, _ = _state(session_id)
    assert segs[0].raw_text.startswith("では以降の")
    assert segs[0].text.startswith("では移行の")


def test_malformed_rules_are_rejected_and_nothing_changes():
    session_id = _session_with_mishearing()
    with pytest.raises(ValueError):
        service.apply_session_corrections(session_id, "使用 仕様")
    segs, _, _ = _state(session_id)
    assert segs[0].text.startswith("では以降の使用")


# ---------------------------------------------------- automatic application
class _StandInTranscriber(Transcriber):
    def __init__(self) -> None:
        self.keywords: list[str] = []

    def transcribe_chunks(
        self, chunks: list[Chunk], context: TranscriptionContext, on_progress=None
    ) -> TranscriptionResult:
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
                    text="以降の使用策定について話します。",
                    channel=chunk.channel,
                )
            )
        return result


def test_new_transcripts_get_the_session_rules_and_hint_the_stt_model(fixture_wav):
    with db_session() as db:
        sess = Session(
            title="auto",
            source_type="upload",
            status="queued",
            audio_path=str(fixture_wav),
            channels=json.dumps(["mixed"]),
            glossary="機能",
            corrections=RULES,
        )
        db.add(sess)
        db.flush()
        session_id = sess.id
    transcriber = _StandInTranscriber()
    provider = MockProvider(
        [
            {
                "questions": [
                    {
                        "text": "仕様策定の期限は?",
                        "type": "clarify",
                        "priority": 1,
                        "evidence_quote": "移行の仕様策定",
                    }
                ]
            }
        ]
    )
    service.process_session(session_id, transcriber=transcriber, provider=provider)
    segs, qs, sess = _state(session_id)

    assert sess.status == "done", sess.error
    assert all(s.text == "移行の仕様策定について話します。" for s in segs)
    assert all(s.raw_text == "以降の使用策定について話します。" for s in segs)
    assert transcriber.keywords == ["機能", "仕様", "移行"]
    # questions are generated from the corrected text, so their evidence uses it
    assert [q.evidence_quote for q in qs] == ["移行の仕様策定"]


def test_live_turns_are_corrected_as_they_arrive():
    with db_session() as db:
        sess = Session(title="live", source_type="record", status="recording", corrections=RULES)
        db.add(sess)
        db.flush()
        session_id = sess.id
    controller = LiveController(session_id=session_id, language="ja", keywords=[])
    controller._on_final(
        LiveFinal(start_sec=0.0, end_sec=5.0, text="使用策定です", channel="others")
    )
    segs, _, _ = _state(session_id)
    assert segs[0].text == "仕様策定です"
    assert segs[0].raw_text == "使用策定です"


# ------------------------------------------------------------------- API
def test_corrections_api_previews_applies_and_can_save_as_default():
    session_id = _session_with_mishearing()
    client = TestClient(create_app())

    preview_body = client.post(
        f"/api/sessions/{session_id}/corrections/preview", json={"rules": RULES}
    ).json()
    assert preview_body["errors"] == []
    assert preview_body["total"] == 4
    assert {r["wrong"]: r["count"] for r in preview_body["rules"]} == {"使用": 3, "以降": 1}

    applied = client.post(
        f"/api/sessions/{session_id}/corrections", json={"rules": RULES, "save_global": True}
    )
    assert applied.status_code == 200, applied.text
    assert applied.json()["replacements"] == 4
    assert appsettings.all_settings()["corrections"] == RULES
    appsettings.save({"corrections": ""})

    bad = client.post(f"/api/sessions/{session_id}/corrections", json={"rules": "使用"})
    assert bad.status_code == 400
    assert client.put("/api/settings", json={"corrections": "使用"}).status_code == 400


def test_corrections_are_refused_while_recording():
    with db_session() as db:
        sess = Session(title="busy", source_type="record", status="recording")
        db.add(sess)
        db.flush()
        session_id = sess.id
    response = TestClient(create_app()).post(
        f"/api/sessions/{session_id}/corrections", json={"rules": RULES}
    )
    assert response.status_code == 409


def test_transcript_download_is_plain_text_with_timestamps():
    session_id = _session_with_mishearing()
    with db_session() as db:
        db.add(
            Segment(
                session_id=session_id,
                idx=3,
                start_sec=5004.4,
                end_sec=5010.0,
                text="はい。お願いします。",
                raw_text="はい。お願いします。",
            )
        )
    service.apply_session_corrections(session_id, RULES)
    response = TestClient(create_app()).get(f"/api/sessions/{session_id}/export?format=txt")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["content-disposition"].endswith('_transcript.txt"')
    lines = response.text.strip().splitlines()
    assert lines[0] == "[00:00] では移行の仕様変更、仕様策定についてお話しします。"
    assert lines[1].startswith("[00:20] ")
    # one sentence per line, h:mm:ss past one hour
    assert lines[-2] == "[1:23:24] はい。"
    assert lines[-1] == "[1:23:26] お願いします。"
    assert "使用" not in response.text
