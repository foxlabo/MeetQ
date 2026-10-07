"""T1 — the text downloads re-flow the transcript into one sentence per line."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from meetq.api.app import create_app
from meetq.db.models import Segment, Session
from meetq.db.session import db_session
from meetq.transcript import reflow

pytestmark = pytest.mark.t1


def seg(start, end, text, channel="mixed"):
    return {"start_sec": start, "end_sec": end, "text": text, "channel": channel}


def test_a_sentence_cut_between_two_chunks_is_joined_back():
    out = reflow(
        [
            seg(0.0, 20.0, "3月末までにしっかりと"),
            seg(20.0, 30.0, "結合テストを終えることが大切です。"),
        ]
    )
    assert [s.text for s in out] == ["3月末までにしっかりと結合テストを終えることが大切です。"]
    assert out[0].start_sec == 0.0


def test_sentences_inside_one_chunk_get_their_own_lines_and_estimated_times():
    out = reflow([seg(10.0, 20.0, "はい。お願いします。")])
    assert [s.text for s in out] == ["はい。", "お願いします。"]
    assert out[0].start_sec == 10.0
    # 「お願いします」 starts at character 3 of 10 -> 3 s into the 10 s chunk
    assert out[1].start_sec == pytest.approx(13.0)


def test_question_marks_quotes_and_repeated_enders_stay_with_their_sentence():
    out = reflow([seg(0.0, 10.0, "彼は「来週です。」と言った。本当に？！次へ")])
    assert [s.text for s in out] == ["彼は「来週です。」", "と言った。", "本当に？！", "次へ"]


def test_a_sentence_never_spans_two_speakers():
    out = reflow(
        [
            seg(0.0, 5.0, "担当者は", channel="others"),
            seg(5.0, 8.0, "決まりましたか", channel="self"),
            seg(8.0, 12.0, "まだです。", channel="others"),
        ]
    )
    assert [(s.channel, s.text) for s in out] == [
        ("others", "担当者は"),
        ("self", "決まりましたか"),
        ("others", "まだです。"),
    ]


def test_ascii_words_meeting_at_a_chunk_boundary_keep_a_space():
    out = reflow([seg(0.0, 5.0, "ツールは Google"), seg(5.0, 9.0, "Meet を使います。")])
    assert out[0].text == "ツールは Google Meet を使います。"


def test_blank_segments_and_leading_spaces_are_ignored():
    out = reflow([seg(0.0, 5.0, "   "), seg(5.0, 9.0, "  はい。 そうです。")])
    assert [s.text for s in out] == ["はい。", "そうです。"]
    # times are measured on the stripped text 「はい。 そうです。」 (9 chars, 4 s)
    assert out[0].start_sec == 5.0
    assert out[1].start_sec == pytest.approx(5.0 + 4.0 * 4 / 9)


def test_segments_are_ordered_by_time_even_if_given_out_of_order():
    out = reflow([seg(20.0, 30.0, "二文目です。"), seg(0.0, 10.0, "一文目です。")])
    assert [s.text for s in out] == ["一文目です。", "二文目です。"]


def test_txt_and_markdown_downloads_are_one_sentence_per_line():
    with db_session() as db:
        sess = Session(title="reflow", source_type="upload", status="done")
        db.add(sess)
        db.flush()
        for i, (start, end, text) in enumerate(
            [
                (0.0, 20.0, "では始めます。予算は80%を"),
                (20.0, 32.0, "消化しています。追加申請が"),
                (32.0, 40.0, "必要です。"),
            ]
        ):
            db.add(
                Segment(
                    session_id=sess.id,
                    idx=i,
                    start_sec=start,
                    end_sec=end,
                    text=text,
                    raw_text=text,
                    channel="others",
                )
            )
        session_id = sess.id

    client = TestClient(create_app())
    txt = client.get(f"/api/sessions/{session_id}/export?format=txt").text.strip().splitlines()
    # 「予算は」 starts at char 7 of 14 in a 20 s chunk -> 10 s;
    # 「追加申請が」 at char 8 of 13 in the 20-32 s chunk -> 27.4 s
    assert txt == [
        "[00:00] では始めます。",
        "[00:10] 予算は80%を消化しています。",
        "[00:27] 追加申請が必要です。",
    ]

    md = client.get(f"/api/sessions/{session_id}/export?format=md").text
    transcript_lines = [line for line in md.splitlines() if line.startswith("- `")]
    assert transcript_lines == [
        "- `00:00` [others] では始めます。",
        "- `00:10` [others] 予算は80%を消化しています。",
        "- `00:27` [others] 追加申請が必要です。",
    ]

    # JSON keeps the segments untouched: questions point at segment ids
    data = client.get(f"/api/sessions/{session_id}/export?format=json").json()
    assert [s["text"] for s in data["segments"]] == [
        "では始めます。予算は80%を",
        "消化しています。追加申請が",
        "必要です。",
    ]
