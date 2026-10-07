"""T1 (Phase 2 additions) — unit tests for the live resampler and the rolling
generation filters. OpenAI is mocked here; T6 uses the real API."""

from __future__ import annotations

import numpy as np
import pytest

from meetq.qgen.base import GeneratedQuestion, TokenUsage
from meetq.qgen.mock import MockProvider
from meetq.qgen.rolling import (
    drop_already_answered,
    drop_duplicates,
    generate_rolling,
    update_summary,
    window_segments,
)
from meetq.stt.openai_live import upsample

pytestmark = pytest.mark.t1

SEGMENTS = [
    {"id": 1, "start_sec": 0.0, "end_sec": 10.0, "text": "リリース日は来週の金曜です。"},
    {"id": 2, "start_sec": 10.0, "end_sec": 20.0, "text": "予算は80%消化しました。"},
    {"id": 3, "start_sec": 20.0, "end_sec": 30.0, "text": "移行の担当者は未定です。"},
    {"id": 4, "start_sec": 30.0, "end_sec": 40.0, "text": "担当者は田中さんに決まりました。"},
]


# ------------------------------------------------------------------ resampler
def test_upsample_16k_to_24k_keeps_duration_and_shape():
    sr_in, sr_out = 16000, 24000
    t = np.arange(sr_in) / sr_in
    pcm = (8000 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
    out, carry = upsample(pcm, sr_in, sr_out)
    assert out.dtype == np.int16
    assert len(out) == pytest.approx(sr_out, abs=2)
    assert carry == pytest.approx(float(pcm[-1]))
    assert np.max(np.abs(out)) > 4000  # the tone survived


def test_upsample_is_continuous_across_blocks():
    sr_in, sr_out = 16000, 24000
    t = np.arange(sr_in) / sr_in
    pcm = (8000 * np.sin(2 * np.pi * 50 * t)).astype(np.int16)
    whole, _ = upsample(pcm, sr_in, sr_out)
    carry = None
    parts = []
    for i in range(0, len(pcm), 1600):
        block, carry = upsample(pcm[i : i + 1600], sr_in, sr_out, carry)
        parts.append(block)
    streamed = np.concatenate(parts)
    assert len(streamed) == pytest.approx(len(whole), abs=4)
    n = min(len(streamed), len(whole))
    assert np.max(np.abs(streamed[:n].astype(np.int32) - whole[:n].astype(np.int32))) < 400


def test_upsample_noop_when_rates_match():
    pcm = np.array([1, 2, 3], dtype=np.int16)
    out, carry = upsample(pcm, 16000, 16000)
    assert list(out) == [1, 2, 3]
    assert carry == 3.0


# --------------------------------------------------------------------- window
def test_window_segments_keeps_only_recent_audio():
    assert [s["id"] for s in window_segments(SEGMENTS, 15.0)] == [3, 4]
    assert len(window_segments(SEGMENTS, 1000.0)) == 4
    assert window_segments([], 60.0) == []


# ----------------------------------------------------------------- duplicates
def test_drop_duplicates_removes_repeats_of_stored_questions():
    candidates = [
        GeneratedQuestion("担当者はいつ決まりますか?", "unresolved", 1, "担当者は未定"),
        GeneratedQuestion("予算の追加申請は必要ですか?", "risk", 2, "予算は80%消化"),
    ]
    kept, dropped = drop_duplicates(candidates, ["担当者はいつ決まりますか？"])
    assert dropped == 1
    assert [q.text for q in kept] == ["予算の追加申請は必要ですか?"]


def test_drop_duplicates_also_dedupes_within_one_batch():
    candidates = [
        GeneratedQuestion("予算は足りますか?", "risk", 1, "予算は80%消化"),
        GeneratedQuestion("予算は足りますか？", "risk", 2, "予算は80%消化"),
    ]
    kept, dropped = drop_duplicates(candidates, [])
    assert len(kept) == 1
    assert dropped == 1


# ------------------------------------------------------------ answered filter
def test_drop_already_answered_uses_the_provider_verdicts():
    candidates = [
        GeneratedQuestion("担当者は誰ですか?", "unresolved", 1, "担当者は未定"),
        GeneratedQuestion("リリース日は確定ですか?", "clarify", 1, "リリース日は来週の金曜"),
    ]
    provider = MockProvider([{"verdicts": [{"index": 0, "answered": True}]}])
    kept, dropped, _ = drop_already_answered(provider, candidates, SEGMENTS)
    assert dropped == 1
    assert [q.text for q in kept] == ["リリース日は確定ですか?"]


def test_drop_already_answered_keeps_everything_when_the_provider_fails():
    class Boom(MockProvider):
        def generate_json(self, system, user, schema, schema_name):
            raise RuntimeError("down")

    candidates = [GeneratedQuestion("q", "clarify", 1, "予算は80%消化")]
    kept, dropped, usage = drop_already_answered(Boom([]), candidates, SEGMENTS)
    assert kept == candidates
    assert dropped == 0
    assert usage == TokenUsage()


# --------------------------------------------------------------------- summary
def test_update_summary_uses_the_new_text_and_survives_failures():
    provider = MockProvider([{"summary": "リリースと予算の議論"}])
    summary, usage = update_summary(provider, "", SEGMENTS)
    assert summary == "リリースと予算の議論"
    assert usage.requests == 1

    class Boom(MockProvider):
        def generate_json(self, system, user, schema, schema_name):
            raise RuntimeError("down")

    unchanged, _ = update_summary(Boom([]), "前の要約", SEGMENTS)
    assert unchanged == "前の要約"


# --------------------------------------------------------------------- rolling
def test_generate_rolling_filters_duplicates_and_answered_questions():
    """summary -> generate -> answered filter, in that call order."""
    responses = [
        {"summary": "担当者と予算の話"},
        {
            "questions": [
                {
                    "text": "担当者はいつ決まりますか?",
                    "type": "unresolved",
                    "priority": 1,
                    "evidence_quote": "移行の担当者は未定",
                },
                {
                    "text": "予算はどれくらい残っていますか?",
                    "type": "risk",
                    "priority": 2,
                    "evidence_quote": "予算は80%消化",
                },
                {
                    "text": "リリース日は確定ですか?",
                    "type": "clarify",
                    "priority": 1,
                    "evidence_quote": "リリース日は来週の金曜",
                },
            ]
        },
    ]

    def responder(system: str, user: str) -> dict:
        """Answer the summary/generation calls in order, then mark 担当者 as answered."""
        if "累積要約" in user:
            return responses[0]
        if "候補の質問" in user:
            verdicts = []
            for line in user.splitlines():
                head, _, rest = line.partition(":")
                if head.strip().isdigit():
                    verdicts.append(
                        {"index": int(head.strip()), "answered": "担当者" in rest}
                    )
            return {"verdicts": verdicts}
        return responses[1]

    provider = MockProvider(responder)
    result = generate_rolling(
        provider,
        SEGMENTS,
        existing_question_texts=["リリース日は確定していますか？"],
        previous_summary="",
        window_sec=1000.0,
    )
    texts = [q.text for q in result.questions]
    assert result.summary == "担当者と予算の話"
    assert result.dropped_duplicates == 1  # the release-date question was already asked
    assert result.dropped_answered == 1  # the owner question got answered later
    assert texts == ["予算はどれくらい残っていますか?"]
    assert result.usage.requests == 3


def test_generate_rolling_validates_evidence_against_the_full_transcript():
    provider = MockProvider(
        [
            {"summary": "s"},
            {
                "questions": [
                    {
                        "text": "でっち上げ",
                        "type": "clarify",
                        "priority": 1,
                        "evidence_quote": "この文はどこにもない",
                    }
                ]
            },
            {"questions": []},
            {"questions": []},
        ]
    )
    result = generate_rolling(provider, SEGMENTS, [], "", window_sec=1000.0)
    assert result.questions == []


def test_generate_rolling_on_empty_transcript_calls_nothing():
    provider = MockProvider([])
    result = generate_rolling(provider, [], [], "", window_sec=1000.0)
    assert result.questions == []
    assert provider.calls == 0
