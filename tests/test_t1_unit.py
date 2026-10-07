"""T1 — unit tests for the chunker and the question pipeline. OpenAI is mocked here
(and ONLY here: T2..T5 always use the real API)."""

from __future__ import annotations

import numpy as np
import pytest

from meetq.audio.chunker import EnergyVad, VadChunker, WebRtcVad
from meetq.qgen.base import QUESTION_TYPES, question_schema
from meetq.qgen.mock import MockProvider
from meetq.qgen.pipeline import build_topic_chunks, generate_questions
from meetq.qgen.validate import TranscriptIndex, dedupe, normalize, validate_question

pytestmark = pytest.mark.t1

SR = 16000


def make_pcm(pattern: list[tuple[float, bool]]) -> np.ndarray:
    """Build PCM from (seconds, is_speech) pairs. Speech is a loud tone."""
    parts = []
    rng = np.random.default_rng(0)
    for seconds, speech in pattern:
        n = int(seconds * SR)
        if speech:
            t = np.arange(n) / SR
            wave = 9000 * np.sin(2 * np.pi * 220 * t) + rng.normal(0, 400, n)
            parts.append(wave.astype(np.int16))
        else:
            parts.append(np.zeros(n, dtype=np.int16))
    return np.concatenate(parts).astype(np.int16)


# ----------------------------------------------------------------- chunker
def test_chunker_backends_available():
    assert WebRtcVad().name == "webrtcvad"
    assert EnergyVad().name == "energy"


@pytest.mark.parametrize("vad", [EnergyVad(), WebRtcVad()])
def test_chunks_are_monotonic_contiguous_and_bounded(vad):
    pcm = make_pcm([(12, True), (1.0, False), (12, True), (1.0, False), (30, True)])
    chunker = VadChunker(target_sec=20, min_sec=10, max_sec=60, vad=vad)
    chunks = chunker.chunk(pcm)

    assert chunks, "chunker returned nothing"
    total_sec = len(pcm) / SR
    assert chunks[0].start_sec == 0.0
    assert chunks[-1].end_sec == pytest.approx(total_sec, abs=0.05)
    for i, c in enumerate(chunks):
        assert c.index == i
        assert c.start_sec < c.end_sec
        assert c.duration_sec <= 60.0 + 0.05
        if i:
            assert c.start_sec == pytest.approx(chunks[i - 1].end_sec, abs=1e-6)
            assert c.start_sec > chunks[i - 1].start_sec


def test_chunker_splits_long_audio_and_respects_max():
    pcm = make_pcm([(200, True)])
    chunks = VadChunker(target_sec=20, min_sec=10, max_sec=60, vad=EnergyVad()).chunk(pcm)
    assert len(chunks) >= 3
    assert max(c.duration_sec for c in chunks) <= 60.05


def test_chunker_empty_input_and_channel_tag():
    assert VadChunker(vad=EnergyVad()).chunk(np.zeros(0, dtype=np.int16)) == []
    chunks = VadChunker(vad=EnergyVad()).chunk(make_pcm([(11, True)]), channel="self")
    assert {c.channel for c in chunks} == {"self"}


def test_chunker_rejects_bad_bounds():
    with pytest.raises(ValueError):
        VadChunker(target_sec=5, min_sec=10, max_sec=60)


# ------------------------------------------------------------- validation
SEGMENTS = [
    {
        "id": 11,
        "start_sec": 0.0,
        "end_sec": 10.0,
        "text": "リリース日は来週の金曜を予定しています。",
    },
    {"id": 12, "start_sec": 10.0, "end_sec": 20.0, "text": "予算はすでに80%を消化しています。"},
    {"id": 13, "start_sec": 20.0, "end_sec": 30.0, "text": "移行作業の担当者はまだ未定です。"},
]


def test_normalize_ignores_whitespace_and_width():
    assert normalize("ＡＢ Ｃ") == normalize("abc")


def test_transcript_index_locates_quotes_across_segments():
    index = TranscriptIndex(SEGMENTS)
    ids, start = index.locate("予算はすでに80%を消化")
    assert ids == [12]
    assert start == 10.0
    spanning = index.locate("予定しています。予算はすでに")
    assert spanning is not None
    assert spanning[0] == [11, 12]
    assert index.locate("存在しない引用です") is None


def test_validate_question_accepts_good_and_rejects_bad():
    index = TranscriptIndex(SEGMENTS)
    good = validate_question(
        {
            "text": "リリース日は確定ですか?",
            "type": "clarify",
            "priority": 1,
            "evidence_quote": "リリース日は来週の金曜",
        },
        index,
    )
    assert good is not None
    assert good.evidence_segment_ids == [11]
    assert good.evidence_start_sec == 0.0

    bad_cases = [
        {"text": "", "type": "clarify", "priority": 1, "evidence_quote": "リリース日"},
        {"text": "q", "type": "invalid", "priority": 1, "evidence_quote": "リリース日"},
        {"text": "q", "type": "clarify", "priority": 9, "evidence_quote": "リリース日"},
        {"text": "q", "type": "clarify", "priority": 1, "evidence_quote": "作り話の引用"},
        {"text": "q", "type": "clarify", "priority": 1, "evidence_quote": ""},
    ]
    for raw in bad_cases:
        assert validate_question(raw, index) is None, raw


def test_dedupe_removes_near_duplicates_and_keeps_priority():
    index = TranscriptIndex(SEGMENTS)
    base = {"type": "clarify", "evidence_quote": "リリース日は来週の金曜"}
    qs = [
        validate_question({**base, "text": "リリース日は確定していますか?", "priority": 2}, index),
        validate_question({**base, "text": "リリース日は確定していますか？", "priority": 1}, index),
        validate_question(
            {**base, "text": "予算の追加申請は誰が行いますか?", "priority": 3}, index
        ),
    ]
    kept = dedupe([q for q in qs if q])
    assert len(kept) == 2
    assert kept[0].priority == 1


# --------------------------------------------------------------- schema
def test_question_schema_is_strict_and_typed():
    schema = question_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["questions"]
    item = schema["properties"]["questions"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])
    assert item["properties"]["type"]["enum"] == list(QUESTION_TYPES)


# --------------------------------------------------------------- pipeline
def test_build_topic_chunks_groups_segments():
    many = [
        {"id": i, "start_sec": i * 30.0, "end_sec": i * 30.0 + 30.0, "text": "あ" * 500}
        for i in range(10)
    ]
    groups = build_topic_chunks(many)
    assert len(groups) > 1
    assert sum(len(g) for g in groups) == len(many)
    assert [s["id"] for g in groups for s in g] == [s["id"] for s in many]


def test_pipeline_discards_hallucinated_evidence_and_retries():
    """A first response with a fabricated quote must be dropped and re-requested."""
    responses = [
        {
            "questions": [
                {
                    "text": "でっち上げの根拠",
                    "type": "clarify",
                    "priority": 1,
                    "evidence_quote": "この文はどこにも存在しません",
                },
                {
                    "text": "リリース日は確定ですか?",
                    "type": "clarify",
                    "priority": 1,
                    "evidence_quote": "リリース日は来週の金曜",
                },
            ]
        },
        {
            "questions": [
                {
                    "text": "担当者はいつ決まりますか?",
                    "type": "unresolved",
                    "priority": 2,
                    "evidence_quote": "移行作業の担当者はまだ未定",
                }
            ]
        },
    ]
    provider = MockProvider(responses)
    questions, usage = generate_questions(provider, SEGMENTS, mode="followup")
    texts = [q.text for q in questions]
    assert "でっち上げの根拠" not in texts
    assert "リリース日は確定ですか?" in texts
    assert "担当者はいつ決まりますか?" in texts
    assert provider.calls >= 2, "the pipeline must retry after discarding a question"
    assert usage.requests == provider.calls


def test_pipeline_gives_up_after_two_retries_and_returns_nothing_hardcoded():
    provider = MockProvider(
        [
            {
                "questions": [
                    {
                        "text": "根拠なし",
                        "type": "risk",
                        "priority": 1,
                        "evidence_quote": "存在しない引用",
                    }
                ]
            }
        ]
    )
    questions, _ = generate_questions(provider, SEGMENTS, mode="followup")
    assert questions == []
    assert provider.calls == 3  # initial + MAX_RETRIES(2)


def test_pipeline_returns_prioritized_questions():
    provider = MockProvider(
        [
            {
                "questions": [
                    {
                        "text": "予算は足りますか?",
                        "type": "risk",
                        "priority": 3,
                        "evidence_quote": "予算はすでに80%",
                    },
                    {
                        "text": "担当者は誰ですか?",
                        "type": "unresolved",
                        "priority": 1,
                        "evidence_quote": "担当者はまだ未定",
                    },
                ]
            }
        ]
    )
    questions, _ = generate_questions(provider, SEGMENTS, mode="followup")
    assert [q.priority for q in questions] == sorted(q.priority for q in questions)
    assert questions[0].text == "担当者は誰ですか?"


def test_pipeline_survives_provider_errors():
    class Boom(MockProvider):
        def generate_json(self, system, user, schema, schema_name):
            raise RuntimeError("upstream down")

    questions, usage = generate_questions(Boom([]), SEGMENTS, mode="followup")
    assert questions == []
    assert usage.requests == 0


def test_empty_transcript_produces_no_questions():
    provider = MockProvider([{"questions": []}])
    questions, _ = generate_questions(provider, [], mode="followup")
    assert questions == []
    assert provider.calls == 0
