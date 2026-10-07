"""F9/F10 — rolling question generation over a sliding window plus a cumulative
summary, with duplicate removal and an "already answered in the meeting" filter."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from meetq.qgen.base import GeneratedQuestion, LLMProvider, TokenUsage
from meetq.qgen.pipeline import generate_questions
from meetq.qgen.validate import TranscriptIndex, normalize, prioritize

log = logging.getLogger(__name__)

# Japanese questions carry no word boundaries, so token_set_ratio degenerates to
# plain ratio. Measured on real paraphrases: restatements score ~82 while genuinely
# different questions stay below 63, so 78 separates them cleanly.
DUPLICATE_THRESHOLD = 78


def similarity(a: str, b: str) -> float:
    return max(fuzz.ratio(a, b), fuzz.token_set_ratio(a, b))

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

ANSWERED_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "answered": {"type": "boolean"},
                },
                "required": ["index", "answered"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

SUMMARY_SYSTEM = (
    "あなたは進行中の会議の要約を更新するアシスタントです。"
    "これまでの要約と新しい発言を受け取り、会議全体の累積要約を日本語 400 字以内で"
    "書き直してください。決定事項・未決事項・数値・期日を優先して残します。"
)

ANSWERED_SYSTEM = (
    "あなたは会議の文字起こしを読み、候補の質問が**その会議の中で既に回答されているか**を"
    "判定するアシスタントです。文字起こしに明確な答えがある場合だけ answered=true にします。"
    "答えが曖昧、保留、未定の場合は answered=false です。"
)


@dataclass
class RollingResult:
    questions: list[GeneratedQuestion] = field(default_factory=list)
    summary: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    window_segments: int = 0
    dropped_duplicates: int = 0
    dropped_answered: int = 0


def window_segments(segments: list[dict], window_sec: float) -> list[dict]:
    """The most recent `window_sec` of transcript."""
    if not segments:
        return []
    latest = max(float(s.get("end_sec", 0.0)) for s in segments)
    cutoff = latest - window_sec
    return [s for s in segments if float(s.get("end_sec", 0.0)) >= cutoff]


def update_summary(
    provider: LLMProvider, previous: str, segments: list[dict]
) -> tuple[str, TokenUsage]:
    if not segments:
        return previous, TokenUsage()
    texts = [(s.get("text") or "").strip() for s in segments]
    body = "\n".join(t for t in texts if t)
    if not body:
        return previous, TokenUsage()
    user = (
        f"これまでの累積要約:\n{previous or '(まだありません)'}\n\n"
        f"新しい発言:\n{body}\n\n"
        "更新後の累積要約だけを返してください。"
    )
    try:
        payload, usage = provider.generate_json(
            SUMMARY_SYSTEM, user, SUMMARY_SCHEMA, "meetq_summary"
        )
    except Exception as exc:  # noqa: BLE001 - a failed summary must not stop generation
        log.warning("summary update failed: %s", exc)
        return previous, TokenUsage()
    return str(payload.get("summary", previous) or previous), usage


def drop_duplicates(
    candidates: list[GeneratedQuestion], existing: list[str], threshold: int = DUPLICATE_THRESHOLD
) -> tuple[list[GeneratedQuestion], int]:
    """Remove candidates that repeat an already-stored question (F10)."""
    known = [normalize(t) for t in existing]
    kept: list[GeneratedQuestion] = []
    dropped = 0
    for q in candidates:
        norm = normalize(q.text)
        if any(similarity(norm, k) >= threshold for k in known):
            dropped += 1
            continue
        kept.append(q)
        known.append(norm)
    return kept, dropped


def drop_already_answered(
    provider: LLMProvider, candidates: list[GeneratedQuestion], segments: list[dict]
) -> tuple[list[GeneratedQuestion], int, TokenUsage]:
    """Drop questions the meeting already answered (F10)."""
    if not candidates or not segments:
        return candidates, 0, TokenUsage()
    transcript = "\n".join(
        f"[{float(s.get('start_sec', 0.0)):.0f}s] {s.get('text', '')}" for s in segments
    )
    listing = "\n".join(
        f"{i}: {q.text}（根拠: {q.evidence_quote}）" for i, q in enumerate(candidates)
    )
    user = (
        f"会議の文字起こし:\n---\n{transcript}\n---\n\n"
        f"候補の質問:\n{listing}\n\n"
        "各 index について、その質問の答えが文字起こしの中に既にあるかを判定してください。"
    )
    try:
        payload, usage = provider.generate_json(
            ANSWERED_SYSTEM, user, ANSWERED_SCHEMA, "meetq_answered"
        )
    except Exception as exc:  # noqa: BLE001 - keep the questions if the filter fails
        log.warning("answered filter failed: %s", exc)
        return candidates, 0, TokenUsage()
    answered = {
        int(v["index"])
        for v in payload.get("verdicts", [])
        if isinstance(v, dict) and v.get("answered") is True
    }
    kept = [q for i, q in enumerate(candidates) if i not in answered]
    return kept, len(candidates) - len(kept), usage


def generate_rolling(
    provider: LLMProvider,
    segments: list[dict],
    existing_question_texts: list[str],
    previous_summary: str,
    mode: str = "followup",
    window_sec: float = 300.0,
    per_chunk: int = 3,
) -> RollingResult:
    """One rolling round: recent window + cumulative summary -> new questions only."""
    result = RollingResult(summary=previous_summary)
    segments = [s for s in segments if (s.get("text") or "").strip()]
    if not segments:
        return result

    recent = window_segments(segments, window_sec)
    result.window_segments = len(recent)

    summary, summary_usage = update_summary(provider, previous_summary, recent)
    result.summary = summary
    result.usage.add(summary_usage)

    context = ""
    if summary:
        context = f"ここまでの会議の要約:\n{summary}\n\n"
    if existing_question_texts:
        context += (
            "既に作成済みの質問（これらと重複する質問は作らないでください）:\n"
            + json.dumps(existing_question_texts[-20:], ensure_ascii=False)
            + "\n\n"
        )

    candidates, gen_usage = generate_questions(
        provider, recent, mode=mode, per_chunk=per_chunk, context=context
    )
    result.usage.add(gen_usage)
    if not candidates:
        return result

    # evidence is validated against the FULL transcript, not just the window
    index = TranscriptIndex(segments)
    revalidated: list[GeneratedQuestion] = []
    for q in candidates:
        located = index.locate(q.evidence_quote)
        if located is None:
            continue
        q.evidence_segment_ids, q.evidence_start_sec = located
        revalidated.append(q)

    kept, dropped_dupes = drop_duplicates(revalidated, existing_question_texts)
    result.dropped_duplicates = dropped_dupes

    kept, dropped_answered, answered_usage = drop_already_answered(provider, kept, segments)
    result.dropped_answered = dropped_answered
    result.usage.add(answered_usage)

    result.questions = prioritize(kept)
    return result
