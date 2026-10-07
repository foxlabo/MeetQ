"""Question generation pipeline: segments -> topic chunks -> generate -> validate ->
dedupe -> prioritize (F4)."""

from __future__ import annotations

import json
import logging

from meetq.qgen.base import (
    GeneratedQuestion,
    LLMProvider,
    TokenUsage,
    question_schema,
)
from meetq.qgen.validate import TranscriptIndex, dedupe, prioritize, validate_question

log = logging.getLogger(__name__)

MAX_RETRIES = 2
TOPIC_CHAR_BUDGET = 1800
TOPIC_SEC_BUDGET = 300.0

SYSTEM_PROMPTS = {
    "followup": (
        "あなたは会議の議事録を読み込み、会議後のフォローアップとして関係者に投げるべき"
        "質問を作るアシスタントです。曖昧な点(clarify)、未解決の論点(unresolved)、"
        "矛盾(contradiction)、リスク(risk)に注目してください。"
        "質問は具体的で、その会議を知らない人でも意味が分かる日本語にしてください。"
    ),
    "comprehension": (
        "あなたは会議や講義の文字起こしから、聞き手の理解度を確認するための質問を作る"
        "アシスタントです。type は原則 comprehension を使い、内容の重要点を問う質問を"
        "日本語で作ってください。"
    ),
}

USER_TEMPLATE = (
    "以下は会議の文字起こしの一部です。\n"
    "---\n{body}\n---\n"
    "この範囲から質問を最大 {n} 件作ってください。\n"
    "制約:\n"
    "- evidence_quote は上の文字起こしから**一字一句そのまま**引用すること"
    "（要約・言い換え・句読点の追加をしない。10〜60文字程度）。\n"
    "- priority は 1(最優先)〜3(低) の整数。\n"
    "- 同じ内容の質問を繰り返さない。\n"
    "{extra}"
)


def build_topic_chunks(segments: list[dict]) -> list[list[dict]]:
    """Group consecutive segments into topic-sized windows."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    chars = 0
    for seg in segments:
        text = seg.get("text", "")
        if current and (
            chars + len(text) > TOPIC_CHAR_BUDGET
            or float(seg.get("start_sec", 0.0)) - float(current[0].get("start_sec", 0.0))
            > TOPIC_SEC_BUDGET
        ):
            chunks.append(current)
            current = []
            chars = 0
        current.append(seg)
        chars += len(text)
    if current:
        chunks.append(current)
    return chunks


def _render(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        ts = float(seg.get("start_sec", 0.0))
        lines.append(f"[{int(ts // 60):02d}:{int(ts % 60):02d}] {seg.get('text', '')}")
    return "\n".join(lines)


def generate_questions(
    provider: LLMProvider,
    segments: list[dict],
    mode: str = "followup",
    per_chunk: int = 4,
    context: str = "",
) -> tuple[list[GeneratedQuestion], TokenUsage]:
    """Run the full pipeline. Returns validated, deduped, prioritized questions."""
    usage = TokenUsage()
    segments = [s for s in segments if (s.get("text") or "").strip()]
    if not segments:
        return [], usage
    index = TranscriptIndex(segments)
    schema = question_schema()
    system = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["followup"])
    collected: list[GeneratedQuestion] = []

    for topic in build_topic_chunks(segments):
        body = _render(topic)
        rejected_notes = ""
        accepted_here: list[GeneratedQuestion] = []
        for attempt in range(MAX_RETRIES + 1):
            want = per_chunk - len(accepted_here)
            if want <= 0:
                break
            user = USER_TEMPLATE.format(
                body=body, n=want, extra=rejected_notes, context=context
            )
            try:
                payload, tokens = provider.generate_json(system, user, schema, "meetq_questions")
            except Exception as exc:  # noqa: BLE001 - one bad topic must not kill the run
                log.warning("question generation failed for a topic chunk: %s", exc)
                break
            usage.add(tokens)
            raw_questions = payload.get("questions", []) if isinstance(payload, dict) else []
            rejected: list[str] = []
            for raw in raw_questions:
                if not isinstance(raw, dict):
                    continue
                q = validate_question(raw, index)
                if q is None:
                    rejected.append(str(raw.get("evidence_quote", ""))[:60])
                    continue
                accepted_here.append(q)
            if len(accepted_here) >= per_chunk or not rejected:
                break
            if attempt < MAX_RETRIES:
                rejected_notes = (
                    "- 前回、次の evidence_quote は文字起こしに存在しなかったため破棄しました: "
                    + json.dumps(rejected[:3], ensure_ascii=False)
                    + "\n- 必ず上の文字起こし本文からコピーした文字列を使ってください。\n"
                )
        collected.extend(accepted_here)

    return prioritize(dedupe(collected)), usage
