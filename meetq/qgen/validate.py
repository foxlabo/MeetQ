"""Evidence validation and duplicate removal (F4 / F10)."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from meetq.qgen.base import QUESTION_TYPES, GeneratedQuestion

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC + whitespace removal + casefold. Used for substring evidence checks."""
    return _WS.sub("", unicodedata.normalize("NFKC", text or "")).casefold()


class TranscriptIndex:
    """Normalised transcript with a map back to segment ids / start seconds."""

    def __init__(self, segments: list[dict]):
        self.segments = segments
        self._norm_parts: list[str] = []
        self._owner: list[int] = []  # index into segments, per normalised character
        for i, seg in enumerate(segments):
            norm = normalize(seg.get("text", ""))
            self._norm_parts.append(norm)
            self._owner.extend([i] * len(norm))
        self.normalized = "".join(self._norm_parts)

    def locate(self, quote: str) -> tuple[list[int], float] | None:
        """Return (segment_ids, start_sec) when the quote is a substring, else None."""
        norm_quote = normalize(quote)
        if not norm_quote:
            return None
        pos = self.normalized.find(norm_quote)
        if pos < 0:
            return None
        owners = sorted(set(self._owner[pos : pos + len(norm_quote)]))
        if not owners:
            return None
        seg_ids = [int(self.segments[i]["id"]) for i in owners]
        start_sec = float(self.segments[owners[0]].get("start_sec", 0.0))
        return seg_ids, start_sec


def validate_question(raw: dict, index: TranscriptIndex) -> GeneratedQuestion | None:
    """Return a question only when it is well formed AND its evidence is a real quote."""
    text = str(raw.get("text", "")).strip()
    qtype = str(raw.get("type", "")).strip()
    quote = str(raw.get("evidence_quote", "")).strip()
    if not text or qtype not in QUESTION_TYPES or not quote:
        return None
    try:
        priority = int(raw.get("priority", 2))
    except (TypeError, ValueError):
        return None
    if priority < 1 or priority > 3:
        return None
    located = index.locate(quote)
    if located is None:
        return None
    seg_ids, start_sec = located
    return GeneratedQuestion(
        text=text,
        type=qtype,
        priority=priority,
        evidence_quote=quote,
        evidence_start_sec=start_sec,
        evidence_segment_ids=seg_ids,
    )


def dedupe(
    questions: list[GeneratedQuestion], threshold: int = 88
) -> list[GeneratedQuestion]:
    """Drop near-duplicate question texts, keeping the highest priority one."""
    kept: list[GeneratedQuestion] = []
    for q in sorted(questions, key=lambda x: (x.priority, -len(x.text))):
        norm = normalize(q.text)
        if any(fuzz.token_set_ratio(norm, normalize(k.text)) >= threshold for k in kept):
            continue
        kept.append(q)
    return kept


def prioritize(questions: list[GeneratedQuestion]) -> list[GeneratedQuestion]:
    return sorted(questions, key=lambda q: (q.priority, q.evidence_start_sec))
