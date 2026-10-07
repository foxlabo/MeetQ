"""Re-flow a transcript into one sentence per line for the text downloads.

Segments are the audio chunks that were sent to the STT model, cut at pauses rather
than at sentence ends, so their boundaries often fall mid-sentence. The downloads
join them back up and split at sentence ends.
"""

from __future__ import annotations

from dataclasses import dataclass

SENTENCE_ENDS = "。？！?!"
CLOSERS = "」』）)】〉》\"'”’"


@dataclass
class Sentence:
    start_sec: float
    channel: str
    text: str


def _joins_words(left: str, right: str) -> bool:
    """Two ASCII words meeting at a chunk boundary need a space between them."""
    return bool(left and right) and left[-1].isascii() and left[-1].isalnum() and (
        right[0].isascii() and right[0].isalnum()
    )


def reflow(segments: list[dict]) -> list[Sentence]:
    """Join consecutive segments and split them at sentence ends.

    A sentence never spans two channels (self / others are different speakers).
    When a sentence starts inside a segment, its time is estimated from the
    character position within that segment — segments only carry chunk-level times.
    """
    sentences: list[Sentence] = []
    buf = ""
    buf_start = 0.0
    buf_channel = ""

    def flush() -> None:
        nonlocal buf
        text = buf.strip()
        if text:
            sentences.append(Sentence(start_sec=buf_start, channel=buf_channel, text=text))
        buf = ""

    for seg in sorted(segments, key=lambda s: float(s.get("start_sec", 0.0))):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        channel = seg.get("channel") or "mixed"
        start = float(seg.get("start_sec", 0.0))
        end = max(float(seg.get("end_sec", start)), start)
        if buf and channel != buf_channel:
            flush()
        if buf and _joins_words(buf, text):
            buf += " "

        i = 0
        while i < len(text):
            ch = text[i]
            if not buf.strip():
                if ch.isspace():
                    i += 1
                    continue
                buf = ""
                buf_start = start + (end - start) * (i / len(text))
                buf_channel = channel
            buf += ch
            i += 1
            if ch in SENTENCE_ENDS:
                # keep 「…です。」 and 「？！」 together with their sentence
                while i < len(text) and (text[i] in CLOSERS or text[i] in SENTENCE_ENDS):
                    buf += text[i]
                    i += 1
                flush()
    flush()
    return sentences
