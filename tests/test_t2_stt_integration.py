"""T2 — STT integration against the real OpenAI transcription API.

No mocking here by design. The only audio sent is the SAPI fixture.
"""

from __future__ import annotations

import pytest

from meetq.audio.chunker import VadChunker
from meetq.audio.file_source import decode_to_pcm
from meetq.qgen.validate import normalize
from meetq.stt.base import TranscriptionContext
from meetq.stt.openai_file import OpenAIFileTranscriber

pytestmark = pytest.mark.t2


@pytest.fixture(scope="module")
def transcription(request):
    manifest = request.getfixturevalue("manifest")
    wav = request.getfixturevalue("fixture_wav")
    pcm = decode_to_pcm(wav)
    audio_sec = len(pcm) / 16000
    chunks = VadChunker().chunk(pcm)
    transcriber = OpenAIFileTranscriber()
    result = transcriber.transcribe_chunks(
        chunks,
        TranscriptionContext(language=manifest["language"], keywords=manifest["keywords"]),
    )
    return {
        "manifest": manifest,
        "audio_sec": audio_sec,
        "chunks": chunks,
        "result": result,
        "model": transcriber.model,
    }


def test_transcription_contains_most_keywords(transcription):
    manifest = transcription["manifest"]
    full = normalize("".join(s.text for s in transcription["result"].segments))
    found = [kw for kw in manifest["keywords"] if normalize(kw) in full]
    print(
        f"T2 model={transcription['model']} segments={len(transcription['result'].segments)} "
        f"keywords found={found} of {manifest['keywords']}"
    )
    assert len(found) > len(manifest["keywords"]) / 2, (
        f"only {found} of {manifest['keywords']} were transcribed"
    )


def test_segment_timestamps_are_monotonic_and_within_audio(transcription):
    segments = transcription["result"].segments
    audio_sec = transcription["audio_sec"]
    assert segments, "no segments produced"
    previous = -1.0
    for seg in segments:
        assert seg.start_sec > previous, f"start_sec not increasing at segment {seg.index}"
        assert 0.0 <= seg.start_sec <= audio_sec, f"{seg.start_sec} outside 0..{audio_sec}"
        assert seg.start_sec < seg.end_sec <= audio_sec + 0.05
        previous = seg.start_sec


def test_usage_is_measured(transcription):
    result = transcription["result"]
    assert result.requests == len(transcription["chunks"])
    assert result.transcribed_sec == pytest.approx(transcription["audio_sec"], abs=0.5)
