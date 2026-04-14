"""JSON and SRT serializers for Transcription objects."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from core.transcription.models import Transcription


def write_json(transcription: Transcription, out_path: Path | str) -> None:
    """Write a Transcription to a pretty-printed UTF-8 JSON file.

    Tuples are serialized as JSON arrays. Non-ASCII characters are preserved
    in the output (ensure_ascii=False).
    """
    data = asdict(transcription)
    out = Path(out_path)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_srt(transcription: Transcription, out_path: Path | str) -> None:
    """Write a Transcription as a SubRip (.srt) subtitle file.

    One subtitle per segment. Indices are 1-based. Timestamps use the
    standard HH:MM:SS,mmm format.
    """
    out = Path(out_path)
    blocks = []
    for i, seg in enumerate(transcription.segments, start=1):
        start_ts = _format_srt_timestamp(seg.start)
        end_ts = _format_srt_timestamp(seg.end)
        blocks.append(f"{i}\n{start_ts} --> {end_ts}\n{seg.text}\n")
    out.write_text("\n".join(blocks), encoding="utf-8")


def _format_srt_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to SRT timestamp HH:MM:SS,mmm."""
    total_ms = int(round(seconds * 1000))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
