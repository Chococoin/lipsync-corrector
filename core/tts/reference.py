"""Voice reference selection for XTTS-v2 cloning."""

from __future__ import annotations

import numpy as np

from core.transcription.models import Segment, Transcription


def select_reference_segments(
    transcription: Transcription,
    min_duration: float = 6.0,
) -> list[Segment]:
    """Select the best segments for voice cloning reference.

    Picks segments with the lowest no_speech_prob (most likely to be clean
    speech) until at least min_duration seconds are accumulated. Returns
    segments in temporal order. Falls back to all segments if total audio
    is shorter than min_duration.
    """
    if not transcription.segments:
        return []

    ranked = sorted(transcription.segments, key=lambda s: s.no_speech_prob)

    selected: list[Segment] = []
    total = 0.0
    for seg in ranked:
        selected.append(seg)
        total += seg.end - seg.start
        if total >= min_duration:
            break

    selected.sort(key=lambda s: s.start)
    return selected


def extract_reference_audio(
    source_wav: np.ndarray,
    segments: list[Segment],
    sample_rate: int = 16000,
) -> np.ndarray:
    """Extract and concatenate audio slices for the selected reference segments.

    Args:
        source_wav: 1-D float32 array of the full source audio.
        segments: segments to extract (should be in temporal order).
        sample_rate: sample rate of source_wav.

    Returns:
        1-D float32 array of the concatenated reference audio.
    """
    if not segments:
        return np.array([], dtype=np.float32)

    slices = []
    for seg in segments:
        start_sample = int(seg.start * sample_rate)
        end_sample = int(seg.end * sample_rate)
        end_sample = min(end_sample, len(source_wav))
        if start_sample < end_sample:
            slices.append(source_wav[start_sample:end_sample])

    if not slices:
        return np.array([], dtype=np.float32)
    return np.concatenate(slices)
