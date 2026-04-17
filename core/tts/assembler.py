"""Assemble per-segment audio into a single track with correct timing."""

from __future__ import annotations

import numpy as np

from core.transcription.models import Segment


def assemble_track(
    segment_audios: list[np.ndarray],
    segments: tuple[Segment, ...] | list[Segment],
    total_duration: float,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Position generated audio segments at their original timestamps.

    Creates a silence array of total_duration length and pastes each
    segment's audio at its start position. Audio that overflows the
    segment's time slot is truncated.

    Args:
        segment_audios: list of 1-D float32 arrays, one per segment.
        segments: the Segment objects with start/end timestamps.
        total_duration: total duration of the output track in seconds.
        sample_rate: sample rate of the audio arrays.

    Returns:
        1-D float32 array of the assembled track.
    """
    total_samples = int(total_duration * sample_rate)
    if total_samples <= 0:
        return np.array([], dtype=np.float32)

    track = np.zeros(total_samples, dtype=np.float32)

    for audio, seg in zip(segment_audios, segments):
        start_sample = int(seg.start * sample_rate)
        slot_samples = int((seg.end - seg.start) * sample_rate)

        chunk = audio[:slot_samples] if len(audio) > slot_samples else audio

        end_sample = min(start_sample + len(chunk), total_samples)
        actual_len = end_sample - start_sample
        if actual_len > 0:
            track[start_sample:end_sample] = chunk[:actual_len]

    return track
