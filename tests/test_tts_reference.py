import numpy as np
import pytest

from core.transcription.models import Segment, Transcription, Word
from core.tts.reference import extract_reference_audio, select_reference_segments


def _seg(start, end, no_speech_prob=0.01):
    return Segment(
        text="placeholder",
        start=start,
        end=end,
        words=(),
        avg_logprob=-0.3,
        no_speech_prob=no_speech_prob,
    )


def _transcription(segments):
    return Transcription(
        language="es",
        segments=tuple(segments),
        duration=segments[-1].end if segments else 0.0,
        model_size="medium",
    )


class TestSelectReferenceSegments:
    def test_selects_lowest_no_speech_prob_first(self):
        segs = [
            _seg(0.0, 2.0, no_speech_prob=0.5),
            _seg(2.0, 4.0, no_speech_prob=0.01),
            _seg(4.0, 6.0, no_speech_prob=0.1),
            _seg(6.0, 8.0, no_speech_prob=0.02),
        ]
        result = select_reference_segments(_transcription(segs), min_duration=4.0)
        # Should pick seg[1] (0.01) and seg[3] (0.02) = 4.0s total
        assert len(result) == 2
        assert result[0].no_speech_prob < result[1].no_speech_prob or \
            result[0].start < result[1].start

    def test_accumulates_to_min_duration(self):
        segs = [
            _seg(0.0, 3.0, no_speech_prob=0.01),
            _seg(3.0, 5.0, no_speech_prob=0.02),
            _seg(5.0, 8.0, no_speech_prob=0.03),
        ]
        result = select_reference_segments(_transcription(segs), min_duration=6.0)
        total = sum(s.end - s.start for s in result)
        assert total >= 6.0

    def test_fallback_returns_all_if_total_less_than_min(self):
        segs = [
            _seg(0.0, 2.0, no_speech_prob=0.01),
            _seg(2.0, 3.0, no_speech_prob=0.02),
        ]
        result = select_reference_segments(_transcription(segs), min_duration=6.0)
        assert len(result) == 2

    def test_returns_in_temporal_order(self):
        segs = [
            _seg(0.0, 3.0, no_speech_prob=0.5),
            _seg(3.0, 6.0, no_speech_prob=0.01),
            _seg(6.0, 9.0, no_speech_prob=0.1),
        ]
        result = select_reference_segments(_transcription(segs), min_duration=3.0)
        for a, b in zip(result, result[1:]):
            assert a.start <= b.start

    def test_empty_transcription_returns_empty(self):
        result = select_reference_segments(_transcription([]), min_duration=6.0)
        assert result == []


class TestExtractReferenceAudio:
    def test_extracts_correct_slices(self):
        audio = np.arange(80000, dtype=np.float32)  # 5s at 16kHz
        segs = [_seg(1.0, 2.0), _seg(3.0, 4.0)]
        result = extract_reference_audio(audio, segs, sample_rate=16000)
        expected_len = 16000 + 16000  # 1s + 1s
        assert len(result) == expected_len
        np.testing.assert_array_equal(result[:16000], audio[16000:32000])
        np.testing.assert_array_equal(result[16000:], audio[48000:64000])

    def test_single_segment(self):
        audio = np.arange(48000, dtype=np.float32)  # 3s at 16kHz
        segs = [_seg(0.0, 2.0)]
        result = extract_reference_audio(audio, segs, sample_rate=16000)
        assert len(result) == 32000

    def test_empty_segments_returns_empty(self):
        audio = np.arange(16000, dtype=np.float32)
        result = extract_reference_audio(audio, [], sample_rate=16000)
        assert len(result) == 0
