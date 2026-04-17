import numpy as np
import pytest

from core.transcription.models import Segment
from core.tts.assembler import assemble_track


def _seg(start, end):
    return Segment(
        text="placeholder",
        start=start,
        end=end,
        words=(),
        avg_logprob=-0.3,
        no_speech_prob=0.01,
    )


class TestAssembleTrack:
    def test_output_has_correct_length(self):
        segments = (_seg(0.0, 1.0),)
        audios = [np.ones(24000, dtype=np.float32)]
        result = assemble_track(audios, segments, total_duration=2.0, sample_rate=24000)
        assert len(result) == 48000

    def test_segment_positioned_at_correct_timestamp(self):
        segments = (_seg(1.0, 2.0),)
        audio = np.ones(24000, dtype=np.float32) * 0.5
        result = assemble_track([audio], segments, total_duration=3.0, sample_rate=24000)
        # First second should be silence
        assert np.all(result[:24000] == 0.0)
        # Second second should be the audio
        assert np.all(result[24000:48000] == 0.5)
        # Third second should be silence
        assert np.all(result[48000:] == 0.0)

    def test_audio_longer_than_slot_is_truncated(self):
        segments = (_seg(0.0, 1.0),)
        audio = np.ones(48000, dtype=np.float32)  # 2s of audio for 1s slot
        result = assemble_track([audio], segments, total_duration=1.0, sample_rate=24000)
        assert len(result) == 24000
        assert np.all(result == 1.0)

    def test_audio_shorter_than_slot_leaves_silence(self):
        segments = (_seg(0.0, 2.0),)
        audio = np.ones(24000, dtype=np.float32)  # 1s of audio for 2s slot
        result = assemble_track([audio], segments, total_duration=2.0, sample_rate=24000)
        assert np.all(result[:24000] == 1.0)
        assert np.all(result[24000:] == 0.0)

    def test_multiple_segments_with_gaps(self):
        segments = (_seg(0.0, 1.0), _seg(2.0, 3.0))
        audio1 = np.ones(24000, dtype=np.float32) * 0.3
        audio2 = np.ones(24000, dtype=np.float32) * 0.7
        result = assemble_track([audio1, audio2], segments, total_duration=3.0, sample_rate=24000)
        # First second: audio1
        assert result[12000] == pytest.approx(0.3)
        # Second second: silence (gap)
        assert result[36000] == pytest.approx(0.0)
        # Third second: audio2
        assert result[60000] == pytest.approx(0.7)

    def test_zero_duration_returns_empty(self):
        result = assemble_track([], (), total_duration=0.0, sample_rate=24000)
        assert len(result) == 0
