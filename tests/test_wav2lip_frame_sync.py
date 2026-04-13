import numpy as np
import pytest

from core.wav2lip.frame_sync import MEL_STEP_SIZE, get_mel_chunk_for_frame


class TestConstants:
    def test_mel_step_size_is_16(self):
        assert MEL_STEP_SIZE == 16


class TestGetMelChunkForFrame:
    def _mel(self, t=500):
        return np.arange(80 * t, dtype=np.float32).reshape(80, t)

    def test_returns_correct_shape(self):
        mel = self._mel()
        chunk = get_mel_chunk_for_frame(mel, frame_idx=0, fps=25.0)
        assert chunk.shape == (80, 16)

    def test_frame_zero_starts_at_column_zero(self):
        mel = self._mel()
        chunk = get_mel_chunk_for_frame(mel, frame_idx=0, fps=25.0)
        np.testing.assert_array_equal(chunk, mel[:, 0:16])

    def test_frame_one_at_25fps_starts_at_column_3(self):
        # int(80 * 1 / 25) == 3
        mel = self._mel()
        chunk = get_mel_chunk_for_frame(mel, frame_idx=1, fps=25.0)
        np.testing.assert_array_equal(chunk, mel[:, 3:19])

    def test_frame_two_at_24fps_starts_at_column_6(self):
        # int(80 * 2 / 24) == 6
        mel = self._mel()
        chunk = get_mel_chunk_for_frame(mel, frame_idx=2, fps=24.0)
        np.testing.assert_array_equal(chunk, mel[:, 6:22])

    def test_near_end_is_left_padded(self):
        mel = self._mel(t=20)
        chunk = get_mel_chunk_for_frame(mel, frame_idx=10, fps=25.0)
        assert chunk.shape == (80, 16)

    def test_beyond_end_still_returns_shape_16(self):
        mel = self._mel(t=10)
        chunk = get_mel_chunk_for_frame(mel, frame_idx=100, fps=25.0)
        assert chunk.shape == (80, 16)
