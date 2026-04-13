import numpy as np
import pytest

from core.wav2lip.audio import (
    SAMPLE_RATE,
    N_MELS,
    load_wav_mono_16k,
    melspectrogram,
)


class TestConstants:
    def test_sample_rate_is_16k(self):
        assert SAMPLE_RATE == 16000

    def test_n_mels_is_80(self):
        assert N_MELS == 80


class TestLoadWavMono16k:
    def test_returns_float32_1d(self, tmp_path):
        import soundfile as sf
        t = np.linspace(0, 1.0, 48000, endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        path = tmp_path / "sine.wav"
        sf.write(str(path), y, 48000)
        out = load_wav_mono_16k(path)
        assert out.dtype == np.float32
        assert out.ndim == 1

    def test_resamples_to_16k(self, tmp_path):
        import soundfile as sf
        t = np.linspace(0, 1.0, 48000, endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        path = tmp_path / "sine.wav"
        sf.write(str(path), y, 48000)
        out = load_wav_mono_16k(path)
        assert abs(len(out) - 16000) < 100  # allow resampler edge effects

    def test_downmixes_stereo_to_mono(self, tmp_path):
        import soundfile as sf
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = 0.5 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
        stereo = np.stack([left, right], axis=1)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, 16000)
        out = load_wav_mono_16k(path)
        assert out.ndim == 1


class TestMelSpectrogram:
    def test_shape_has_80_mel_bands(self):
        y = np.random.randn(16000).astype(np.float32) * 0.1
        m = melspectrogram(y)
        assert m.shape[0] == 80

    def test_returns_float32(self):
        y = np.random.randn(16000).astype(np.float32) * 0.1
        m = melspectrogram(y)
        assert m.dtype == np.float32

    def test_silence_is_finite(self):
        y = np.zeros(16000, dtype=np.float32)
        m = melspectrogram(y)
        assert np.all(np.isfinite(m))

    def test_longer_audio_gives_more_time_frames(self):
        short = np.random.randn(16000).astype(np.float32) * 0.1
        long = np.random.randn(32000).astype(np.float32) * 0.1
        m_short = melspectrogram(short)
        m_long = melspectrogram(long)
        assert m_long.shape[1] > m_short.shape[1]
