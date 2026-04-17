from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from core.transcription.models import Segment, Transcription
from core.tts.synthesizer import synthesize

# Detect if XTTS model is available
_TTS_CACHE = Path.home() / ".local" / "share" / "tts"
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
XTTS_AVAILABLE = (
    (_TTS_CACHE / "tts_models--multilingual--multi-dataset--xtts_v2").exists()
    or (any(_HF_CACHE.glob("*xtts*")) if _HF_CACHE.exists() else False)
)
requires_xtts = pytest.mark.skipif(
    not XTTS_AVAILABLE,
    reason="XTTS-v2 model not downloaded",
)


def _transcription(segment_texts, language="en"):
    segments = tuple(
        Segment(
            text=text,
            start=float(i * 2),
            end=float(i * 2 + 1.8),
            words=(),
            avg_logprob=-0.3,
            no_speech_prob=0.01,
        )
        for i, text in enumerate(segment_texts)
    )
    return Transcription(
        language=language,
        segments=segments,
        duration=float(len(segment_texts) * 2) if segment_texts else 0.0,
        model_size="medium",
    )


def _make_source_wav(tmp_path, duration_s=3.0, sample_rate=16000):
    n = int(duration_s * sample_rate)
    wav = np.random.randn(n).astype(np.float32) * 0.1
    path = tmp_path / "source.wav"
    sf.write(str(path), wav, sample_rate)
    return path


class TestSynthesizerUnit:
    def test_missing_source_raises_filenotfound(self, tmp_path):
        t = _transcription(["Hello."])
        missing = tmp_path / "not_there.wav"
        with pytest.raises(FileNotFoundError):
            synthesize(t, missing, tmp_path / "out.wav")

    def test_empty_segments_writes_silence(self, tmp_path):
        t = Transcription(
            language="en", segments=(), duration=2.0, model_size="medium"
        )
        source = _make_source_wav(tmp_path)
        output = tmp_path / "out.wav"
        synthesize(t, source, output)
        assert output.exists()
        data, sr = sf.read(str(output))
        assert len(data) > 0
        assert np.allclose(data, 0.0, atol=1e-6)


@requires_xtts
class TestSynthesizerIntegration:
    def test_synthesize_produces_valid_wav(self, tmp_path):
        t = _transcription(["Hello everyone, welcome."], language="en")
        source = _make_source_wav(tmp_path, duration_s=8.0)
        output = tmp_path / "dubbed.wav"
        result = synthesize(t, source, output)
        assert result == output
        assert output.exists()
        data, sr = sf.read(str(output))
        assert sr == 24000
        assert len(data) > 0
        assert not np.allclose(data, 0.0)
