from pathlib import Path

import pytest

from core.transcription.models import Transcription
from core.transcription.transcriber import transcribe

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
WEIGHTS_CACHED = (HF_CACHE / "models--mlx-community--whisper-medium-mlx").exists()
requires_weights = pytest.mark.skipif(
    not WEIGHTS_CACHED,
    reason="mlx-community/whisper-medium-mlx not in HF cache",
)

VEO_AUDIO = Path(__file__).parent.parent / "examples" / "veo_audio_16k_mono.wav"
requires_audio = pytest.mark.skipif(
    not VEO_AUDIO.exists(),
    reason="examples/veo_audio_16k_mono.wav not present",
)


class TestUnitValidation:
    def test_missing_audio_raises_filenotfound(self, tmp_path):
        missing = tmp_path / "not_there.wav"
        with pytest.raises(FileNotFoundError) as exc:
            transcribe(missing)
        assert "not_there.wav" in str(exc.value)

    def test_invalid_model_size_raises_valueerror(self, tmp_path):
        fake = tmp_path / "fake.wav"
        fake.write_bytes(b"RIFF0000")
        with pytest.raises(ValueError) as exc:
            transcribe(fake, model_size="notreal")
        assert "notreal" in str(exc.value)


@requires_weights
@requires_audio
class TestInference:
    def test_returns_transcription(self):
        result = transcribe(VEO_AUDIO)
        assert isinstance(result, Transcription)

    def test_detects_spanish(self):
        result = transcribe(VEO_AUDIO)
        assert result.language == "es"

    def test_has_at_least_one_segment(self):
        result = transcribe(VEO_AUDIO)
        assert len(result.segments) >= 1

    def test_duration_is_positive(self):
        result = transcribe(VEO_AUDIO)
        assert result.duration > 0.0

    def test_model_size_stored(self):
        result = transcribe(VEO_AUDIO)
        assert result.model_size == "medium"

    def test_words_are_monotonic_within_segment(self):
        result = transcribe(VEO_AUDIO)
        for seg in result.segments:
            for w in seg.words:
                assert w.start < w.end
            for a, b in zip(seg.words, seg.words[1:]):
                assert a.start <= b.start

    def test_forced_language_es_still_returns_es(self):
        result = transcribe(VEO_AUDIO, language="es")
        assert result.language == "es"
