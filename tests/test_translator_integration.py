import os

import pytest

from core.transcription.models import Segment, Transcription, Word
from core.translation import translate

requires_api_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@requires_api_key
class TestRealApi:
    def test_translate_simple_spanish_to_english(self):
        t = Transcription(
            language="es",
            segments=(
                Segment(
                    text="Hola a todos.",
                    start=0.0,
                    end=1.0,
                    words=(
                        Word(text="Hola", start=0.0, end=0.4, probability=0.9),
                    ),
                    avg_logprob=-0.2,
                    no_speech_prob=0.01,
                ),
                Segment(
                    text="Bienvenidos a Cartagena.",
                    start=1.0,
                    end=2.5,
                    words=(),
                    avg_logprob=-0.3,
                    no_speech_prob=0.02,
                ),
            ),
            duration=2.5,
            model_size="medium",
        )
        result = translate(t, target_language="en")

        # Structural assertions (deterministic):
        assert result.language == "en"
        assert len(result.segments) == 2
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 1.0
        assert result.segments[0].words == ()
        assert result.segments[1].words == ()
        assert result.duration == 2.5
        assert result.model_size == "medium"

        # Weak content assertions (LLM output is non-deterministic):
        lower_text = " ".join(s.text.lower() for s in result.segments)
        assert "welcome" in lower_text or "cartagena" in lower_text
