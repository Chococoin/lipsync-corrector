from dataclasses import FrozenInstanceError

import pytest

from core.transcription.models import Segment, Transcription, Word


def _word(text="hola", start=0.0, end=0.5, probability=0.95):
    return Word(text=text, start=start, end=end, probability=probability)


def _segment(
    text="hola mundo",
    start=0.0,
    end=1.0,
    words=None,
    avg_logprob=-0.3,
    no_speech_prob=0.01,
):
    if words is None:
        words = (_word(),)
    return Segment(
        text=text,
        start=start,
        end=end,
        words=words,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


class TestWord:
    def test_construction(self):
        w = _word()
        assert w.text == "hola"
        assert w.start == 0.0
        assert w.end == 0.5
        assert w.probability == 0.95

    def test_is_frozen(self):
        w = _word()
        with pytest.raises(FrozenInstanceError):
            w.text = "otra"  # type: ignore[misc]

    def test_is_hashable(self):
        w = _word()
        hash(w)


class TestSegment:
    def test_construction(self):
        s = _segment()
        assert s.text == "hola mundo"
        assert len(s.words) == 1
        assert s.avg_logprob == -0.3

    def test_is_frozen(self):
        s = _segment()
        with pytest.raises(FrozenInstanceError):
            s.text = "otro"  # type: ignore[misc]

    def test_is_hashable(self):
        s = _segment()
        hash(s)

    def test_words_is_tuple(self):
        s = _segment()
        assert isinstance(s.words, tuple)

    def test_empty_words_allowed(self):
        s = _segment(words=())
        assert s.words == ()


class TestTranscription:
    def test_construction(self):
        t = Transcription(
            language="es",
            segments=(_segment(),),
            duration=1.0,
            model_size="medium",
        )
        assert t.language == "es"
        assert len(t.segments) == 1
        assert t.duration == 1.0
        assert t.model_size == "medium"

    def test_is_frozen(self):
        t = Transcription(
            language="es", segments=(), duration=0.0, model_size="medium"
        )
        with pytest.raises(FrozenInstanceError):
            t.language = "en"  # type: ignore[misc]

    def test_is_hashable(self):
        t = Transcription(
            language="es",
            segments=(_segment(),),
            duration=1.0,
            model_size="medium",
        )
        hash(t)

    def test_segments_is_tuple(self):
        t = Transcription(
            language="es", segments=(), duration=0.0, model_size="medium"
        )
        assert isinstance(t.segments, tuple)

    def test_empty_segments_allowed(self):
        t = Transcription(
            language="es", segments=(), duration=0.0, model_size="medium"
        )
        assert t.segments == ()
