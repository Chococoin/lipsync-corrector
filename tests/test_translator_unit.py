from unittest.mock import MagicMock

import pytest

from core.transcription.models import Segment, Transcription, Word
from core.translation import translate


def _build_source_transcription(segment_texts, language="es"):
    """Build a Transcription with deterministic timestamps for testing."""
    segments = tuple(
        Segment(
            text=text,
            start=float(i * 2),
            end=float(i * 2 + 1.8),
            words=(
                Word(text=w, start=float(i * 2), end=float(i * 2 + 0.3),
                     probability=0.9)
                for w in text.split()
            ) if False else (),  # empty words in source for simplicity
            avg_logprob=-0.3,
            no_speech_prob=0.01,
        )
        for i, text in enumerate(segment_texts)
    )
    return Transcription(
        language=language,
        segments=segments,
        duration=float(len(segment_texts) * 2),
        model_size="medium",
    )


def _make_mock_client(tool_input: dict):
    """Mock client whose messages.create returns a message with a
    submit_translation tool_use block containing tool_input."""
    client = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "submit_translation"
    tool_block.input = tool_input
    message = MagicMock()
    message.content = [tool_block]
    client.messages.create.return_value = message
    return client


def _make_mock_client_no_tool_use():
    """Mock client that returns a text-only response (no tool_use block)."""
    client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Sorry, I cannot translate this."
    message = MagicMock()
    message.content = [text_block]
    client.messages.create.return_value = message
    return client


class TestInputValidation:
    def test_empty_target_language_raises_valueerror(self):
        t = _build_source_transcription(["hola"])
        client = _make_mock_client({"segments": []})
        with pytest.raises(ValueError, match="target_language"):
            translate(t, target_language="", client=client)

    def test_same_language_raises_valueerror(self):
        t = _build_source_transcription(["hola"], language="es")
        client = _make_mock_client({"segments": []})
        with pytest.raises(ValueError, match="same as source"):
            translate(t, target_language="es", client=client)

    def test_empty_segments_returns_empty_transcription_without_api_call(self):
        t = _build_source_transcription([])
        client = _make_mock_client({"segments": []})
        result = translate(t, target_language="en", client=client)
        assert result.language == "en"
        assert result.segments == ()
        assert result.duration == t.duration
        assert result.model_size == t.model_size
        assert client.messages.create.call_count == 0


class TestHappyPath:
    def test_basic_translation_builds_correct_transcription(self):
        t = _build_source_transcription(["Hola a todos.", "Bienvenidos."])
        client = _make_mock_client({
            "segments": [
                {"id": 0, "text": "Hi everyone."},
                {"id": 1, "text": "Welcome."},
            ]
        })
        result = translate(t, target_language="en", client=client)
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hi everyone."
        assert result.segments[1].text == "Welcome."

    def test_preserves_segment_timestamps(self):
        t = _build_source_transcription(["Hola.", "Chau."])
        client = _make_mock_client({
            "segments": [
                {"id": 0, "text": "Hi."},
                {"id": 1, "text": "Bye."},
            ]
        })
        result = translate(t, target_language="en", client=client)
        for orig, trans in zip(t.segments, result.segments):
            assert trans.start == orig.start
            assert trans.end == orig.end

    def test_preserves_avg_logprob_and_no_speech_prob(self):
        t = _build_source_transcription(["Hola."])
        client = _make_mock_client({
            "segments": [{"id": 0, "text": "Hi."}]
        })
        result = translate(t, target_language="en", client=client)
        assert result.segments[0].avg_logprob == t.segments[0].avg_logprob
        assert result.segments[0].no_speech_prob == t.segments[0].no_speech_prob

    def test_translated_segments_have_empty_words_tuple(self):
        t = _build_source_transcription(["Hola."])
        client = _make_mock_client({
            "segments": [{"id": 0, "text": "Hi."}]
        })
        result = translate(t, target_language="en", client=client)
        assert result.segments[0].words == ()

    def test_language_field_set_to_target(self):
        t = _build_source_transcription(["Hola."])
        client = _make_mock_client({
            "segments": [{"id": 0, "text": "Hi."}]
        })
        result = translate(t, target_language="fr", client=client)
        assert result.language == "fr"

    def test_duration_preserved(self):
        t = _build_source_transcription(["Hola.", "Chau."])
        client = _make_mock_client({
            "segments": [
                {"id": 0, "text": "Hi."},
                {"id": 1, "text": "Bye."},
            ]
        })
        result = translate(t, target_language="en", client=client)
        assert result.duration == t.duration

    def test_model_size_preserved(self):
        t = _build_source_transcription(["Hola."])
        client = _make_mock_client({
            "segments": [{"id": 0, "text": "Hi."}]
        })
        result = translate(t, target_language="en", client=client)
        assert result.model_size == t.model_size


class TestClientInvocation:
    def test_passes_model_to_client(self):
        t = _build_source_transcription(["Hola."])
        client = _make_mock_client({
            "segments": [{"id": 0, "text": "Hi."}]
        })
        translate(t, target_language="en", model="claude-sonnet-4-6", client=client)
        _, kwargs = client.messages.create.call_args
        assert kwargs["model"] == "claude-sonnet-4-6"

    def test_calls_with_tool_choice_forced(self):
        t = _build_source_transcription(["Hola."])
        client = _make_mock_client({
            "segments": [{"id": 0, "text": "Hi."}]
        })
        translate(t, target_language="en", client=client)
        _, kwargs = client.messages.create.call_args
        assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_translation"}

    def test_passes_tools_list_with_submit_translation(self):
        t = _build_source_transcription(["Hola."])
        client = _make_mock_client({
            "segments": [{"id": 0, "text": "Hi."}]
        })
        translate(t, target_language="en", client=client)
        _, kwargs = client.messages.create.call_args
        assert len(kwargs["tools"]) == 1
        assert kwargs["tools"][0]["name"] == "submit_translation"

    def test_user_message_includes_target_word_count(self):
        t = _build_source_transcription(["Hola a todos."])
        client = _make_mock_client({
            "segments": [{"id": 0, "text": "Hi everyone."}]
        })
        translate(t, target_language="en", client=client)
        _, kwargs = client.messages.create.call_args
        user_msg = kwargs["messages"][0]["content"]
        assert "(target ~" in user_msg
        assert "words)" in user_msg


class TestErrorPaths:
    def test_missing_tool_use_raises_valueerror(self):
        t = _build_source_transcription(["Hola."])
        client = _make_mock_client_no_tool_use()
        with pytest.raises(ValueError, match="did not call submit_translation"):
            translate(t, target_language="en", client=client)

    def test_segment_count_mismatch_raises_valueerror(self):
        t = _build_source_transcription(["Hola.", "Chau."])
        client = _make_mock_client({
            "segments": [{"id": 0, "text": "Hi."}]
        })
        with pytest.raises(ValueError, match="Expected 2"):
            translate(t, target_language="en", client=client)

    def test_missing_segment_ids_raises_valueerror(self):
        t = _build_source_transcription(["Hola.", "Chau."])
        client = _make_mock_client({
            "segments": [
                {"id": 0, "text": "Hi."},
                {"id": 5, "text": "Bye."},
            ]
        })
        with pytest.raises(ValueError, match="id mismatch"):
            translate(t, target_language="en", client=client)

    def test_missing_segments_field_raises_valueerror(self):
        t = _build_source_transcription(["Hola."])
        client = _make_mock_client({})  # no "segments" key at all
        with pytest.raises(ValueError, match="missing 'segments'"):
            translate(t, target_language="en", client=client)
