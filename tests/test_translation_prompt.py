from core.transcription.models import Segment
from core.translation.prompt import (
    WORDS_PER_SECOND,
    build_system_prompt,
    build_tool_schema,
    target_word_count,
)


class TestBuildSystemPrompt:
    def test_contains_source_and_target_language(self):
        prompt = build_system_prompt("es", "en")
        assert "es" in prompt
        assert "en" in prompt

    def test_mentions_submit_translation_tool(self):
        prompt = build_system_prompt("es", "en")
        assert "submit_translation" in prompt

    def test_mentions_conversational_content(self):
        prompt = build_system_prompt("es", "en")
        assert "conversational" in prompt.lower()

    def test_mentions_target_word_count(self):
        prompt = build_system_prompt("es", "en")
        assert "target word count" in prompt.lower()


class TestBuildToolSchema:
    def test_name_is_submit_translation(self):
        schema = build_tool_schema()
        assert schema["name"] == "submit_translation"

    def test_has_description(self):
        schema = build_tool_schema()
        assert "description" in schema
        assert len(schema["description"]) > 0

    def test_input_schema_requires_segments(self):
        schema = build_tool_schema()
        input_schema = schema["input_schema"]
        assert input_schema["type"] == "object"
        assert "segments" in input_schema["properties"]
        assert "segments" in input_schema["required"]

    def test_segments_items_require_id_and_text(self):
        schema = build_tool_schema()
        item = schema["input_schema"]["properties"]["segments"]["items"]
        assert item["type"] == "object"
        assert set(item["required"]) == {"id", "text"}
        assert item["properties"]["id"]["type"] == "integer"
        assert item["properties"]["text"]["type"] == "string"


class TestTargetWordCount:
    def _seg(self, start, end):
        return Segment(
            text="placeholder", start=start, end=end,
            words=(), avg_logprob=-0.3, no_speech_prob=0.01,
        )

    def test_english_2_8_seconds(self):
        # 2.8s × 2.5 wps = 7
        assert target_word_count(self._seg(0.0, 2.8), "en") == 7

    def test_spanish_2_0_seconds(self):
        # 2.0s × 3.0 wps = 6
        assert target_word_count(self._seg(0.0, 2.0), "es") == 6

    def test_german_3_0_seconds(self):
        # 3.0s × 2.3 wps = 6.9 → 7
        assert target_word_count(self._seg(1.0, 4.0), "de") == 7

    def test_unknown_language_uses_default(self):
        # 2.0s × 2.5 (default) = 5
        assert target_word_count(self._seg(0.0, 2.0), "xx") == 5

    def test_very_short_segment_returns_at_least_one(self):
        assert target_word_count(self._seg(0.0, 0.1), "en") >= 1
