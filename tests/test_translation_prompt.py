from core.translation.prompt import build_system_prompt, build_tool_schema


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
