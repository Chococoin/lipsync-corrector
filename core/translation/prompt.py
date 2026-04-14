"""System prompt and tool schema for the translation API call."""

from __future__ import annotations

SYSTEM_PROMPT_TEMPLATE = """\
You are a professional subtitle translator specializing in conversational
content (YouTube videos, interviews, testimonials). Translate from {source}
to {target}.

Guidelines:
- Preserve the speaker's tone and register (casual stays casual, formal stays formal).
- Prefer natural phrasing in the target language over literal word-by-word translation.
- Keep each segment roughly the same length as the original when reasonable
  (same-length translations work better for dubbing).
- Do not add commentary, disclaimers, or explanations.
- If a segment is already in the target language, return it unchanged.
- Return the translation via the `submit_translation` tool.
"""


def build_system_prompt(source_language: str, target_language: str) -> str:
    """Build the system prompt for the translation call."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        source=source_language,
        target=target_language,
    )


def build_tool_schema() -> dict:
    """Build the tool schema that Claude must use to return the translation."""
    return {
        "name": "submit_translation",
        "description": (
            "Submit the translated segments. The output must have exactly "
            "the same number of items as the input, in the same order, "
            "identified by the same integer ids."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "description": "Segment id matching the input",
                            },
                            "text": {
                                "type": "string",
                                "description": "Translated text for this segment",
                            },
                        },
                        "required": ["id", "text"],
                    },
                },
            },
            "required": ["segments"],
        },
    }
