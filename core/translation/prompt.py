"""System prompt and tool schema for the translation API call."""

from __future__ import annotations

from core.transcription.models import Segment

WORDS_PER_SECOND: dict[str, float] = {
    "es": 3.0,
    "en": 2.5,
    "fr": 3.0,
    "pt": 2.8,
    "de": 2.3,
    "it": 2.8,
    "ja": 4.0,
    "zh": 3.5,
    "ko": 3.0,
}
DEFAULT_WORDS_PER_SECOND = 2.5

SYSTEM_PROMPT_TEMPLATE = """\
You are a professional subtitle translator specializing in conversational
content (YouTube videos, interviews, testimonials). Translate from {source}
to {target}.

Guidelines:
- Preserve the speaker's tone and register (casual stays casual, formal stays formal).
- Prefer natural phrasing in the target language over literal word-by-word translation.
- Each segment includes a target word count. Aim to match it — it represents
  the natural speaking duration for the target language. Prioritize meaning
  over exact count, but stay within ±2 words when possible.
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


def target_word_count(segment: Segment, target_language: str) -> int:
    """Estimate how many words the translation should have to match the
    original segment's speaking duration in the target language."""
    duration = segment.end - segment.start
    wps = WORDS_PER_SECOND.get(target_language, DEFAULT_WORDS_PER_SECOND)
    return max(1, round(duration * wps))


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
