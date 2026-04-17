"""System prompt and tool schema for the translation API call."""

from __future__ import annotations

from core.transcription.models import Segment

WORDS_PER_SECOND: dict[str, float] = {
    "es": 2.5,
    "en": 2.0,
    "fr": 2.5,
    "pt": 2.3,
    "de": 1.8,
    "it": 2.3,
    "ja": 3.2,
    "zh": 2.8,
    "ko": 2.5,
}
DEFAULT_WORDS_PER_SECOND = 2.0

SYSTEM_PROMPT_TEMPLATE = """\
You are a professional subtitle translator specializing in conversational
content (YouTube videos, interviews, testimonials). Translate from {source}
to {target}.

Guidelines:
- Preserve the speaker's tone and register (casual stays casual, formal stays formal).
- Prefer natural phrasing in the target language over literal word-by-word translation.
- Each segment includes a target word count. This is a HARD constraint — the
  translated audio must fit in the original segment's time slot. Simplify the
  message, drop filler words, or rephrase concisely to match the target count.
  Never exceed it by more than 1 word.
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
