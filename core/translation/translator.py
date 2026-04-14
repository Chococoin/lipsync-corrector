"""The public `translate()` function wrapping the Anthropic API."""

from __future__ import annotations

from typing import Optional

import anthropic

from core.transcription.models import Segment, Transcription
from core.translation.prompt import build_system_prompt, build_tool_schema

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4096


def translate(
    transcription: Transcription,
    target_language: str,
    *,
    model: str = DEFAULT_MODEL,
    client: Optional[anthropic.Anthropic] = None,
) -> Transcription:
    """Translate a Transcription into target_language via Claude API.

    Args:
        transcription: Source Transcription with segments to translate.
        target_language: ISO 639-1 code of the target language ("en", "fr", ...).
        model: Anthropic model ID. Default is Haiku 4.5.
        client: Optional Anthropic client for dependency injection.
            None means instantiate a default client which reads
            ANTHROPIC_API_KEY from the environment.

    Returns:
        A new Transcription with target_language as `language`, translated
        segment text, empty `words` tuples, and all segment-level
        timestamps and probabilities preserved from the source.

    Raises:
        ValueError: on invalid input, missing tool_use in the response,
            or segment count / id mismatch.
        anthropic.AnthropicError subclasses: any error from the SDK
            propagates unchanged.
    """
    if not target_language:
        raise ValueError("target_language cannot be empty")
    if target_language == transcription.language:
        raise ValueError(
            f"target_language ({target_language!r}) is the same as "
            f"source language ({transcription.language!r})"
        )
    if not transcription.segments:
        return Transcription(
            language=target_language,
            segments=(),
            duration=transcription.duration,
            model_size=transcription.model_size,
        )

    if client is None:
        client = anthropic.Anthropic()

    system_prompt = build_system_prompt(
        transcription.language, target_language
    )
    tool_schema = build_tool_schema()
    user_content = "Translate these segments:\n\n" + "\n".join(
        f"{i}: {seg.text}"
        for i, seg in enumerate(transcription.segments)
    )

    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": "submit_translation"},
        messages=[{"role": "user", "content": user_content}],
    )

    return _adapt_response(response, transcription, target_language)


def _adapt_response(
    response,
    source: Transcription,
    target_language: str,
) -> Transcription:
    """Locate the submit_translation tool_use and build a new Transcription."""
    tool_use = next(
        (
            block for block in response.content
            if getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == "submit_translation"
        ),
        None,
    )
    if tool_use is None:
        content_types = [getattr(b, "type", "?") for b in response.content]
        raise ValueError(
            f"Claude did not call submit_translation. "
            f"Response content types: {content_types}"
        )

    tool_input = tool_use.input
    translated = tool_input.get("segments") if isinstance(tool_input, dict) else None
    if translated is None:
        raise ValueError("submit_translation response missing 'segments'")

    if len(translated) != len(source.segments):
        raise ValueError(
            f"Expected {len(source.segments)} segments, got {len(translated)}"
        )

    by_id = {item["id"]: item["text"] for item in translated}
    expected_ids = set(range(len(source.segments)))
    got_ids = set(by_id.keys())
    if got_ids != expected_ids:
        missing = sorted(expected_ids - got_ids)
        extra = sorted(got_ids - expected_ids)
        raise ValueError(
            f"Segment id mismatch. Missing: {missing}, Extra: {extra}"
        )

    new_segments = tuple(
        Segment(
            text=by_id[i],
            start=src.start,
            end=src.end,
            words=(),
            avg_logprob=src.avg_logprob,
            no_speech_prob=src.no_speech_prob,
        )
        for i, src in enumerate(source.segments)
    )

    return Transcription(
        language=target_language,
        segments=new_segments,
        duration=source.duration,
        model_size=source.model_size,
    )
