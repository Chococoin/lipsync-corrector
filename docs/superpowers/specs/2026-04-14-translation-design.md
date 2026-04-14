# Translation Sub-project — Design Spec

**Date:** 2026-04-14
**Status:** Approved for implementation
**Author:** chocos (with Claude)
**Predecessors:**
- `2026-04-11-lipsync-corrector-design.md` (parent project)
- `2026-04-14-stt-transcription-design.md` (previous sub-project; produces the `Transcription` type this sub-project consumes)

## 1. Purpose

Add translation capability to the lipsync-corrector project as the second step of the three-part auto-dubbing workflow (STT → translation → TTS). Given a `Transcription` in a source language, produce a new `Transcription` in a target language preserving segment-level timestamps so that the downstream TTS sub-project can regenerate audio that fits the original video's timing.

This sub-project explicitly expands the parent project's non-goal of "We do not translate text" that was already lifted in the STT sub-project spec.

## 2. Scope

**In scope:**

- A `core/translation/` Python subpackage exposing one public function `translate()`.
- A private helper module `prompt.py` containing the system prompt template and tool schema used to call Claude.
- A dependency on the official `anthropic` Python SDK.
- Unit tests that mock the `anthropic.Anthropic` client (always run, no network).
- An integration test that requires `ANTHROPIC_API_KEY` in the environment and hits the real API (marked `skipif` when the key is absent).
- A standalone `examples/translate_demo.py` script that loads a transcription JSON produced by `transcribe_demo.py`, translates it, and writes a new JSON.
- README section documenting the demo and the API key requirement.

**Explicitly out of scope** (deferred):

- Text-to-speech regeneration (next sub-project).
- Duration reconciliation between translated audio and original video timing.
- A `translate` subcommand in `cli/main.py`.
- Batch translation to multiple target languages in one call.
- Retry policies, rate-limit handling, or cost tracking beyond what the SDK provides by default.
- Caching of translation results.
- A `read_json` deserializer in `core/transcription/serializers.py` (the demo script includes its own local helper; if more consumers need it later, promote it then).
- Alternative providers (DeepL, NLLB, etc.) — Claude API is the only backend in this sub-project.

## 3. Constraints and Decisions

All technical decisions below were made during brainstorming and are recorded here for traceability.

1. **Provider:** Claude API via the official `anthropic` Python SDK. Chosen over DeepL and local NLLB for quality on conversational content, low cost per request, and flexibility of prompt engineering.
2. **Default model:** `claude-haiku-4-5-20251001`. Configurable via the `model` keyword argument. Haiku 4.5 is the cheapest model in the Claude 4.5/4.6 family and its translation quality is more than sufficient for conversational content.
3. **Granularity:** one API call per `Transcription`, with the full list of segments sent in a structured form (integer id + text). Tool calling (`tool_choice` forced) guarantees the response shape. This preserves cross-segment context (pronouns, references, tone) while keeping the 1:1 mapping between source and target segments.
4. **Word-level handling in translated output:** each translated `Segment` has `words=()` (empty tuple). We do not interpolate synthetic word timestamps. Reasoning: Wav2Lip never reads word-level timestamps (it reads the audio waveform directly), and the downstream TTS and eventual time-aligned dubbing sub-project will need real timestamps derived from the synthesized audio, not fabricated ones from linear interpolation.
5. **API shape:** pure function (`translate(transcription, target_language, *, model=..., client=None)`), not a class. Translation is stateless and per-call; no large model weights to amortize. Client injection via `client=None` default enables mocking in tests.
6. **Testing strategy:** hybrid. Unit tests mock the Anthropic client and always run. One integration test hits the real API and is marked `skipif` when `ANTHROPIC_API_KEY` is unset. This matches the pattern used in the STT sub-project.
7. **Output type:** reuse the existing `core.transcription.models.Transcription` type. The translated output has `language=target_language`, translated segment text, empty word tuples, preserved segment timestamps and preserved `avg_logprob` / `no_speech_prob` (they describe the source speech signal, not the translation), and preserved `duration` and `model_size` from the input.
8. **Integration with existing CLI:** none. No changes to `cli/main.py`. A unified `dub` command will be designed when all three sub-projects (STT, translation, TTS) are complete.
9. **API key management:** read from the `ANTHROPIC_API_KEY` environment variable via the default behavior of `anthropic.Anthropic()`. No project-specific `.env` parsing; users manage their env via their shell, `direnv`, or `uv run --env-file`.
10. **Error strategy:** SDK exceptions propagate unchanged. Our own `ValueError` is raised only for input validation and response shape mismatches that cannot be interpreted as anything else.

## 4. File Structure

```
lipsync-corrector/
├── core/
│   └── translation/
│       ├── __init__.py            # re-exports `translate`
│       ├── translator.py          # the public function + private _adapt helpers
│       └── prompt.py              # build_system_prompt, build_tool_schema
├── tests/
│   ├── test_translator_unit.py          # mocks the client; always runs
│   └── test_translator_integration.py   # real API; skipif no key
├── examples/
│   └── translate_demo.py          # loads JSON → translate → writes JSON
├── pyproject.toml                  # + anthropic dependency
└── README.md                       # new demo section
```

**Responsibilities:**

- `translator.py` — one public function `translate()`. Internals handle input validation, tool-choice-forced API call construction, response parsing, and building the new `Transcription`. A private `_adapt_response(tool_use_input, original) -> Transcription` helper converts the tool response dict back into our dataclass.
- `prompt.py` — two pure functions (`build_system_prompt`, `build_tool_schema`) with no runtime dependencies on the SDK. Easy to test in isolation. Centralizes prompt content so iteration on wording does not touch HTTP code.
- `translate_demo.py` — standalone script that contains a local `_load_transcription` helper for JSON deserialization (since the STT sub-project only exports serializers, not deserializers).

## 5. Public API

In `core/translation/__init__.py`:

```python
from core.translation.translator import translate

__all__ = ["translate"]
```

In `core/translation/translator.py`:

```python
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def translate(
    transcription: Transcription,
    target_language: str,
    *,
    model: str = DEFAULT_MODEL,
    client: Optional[anthropic.Anthropic] = None,
) -> Transcription:
    ...
```

**Semantics:**

1. If `target_language` is empty, raise `ValueError`.
2. If `target_language == transcription.language`, raise `ValueError` (caller bug; fail fast).
3. If `transcription.segments` is empty, return a new `Transcription` with `language=target_language`, empty segments, and preserved `duration` and `model_size`. **No API call is made.**
4. Instantiate `anthropic.Anthropic()` if `client is None` (reads `ANTHROPIC_API_KEY` from env).
5. Build the system prompt via `build_system_prompt(transcription.language, target_language)`.
6. Build the tool schema via `build_tool_schema()`.
7. Build the user message as a numbered list of segment texts, one per line, zero-indexed.
8. Call `client.messages.create(model=model, max_tokens=4096, system=..., tools=[schema], tool_choice={"type": "tool", "name": "submit_translation"}, messages=[...])`. `max_tokens=4096` is a safe upper bound for conversational transcriptions of up to ~30 segments; tool responses for typical 5-20 segment videos use a fraction of that.
9. Locate the `tool_use` block in `response.content`. If absent, raise `ValueError`.
10. Extract `tool_use.input["segments"]`. Validate: must be a list, same length as `transcription.segments`, with ids `{0..N-1}` exactly.
11. Build new `Segment`s reusing `start`, `end`, `avg_logprob`, `no_speech_prob` from the corresponding source segment, with the translated `text` and `words=()`.
12. Return a new `Transcription(language=target_language, segments=..., duration=..., model_size=...)`.

## 6. Prompt and Tool Schema

In `core/translation/prompt.py`:

```python
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
    return SYSTEM_PROMPT_TEMPLATE.format(
        source=source_language, target=target_language
    )


def build_tool_schema() -> dict:
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
                            "id": {"type": "integer"},
                            "text": {"type": "string"},
                        },
                        "required": ["id", "text"],
                    },
                },
            },
            "required": ["segments"],
        },
    }
```

**User message construction** (lives in `translator.py`, not `prompt.py`, because it needs the runtime segment list):

```python
user_content = "Translate these segments:\n\n" + "\n".join(
    f"{i}: {seg.text}"
    for i, seg in enumerate(transcription.segments)
)
```

The tool call is forced via `tool_choice={"type": "tool", "name": "submit_translation"}`, eliminating the risk that Claude responds with free-form text instead of the structured tool call.

## 7. Error Handling

| Case | Behavior |
|---|---|
| `target_language` empty | `ValueError("target_language cannot be empty")` |
| `target_language == source` | `ValueError(f"target_language is the same as source language")` |
| `transcription.segments` empty | Return `Transcription(language=target_language, segments=(), ...)`; no API call |
| `ANTHROPIC_API_KEY` missing | `anthropic.AuthenticationError` from the SDK, propagates |
| Network error | `anthropic.APIConnectionError`, propagates |
| Rate limit | `anthropic.RateLimitError`, propagates (SDK's built-in retry already tried) |
| Server error | `anthropic.APIStatusError`, propagates |
| Response has no `tool_use` block | `ValueError(f"Claude did not call submit_translation. Content types: ...")` |
| `segments` field missing in tool input | `ValueError("submit_translation response missing 'segments'")` |
| Segment count mismatch | `ValueError(f"Expected N, got M")` |
| Segment id set mismatch | `ValueError(f"Segment id mismatch. Missing: [...], Extra: [...]")` |

**No** retry, no fallback to another model, no caching, no custom exception hierarchy, no logging layer.

## 8. Testing Strategy

### `tests/test_translator_unit.py` — always run, no network

Uses `unittest.mock.MagicMock` to replace `anthropic.Anthropic()`. A helper `make_mock_client(tool_input)` builds a mock that returns a response with a `submit_translation` tool_use containing the given input, matching the shape of `anthropic.types.Message` closely enough for our code.

**Test classes and expected cases (~16 tests):**

**`TestInputValidation`:**
- `test_empty_target_language_raises_valueerror`
- `test_same_language_raises_valueerror`
- `test_empty_segments_returns_empty_transcription_without_api_call` (verifies `client.messages.create.call_count == 0`)

**`TestHappyPath`:**
- `test_basic_translation_builds_correct_transcription`
- `test_preserves_segment_timestamps`
- `test_preserves_avg_logprob_and_no_speech_prob`
- `test_translated_segments_have_empty_words_tuple`
- `test_language_field_set_to_target`
- `test_duration_preserved`
- `test_model_size_preserved`

**`TestClientInvocation`:**
- `test_passes_model_to_client` — verifies the `model` kwarg received by the mock matches what the caller passed.
- `test_calls_with_tool_choice_forced` — verifies `tool_choice={"type": "tool", "name": "submit_translation"}`.

**`TestErrorPaths`:**
- `test_missing_tool_use_raises_valueerror`
- `test_segment_count_mismatch_raises_valueerror`
- `test_missing_segment_ids_raises_valueerror`

**`TestPrompt`** (testing `prompt.py` directly):
- `test_build_system_prompt_contains_source_and_target`
- `test_build_tool_schema_has_required_fields`
- `test_build_tool_schema_segments_shape`

### `tests/test_translator_integration.py` — skipif no API key

```python
import os
import pytest

requires_api_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
```

**One test (`TestRealApi.test_translate_simple_spanish_to_english`):** builds a tiny hand-crafted `Transcription` with 2 segments in Spanish, calls `translate(t, "en")` against the real API, verifies:
- `result.language == "en"`
- `len(result.segments) == 2`
- One weak content assertion (e.g. `"welcome" in result.segments[1].text.lower()`)
- Structural assertions (timestamps preserved, `words == ()`)

The content assertion is weak because LLM output is non-deterministic.

### Expected test counts

- Unit: ~16
- Integration: 1
- **Total new: ~17 tests**
- Suite: 151 → ~168 with API key set, 167 without.

## 9. Demo Script

`examples/translate_demo.py`, approximately 50 lines:

```python
"""Demo: translate a transcription JSON into another language.

Usage:
    uv run python examples/translate_demo.py <input_json> <target_language> [<output_json>]
"""

# ... sys.path fix ...
# ... _load_transcription helper ...

def main(argv):
    # ... parse args ...
    source = _load_transcription(input_json)
    print(f"Loaded {len(source.segments)} segments in {source.language}")
    result = translate(source, target_language=target_language)
    print(f"Translated to {result.language}")
    write_json(result, output_json)
    print(f"Wrote {output_json}")
    return 0
```

The local `_load_transcription(path)` reads the JSON produced by the STT demo and reconstructs a `Transcription` dataclass. It is the only deserialization code in this sub-project; it lives in the demo (not in `core/transcription/serializers.py`) to keep the STT sub-project scope closed.

Default output filename is `<input_stem>.<target_language>.json` next to the input.

## 10. Dependencies

One new Python dependency: `anthropic>=0.40,<1.0`. Pure Python, ~1-2 MB added to `.venv`.

## 11. What Success Looks Like

After implementation:

- `export ANTHROPIC_API_KEY=... && uv run python examples/transcribe_demo.py ~/Downloads/Video_De_Mujer_Saludando_Generado.mp4` produces `veo_stt.json` (Spanish transcription).
- `uv run python examples/translate_demo.py examples/veo_stt.json en` produces `examples/veo_stt.en.json` with three translated segments preserving the original timestamps.
- All unit tests pass without network access. The integration test passes when `ANTHROPIC_API_KEY` is set and skips cleanly when it isn't.
- Suite count grows from 151 to ~168.
- No existing tests regress. No changes to files outside `core/translation/`, `tests/test_translator_*.py`, `examples/translate_demo.py`, `pyproject.toml`, `README.md`.

## 12. Relationship to the Parent Design Spec and STT Sub-project

The parent project spec (`2026-04-11-lipsync-corrector-design.md`, section 1) lists as an explicit non-goal: *"We do not translate text."* The STT sub-project spec already lifted that non-goal for the auto-dub workflow; this sub-project is the direct continuation.

The `Transcription` dataclass type created in the STT sub-project is the shared currency between sub-projects. This translation sub-project both consumes (input) and produces (output) instances of it. No new types are introduced.

The parent project's other non-goals (no voice cloning, no face identity replacement, no cinema-grade VFX) remain in effect.

## 13. Open Questions (Deferred)

- **Which TTS provider** for sub-project 3 (ElevenLabs vs Coqui XTTS-v2 vs OpenAI TTS). Not blocking for this sub-project. Will be chosen based on what can accept segment-level timing hints and/or produce word-level timestamps from synthesis.
- **Duration reconciliation strategy** when the translated TTS audio differs from the source segment duration. Not blocking. Candidates: audio time-stretching (rubber band), frame-level video adjustment, or generation-time constraints in the TTS.
- **When to introduce a `dub` CLI subcommand** unifying STT → translate → TTS → Wav2Lip into one command. Scheduled for after sub-project 3 lands.
- **Multi-target-language workflows** (e.g. dub one video into 5 languages at once). Trivially implemented at the caller level; not a module concern.
