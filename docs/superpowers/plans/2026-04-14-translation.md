# Translation Sub-project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Python-API-only translation module (`core/translation/`) built on the Anthropic SDK that takes a `Transcription` (in a source language) and returns a new `Transcription` (in a target language) while preserving segment-level timestamps, so the downstream TTS sub-project can regenerate audio with matching timing.

**Architecture:** A new `core/translation/` subpackage with two focused files — `prompt.py` (pure functions building the system prompt and tool schema for Claude) and `translator.py` (the public `translate()` function, input validation, Anthropic API call with tool choice forced, response parsing, and output `Transcription` construction). One public function. Unit tests mock the Anthropic client; a single integration test hits the real API when `ANTHROPIC_API_KEY` is set, and skips otherwise. A standalone `examples/translate_demo.py` script loads a transcription JSON produced by the STT sub-project's demo, translates it, and writes the result.

**Tech Stack:** Python 3.11, the official `anthropic` Python SDK (pure Python, ~1-2 MB). No new ML dependencies. Reuses the `Transcription`/`Segment`/`Word` dataclasses from `core/transcription/models.py`.

**Branch:** `translation` off `main` (currently at `094c67d` after the translation spec commit).

**Design spec:** `docs/superpowers/specs/2026-04-14-translation-design.md`.

---

## File Structure (end state of this sub-project)

```
lipsync-corrector/
├── core/
│   └── translation/
│       ├── __init__.py                # NEW: re-exports `translate`
│       ├── prompt.py                  # NEW: build_system_prompt, build_tool_schema
│       └── translator.py              # NEW: the public translate() function
├── tests/
│   ├── test_translator_unit.py        # NEW: mocks the Anthropic client
│   └── test_translator_integration.py # NEW: real API, skipif no key
├── examples/
│   └── translate_demo.py              # NEW: demo script
├── docs/milestones/
│   └── translation.md                 # NEW: written at end of task 4
├── pyproject.toml                      # MODIFIED: adds anthropic
└── README.md                           # MODIFIED: adds translation demo section
```

---

## Task 1: Add anthropic dependency

**Files:**
- Modify: `~/Projects/lipsync-corrector/pyproject.toml`

- [ ] **Step 1: Create the feature branch**

```bash
cd ~/Projects/lipsync-corrector
git checkout -b translation
```

- [ ] **Step 2: Edit `pyproject.toml`**

Replace the `dependencies = [...]` block with:

```toml
dependencies = [
    "insightface>=0.7.3",
    "onnxruntime>=1.17.0",
    "opencv-python>=4.9.0",
    "numpy>=1.26,<2.0",
    "torch>=2.1,<3.0",
    "librosa>=0.10,<0.11",
    "soundfile>=0.12",
    "mlx-whisper>=0.4,<1.0",
    "anthropic>=0.40,<1.0",
]
```

- [ ] **Step 3: Sync env**

```bash
cd ~/Projects/lipsync-corrector
uv sync
```

Expected: installs `anthropic` and its transitive deps (httpx, pydantic, etc.). Pure-Python additions, a few MB to `.venv`.

- [ ] **Step 4: Smoke test the anthropic import**

```bash
uv run python -c "
import anthropic
print('anthropic:', anthropic.__version__)
print('has Anthropic:', hasattr(anthropic, 'Anthropic'))
print('has APIError:', hasattr(anthropic, 'APIError'))
"
```

Expected: prints a version string, `has Anthropic: True`, `has APIError: True`.

- [ ] **Step 5: Run existing suite — verify no regressions**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 151 passed. The new dependency should not affect any existing test.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add pyproject.toml uv.lock
git commit -m "deps: add anthropic SDK for translation sub-project"
```

---

## Task 2: Prompt helpers (`core/translation/prompt.py`)

**Files:**
- Create: `~/Projects/lipsync-corrector/core/translation/__init__.py`
- Create: `~/Projects/lipsync-corrector/core/translation/prompt.py`
- Create: `~/Projects/lipsync-corrector/tests/test_translation_prompt.py`

**Background:** These are pure functions — no imports of `anthropic`, no side effects. They build the system prompt string and tool schema dict that the translator will pass to `client.messages.create(...)`. Keeping them in their own module means the prompt content can be iterated on without touching HTTP code, and both functions are trivially unit-testable.

- [ ] **Step 1: Write failing tests**

Create `tests/test_translation_prompt.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_translation_prompt.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.translation'`.

- [ ] **Step 3: Create `core/translation/__init__.py`**

```python
"""Translation subpackage.

Exposes a Python API: translate a Transcription into another language
via the Claude API, preserving segment-level timestamps.
"""

__all__: list[str] = []
```

Note: empty `__all__` for now. Task 3 will add `translate` to it.

- [ ] **Step 4: Create `core/translation/prompt.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_translation_prompt.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Run full suite**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 158 passed (151 existing + 7 new).

- [ ] **Step 7: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/translation/__init__.py core/translation/prompt.py tests/test_translation_prompt.py
git commit -m "feat: add translation prompt and tool schema helpers"
```

---

## Task 3: Translator function with unit tests

**Files:**
- Create: `~/Projects/lipsync-corrector/core/translation/translator.py`
- Modify: `~/Projects/lipsync-corrector/core/translation/__init__.py`
- Create: `~/Projects/lipsync-corrector/tests/test_translator_unit.py`

**Background:** This is the core of the sub-project. The `translate()` function validates its inputs, builds the Claude API request using `prompt.py`, calls `client.messages.create()` with `tool_choice` forced to `submit_translation`, parses the response, and builds a new `Transcription` with translated text and preserved segment metadata. All unit tests use `unittest.mock.MagicMock` to replace the Anthropic client — no network access at test time.

- [ ] **Step 1: Write failing tests**

Create `tests/test_translator_unit.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_translator_unit.py -v
```

Expected: `ImportError: cannot import name 'translate' from 'core.translation'`.

- [ ] **Step 3: Create `core/translation/translator.py`**

```python
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
```

- [ ] **Step 4: Extend `core/translation/__init__.py`**

Replace the current file with:

```python
"""Translation subpackage.

Exposes a Python API: translate a Transcription into another language
via the Claude API, preserving segment-level timestamps.
"""

from core.translation.translator import translate

__all__ = ["translate"]
```

- [ ] **Step 5: Run unit tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_translator_unit.py -v
```

Expected: 17 passed.

- [ ] **Step 6: Run full suite**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 175 passed (158 existing + 17 new).

- [ ] **Step 7: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/translation/translator.py core/translation/__init__.py tests/test_translator_unit.py
git commit -m "feat: add translate function wrapping Anthropic SDK"
```

---

## Task 4: Integration test + demo script + README + milestone notes

**Files:**
- Create: `~/Projects/lipsync-corrector/tests/test_translator_integration.py`
- Create: `~/Projects/lipsync-corrector/examples/translate_demo.py`
- Modify: `~/Projects/lipsync-corrector/README.md`
- Create: `~/Projects/lipsync-corrector/docs/milestones/translation.md`

**Background:** This task produces the end-to-end observable output of the sub-project. The integration test hits the real Anthropic API when `ANTHROPIC_API_KEY` is set (marked `skipif` otherwise). The demo script chains the STT sub-project's output (`veo_stt.json`) with the new `translate()` function. The real E2E run and the final milestone notes require `ANTHROPIC_API_KEY`; if you don't have it, implement the code and commit, and the controller will run the real E2E afterward.

- [ ] **Step 1: Create the integration test file**

Create `tests/test_translator_integration.py`:

```python
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
```

- [ ] **Step 2: Run the integration test**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_translator_integration.py -v
```

Expected outcomes:
- **If `ANTHROPIC_API_KEY` is not set:** `1 skipped` with the reason "ANTHROPIC_API_KEY not set". This is the expected outcome when the subagent runs without the key — the code is correct, the test just cannot be exercised in the current environment.
- **If `ANTHROPIC_API_KEY` is set:** `1 passed`, with Claude returning a translation that contains either "welcome" or "cartagena" in English.

Do not fail the task if the test skips. Skipping is a valid outcome.

- [ ] **Step 3: Create `examples/translate_demo.py`**

```python
"""Demo: translate a transcription JSON into another language.

Usage:
    uv run python examples/translate_demo.py <input_json> <target_language> [<output_json>]

Reads a Transcription JSON produced by transcribe_demo.py, translates it
via Claude API, and writes the result as a new JSON. Requires
ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.transcription.models import Segment, Transcription, Word  # noqa: E402
from core.transcription.serializers import write_json  # noqa: E402
from core.translation import translate  # noqa: E402


def _load_transcription(path: Path) -> Transcription:
    """Deserialize a Transcription JSON written by transcribe_demo.py."""
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = tuple(
        Segment(
            text=s["text"],
            start=s["start"],
            end=s["end"],
            words=tuple(
                Word(
                    text=w["text"],
                    start=w["start"],
                    end=w["end"],
                    probability=w["probability"],
                )
                for w in s.get("words", [])
            ),
            avg_logprob=s["avg_logprob"],
            no_speech_prob=s["no_speech_prob"],
        )
        for s in data["segments"]
    )
    return Transcription(
        language=data["language"],
        segments=segments,
        duration=data["duration"],
        model_size=data["model_size"],
    )


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    input_json = Path(argv[1])
    target_language = argv[2]
    output_json = (
        Path(argv[3])
        if len(argv) > 3
        else input_json.with_name(f"{input_json.stem}.{target_language}.json")
    )

    if not input_json.exists():
        print(f"error: input not found: {input_json}", file=sys.stderr)
        return 1

    source = _load_transcription(input_json)
    print(f"Loaded {len(source.segments)} segments in {source.language}")

    result = translate(source, target_language=target_language)
    print(f"Translated to {result.language}")

    write_json(result, output_json)
    print(f"Wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 4: Verify the demo imports cleanly without running it end-to-end**

```bash
cd ~/Projects/lipsync-corrector
uv run python -c "
import sys
sys.path.insert(0, 'examples')
import translate_demo
print('demo module OK:', hasattr(translate_demo, 'main'))
print('demo _load_transcription OK:', hasattr(translate_demo, '_load_transcription'))
"
```

Expected: prints `demo module OK: True` and `demo _load_transcription OK: True`.

(Do not actually run `translate_demo.py` against a real file in this step — that requires the API key and is covered by the controller's E2E run after the task.)

- [ ] **Step 5: Add a README section for the translation demo**

Open `README.md` and append at the end:

```markdown
## Translation demo (auto-dub sub-project: translate)

The `core.translation` module translates a Transcription into another
language via the Claude API:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python examples/translate_demo.py examples/veo_stt.json en
```

Reads the source JSON (produced by `transcribe_demo.py`), calls Claude
Haiku 4.5 with a structured tool schema that forces a 1:1 segment mapping,
and writes the translated result as `examples/veo_stt.en.json`. Segment
timestamps are preserved; word-level timestamps are dropped in the
translated output.

Default model is `claude-haiku-4-5-20251001`. Cost is on the order of
a fraction of a cent per 10 minutes of transcribed content at current
pricing.
```

- [ ] **Step 6: Write the milestone notes with placeholders**

Create `docs/milestones/translation.md`:

```markdown
# Translation Sub-project

**Date completed:** 2026-04-14
**Track:** Auto-dub sub-project 2 of 3 (STT → translation → TTS)
**Status:** Done

## What was built

- `core/translation/` subpackage with two focused modules:
  - `prompt.py` — pure functions `build_system_prompt` and
    `build_tool_schema`, no SDK imports.
  - `translator.py` — the public `translate()` function wrapping
    `anthropic.Anthropic.messages.create()` with `tool_choice` forced to
    `submit_translation`, plus a private `_adapt_response` helper that
    validates the response shape and builds the output `Transcription`.
- `examples/translate_demo.py` — standalone script chaining a JSON
  loader with `translate` and `write_json`. Reuses the existing
  `Transcription` serializer.
- 25 new tests (7 prompt + 17 translator unit + 1 integration) for a
  total suite of 176 (with API key set) or 175 (without).
- README section documenting the demo and the API key requirement.

## How to run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python examples/translate_demo.py examples/veo_stt.json en
```

Produces `examples/veo_stt.en.json` with translated segments and
preserved timestamps.

## Measured results

End-to-end run on the Veo-generated clip transcription (`veo_stt.json`
from the STT sub-project):

- Source: **<N>** segments in **<lang>** (to be filled after the
  controller runs the E2E)
- Target language: **en**
- Segments returned: **<N>**
- Wall time: **<N> s**
- Model: `claude-haiku-4-5-20251001`
- Cost (approximate): **<tokens → cents>**

### Sample output

```
<paste 2-3 translated segments here>
```

## What was learned

- <fill in after running>

## Deferred to future sub-projects

- Text-to-speech regeneration in the target language.
- Duration reconciliation between synthesized audio and original video.
- Unified `dub` CLI subcommand chaining all three pipeline steps.
- Multi-language batch translation (trivially done by calling
  `translate()` once per target).

## Next sub-project

**Sub-project 3: TTS (text-to-speech).** Consume the translated
`Transcription` produced here and emit synthesized audio in the target
language, ideally with voice cloning and segment-duration control.
Brainstorm the provider choice (ElevenLabs, Coqui XTTS-v2, OpenAI TTS)
before writing a plan.
```

Note: leave the `<placeholders>` in the "Measured results", "Sample output", and "What was learned" sections. The controller will fill them in after running the real E2E.

- [ ] **Step 7: Run the full suite**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 175 passed, 1 skipped (if no API key) OR 176 passed (if API key set).

- [ ] **Step 8: Commit the subagent's work**

```bash
cd ~/Projects/lipsync-corrector
git add tests/test_translator_integration.py examples/translate_demo.py README.md docs/milestones/translation.md
git commit -m "feat: add translation integration test, demo, and notes"
```

After this commit, the controller will run the real E2E against the Veo STT JSON, fill in the measured results in `docs/milestones/translation.md`, and commit the updated notes.

---

## Done criteria for this sub-project

- `uv run pytest` passes all tests. With `ANTHROPIC_API_KEY` set: 176 passed. Without: 175 passed, 1 skipped.
- `uv run python examples/translate_demo.py examples/veo_stt.json en` produces a valid `examples/veo_stt.en.json` file with translated segments and preserved timestamps (run by the controller after the task completes).
- `core/translation/` subpackage exists with 3 files (`__init__.py`, `prompt.py`, `translator.py`) following the plan's file structure.
- `from core.translation import translate` works.
- `docs/milestones/translation.md` written with actual measurements after the E2E run.
- README documents the demo and the `ANTHROPIC_API_KEY` requirement.
- Everything committed on `translation` branch, ready to merge to `main`.
- No existing tests regress. No changes to `cli/main.py`, `core/transcription/`, `core/wav2lip/`, `core/wav2lip_model.py`, or any other file outside the scope listed in Section 4 of the spec.

The TTS sub-project is out of scope. Do not start it in the same session.
