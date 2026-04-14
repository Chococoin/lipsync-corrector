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
