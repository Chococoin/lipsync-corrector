# STT Transcription Sub-project

**Date completed:** 2026-04-14
**Track:** Auto-dub sub-project 1 of 3 (STT → translation → TTS)
**Status:** Done

## What was built

- `core/transcription/` subpackage with four focused modules:
  - `__init__.py` — re-exports the six public names.
  - `models.py` — frozen dataclasses `Word`, `Segment`, `Transcription`.
  - `serializers.py` — `write_json`, `write_srt`, and `_format_srt_timestamp`.
  - `transcriber.py` — `transcribe(audio_path, model_size, language)`
    wrapping `mlx_whisper.transcribe` and adapting the raw dict into our
    dataclasses via private `_adapt_*` helpers.
- `examples/transcribe_demo.py` — standalone script chaining
  `extract_audio_as_pcm_wav` + `transcribe` + serializers for manual
  E2E runs. 45 lines, no tests (assembler of tested pieces).
- `core/video_io.py` — new `extract_audio_as_pcm_wav` helper that
  re-encodes to 16 kHz mono PCM instead of copying the source codec.
  Fixes a latent bug where the existing `extract_audio` produced an
  AAC-in-WAV container that ML audio loaders can't decode. Existing
  callers (milestone 3a/3b) keep using the original helper unchanged.
- 40 new tests (2 new video_io + 13 model + 16 serializer + 9 transcriber)
  for a total suite of 151 tests, all passing with the medium model
  cached and the Veo audio fixture on disk.
- README section documenting the demo.

## How to run

```bash
uv run python examples/transcribe_demo.py ~/Downloads/Video_De_Mujer_Saludando_Generado.mp4
```

Produces `<stem>.wav`, `<stem>.json`, `<stem>.srt` next to the input video.

## Measured results

End-to-end run on the Veo-generated clip:

- Input: `~/Downloads/Video_De_Mujer_Saludando_Generado.mp4`
  (192 frames, 24 fps, 1280x720, ~8 s audio)
- Detected language: **es** (Spanish)
- Segments: **3**
- Duration (computed from last segment end): **7.5 s**
- Wall time (with model already cached): **~7 s** total
  (extract_audio ~0.3 s + transcribe ~6 s + serializers negligible)
- Model: `mlx-community/whisper-medium-mlx` (~1.4 GB on disk in
  `~/.cache/huggingface/hub/`)
- Word-level timestamps produced with per-word probabilities
  (0.72–0.99 range for this clip)

### Sample output

```
1
00:00:00,000 --> 00:00:02,800
 Hola a todos, bienvenidos a Cartagena.

2
00:00:03,140 --> 00:00:05,060
 Miren qué lugar tan hermoso.

3
00:00:05,180 --> 00:00:07,540
 Estoy muy feliz de estar aquí hoy con ustedes.
```

(Fun learning: the Veo-generated clip is scripted as a Cartagena tourism
greeting, which we didn't know until this milestone ran. The STT is
revealing content we never actually listened to consciously.)

## What was learned

- **The plan's biggest surprise was a latent bug in existing code.**
  The Milestone 1 `extract_audio` helper copies the source audio codec
  (`-acodec copy`), which means extracting AAC from an mp4 produces a
  `.wav` container with AAC data inside. This worked fine for the
  passthrough and Wav2Lip pipelines because the extracted audio was
  only ever muxed back into an mp4 (same codec in, same codec out).
  It broke immediately when mlx-whisper tried to actually decode the
  "wav" file. Fix: a parallel `extract_audio_as_pcm_wav` that forces
  16 kHz mono PCM. The old helper stays unchanged to avoid regression
  risk in existing milestones. This is the kind of bug that only
  shows up when a module's assumed contract meets a new consumer.
- **Model caching was faster than expected.** `mlx-whisper` fetches
  from `mlx-community/whisper-medium-mlx` on HuggingFace Hub. The
  download landed in `~/.cache/huggingface/hub/` so transparently
  that we had to explicitly verify it happened — there was no
  progress bar because the first (failed) demo run had already
  triggered and completed the download before crashing on the
  AAC/PCM bug.
- **The dataclass-with-tuples pattern works cleanly for immutable
  ML result structures.** `frozen=True` + `tuple` instead of `list`
  means `Transcription` objects are hashable, easy to diff, and
  safe to cache by identity. Next sub-project (translation) can
  take one of these as input without worrying about accidental
  mutation between processing stages.
- **Python path gotcha for standalone scripts.** Running
  `python examples/transcribe_demo.py` from the repo root fails with
  `ModuleNotFoundError: No module named 'core'` because the `examples/`
  directory is not a package and pytest's `pythonpath=["."]` only
  applies to test runs. The fix is a two-line `sys.path.insert` at
  the top of the demo. Alternative would be making `examples/` a
  real package with `__init__.py`, but that feels heavy for a demo.
- **MLX uses GPU by default on M4.** `mlx.core.default_device()`
  prints `Device(gpu, 0)` out of the box. No device selection code
  needed — MLX auto-picks Metal on Apple Silicon.

## Deferred to future sub-projects

- Translation of the transcribed text to another language.
- Voice cloning / TTS in the target language.
- Duration reconciliation between the new synthesized audio and the
  original video.
- Integration with `cli/main.py` under a unified `dub` subcommand.
- Whether to expose model size through the demo script (currently
  hardcoded to `medium`).

## Next sub-project

**Sub-project 2: Translation.** Consume the `Transcription` object
produced here and emit a translated version in a target language.
Brainstorm the provider choice (Claude API vs DeepL vs local NLLB)
before writing a plan.
