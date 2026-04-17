# TTS Sub-project

**Date completed:** <YYYY-MM-DD>
**Track:** Auto-dub sub-project 3 of 3 (STT → translation → TTS)
**Status:** Done

## What was built

- `core/tts/` subpackage with three modules:
  - `reference.py` — automatic voice reference selection from the source
    audio's cleanest speech segments.
  - `assembler.py` — temporal positioning of generated audio segments
    with silence gaps matching the original video timing.
  - `synthesizer.py` — the public `synthesize()` function wrapping
    Coqui XTTS-v2 with per-segment voice cloning.
- `examples/tts_demo.py` — standalone demo script.
- <N> new tests for a total suite of <N>.
- README section documenting the demo.

## How to run

```bash
uv run python examples/tts_demo.py \
  examples/veo_stt.en.json \
  examples/veo_audio_16k_mono.wav \
  examples/veo_dubbed_en.wav
```

## Measured results

- Segments generated: <N>
- Total duration: <N>s
- Wall time: <N>s
- Model: XTTS-v2 (~1.8 GB cached)
- Device: <mps/cpu>
- Voice cloning reference: <N>s from <N> segments

### Auditory assessment

- <fill in after listening>

## What was learned

- <fill in after running>

## Deferred

- Unified `dub` CLI command chaining STT → translation → TTS → Wav2Lip.
- Sample rate conversion (24 kHz → 16 kHz) for Wav2Lip integration.
- Multi-speaker support.
- XTTS `speed` parameter tuning if duration mismatch persists.

## Next milestone

Integration milestone: chain all pipeline steps into a single `dub` command.
