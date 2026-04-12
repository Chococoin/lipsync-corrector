# Milestone 1: Track B Setup + video_io

**Date completed:** 2026-04-12
**Track:** B
**Status:** Done

## What was built

- `core/` package with `device.py` (hardware selection) and `video_io.py` (VideoReader, VideoWriter, ffmpeg audio helpers).
- `cli/main.py` — lip-sync CLI entry point with passthrough pipeline.
- Test fixtures: auto-generated tiny test videos (with and without audio) in `conftest.py`.
- Full test suite: 33 tests passing.

## How to run the passthrough

```bash
uv run python -m cli.main \
  --video examples/input.mp4 \
  --output examples/passthrough_output.mp4
```

## Measured results

- Frame count preserved: yes (17151 in → 17151 out)
- Audio preserved: yes
- FPS match: yes (30.0 fps)
- Resolution match: yes (854x480)
- Visual quality: identical (passthrough, no processing)

## Code quality review findings

- Added `ffprobe` check to `ensure_ffmpeg()` per reviewer recommendation.
- Deferred: `has_audio_stream` returncode validation, input path validation in ffmpeg helpers, frame shape validation in VideoWriter. These will become more important in later milestones when ML models produce frames.

## What was learned

- The passthrough confirms the I/O layer is lossless: frame count and audio are preserved exactly through the read → write → mux pipeline.
- VideoReader and VideoWriter work cleanly as context managers, keeping resource management simple.
- All 33 tests run in ~1.2 seconds using tiny 64x64 auto-generated test videos — no dependency on user-provided media files for CI.

## Next milestone

Milestone 2: `face_tracker` — stable face detection and tracking across frames.
See `docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md` section 8.
