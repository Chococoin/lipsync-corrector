# Milestone 3a: Lip-Sync Pipeline Scaffold

**Date completed:** 2026-04-12
**Track:** B
**Status:** Done

## What was built

- `core/mouth_region.py` — `FaceCrop` dataclass + `crop_face_region()` that expands the tracker bbox by a padding fraction, clamps to frame bounds, and resizes to a target size (default 96x96 for Wav2Lip compatibility).
- `core/lipsync_model.py` — `LipSyncModel` abstract base class with `process(face_crops, audio_path)` interface. `IdentityModel` concrete implementation returns crops unchanged. This establishes the API that real models (Wav2Lip, LatentSync, MuseTalk) will plug into.
- `core/blender.py` — `blend_back()` resizes the modified crop to the original bbox size, builds a feathered alpha mask (soft edges via Gaussian blur), and alpha-blends into a copy of the frame.
- `cli/main.py` — new `--lipsync` flag that runs the full pipeline: track face → crop region → run model → blend back → write.
- Test suite: 84 tests passing (25 new for milestone 3a: 8 mouth_region + 9 lipsync_model + 8 blender + 5 CLI).

## How to run

```bash
uv run python -m cli.main \
  --video examples/input.mp4 \
  --output examples/lipsync_scaffold.mp4 \
  --lipsync
```

## Measured results

End-to-end run on the Veo-generated clip (192 frames, 24 fps, 1280x720):

- Frame count preserved: **yes** (192 → 192)
- fps preserved: **yes** (24.0 → 24.0)
- Dimensions preserved: **yes** (1280x720 → 1280x720)
- Audio preserved: **yes** (`has_audio_stream` returns True)
- Visual behavior: the face region is slightly blurred due to the 96x96 double-resize cycle; the rest of the frame is unchanged. Feathered blend produces no visible bbox rectangle at the boundary.

## What was learned

- The placeholder `IdentityModel` produces a useful visual signature (soft face region from double resize) that lets us confirm the pipeline is actually running without needing a real ML model — perfect for validating geometry before Milestone 3b.
- Feathered alpha blending via a Gaussian-blurred inner mask cleanly hides the bbox boundary without requiring face landmarks.
- Keeping `FaceCrop` as a carrier for both the resized image and the exact bbox (after padding + clamping) means the blender does not need to re-derive the paste region — the crop stage is the single source of truth for the geometry.
- The `--lipsync` branch is independent from `--debug-tracking`; they are mutually exclusive via `if/elif` in the frame loop, which keeps earlier milestone behavior untouched.

## Deferred to Milestone 3b

- Real lip-sync model integration (Wav2Lip or a modern alternative).
- Audio preprocessing (mel spectrogram, 16 kHz mono resampling).
- Batch processing of crops (current per-frame call works for IdentityModel but will need refactoring for models that require temporal context).
- Face alignment via landmarks (currently bbox-only; Wav2Lip will work fine without rotation alignment on near-frontal faces).

## Next milestone

Milestone 3b: replace `IdentityModel` with a real Wav2Lip wrapper. See
`docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md` section 8.
