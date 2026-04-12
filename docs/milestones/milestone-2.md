# Milestone 2: Face Tracker

**Date completed:** 2026-04-12
**Track:** B
**Status:** Done

## What was built

- `core/face_tracker.py` with:
  - `TrackedFace` dataclass (bbox, landmarks, confidence, detected flag)
  - `BboxSmoother` — exponential moving average smoothing + gap interpolation (pure numpy)
  - `FaceTracker` — wraps insightface FaceAnalysis (buffalo_l) with the smoother
  - `draw_tracking_overlay()` — debug visualization (green/orange bbox, confidence label, landmark dots)
- `cli/main.py` updated with `--debug-tracking` flag that integrates the tracker into the pipeline.
- Test suite: **54 tests passing** (33 existing + 19 new in test_face_tracker.py + 2 new in test_cli.py).

## How to run

```bash
uv run python -m cli.main \
  --video examples/input.mp4 \
  --output examples/tracking_debug.mp4 \
  --debug-tracking
```

## Measured results

End-to-end run on the Veo-generated clip (`Video_De_Mujer_Saludando_Generado.mp4`, 192 frames, 24 fps, 1280x720):

- Providers used: `CoreMLExecutionProvider`, `CPUExecutionProvider`
- Detection worked on every frame (no gaps observed on this clip)
- Confidence consistently ~0.83 on the primary face
- **Bbox stable** across frames (visible in the output — EMA smoothing removed micro-jitter)
- **Landmarks** (5 keypoints: eyes, nose, mouth corners) drawn in blue, track the face cleanly
- Green bbox confirmed — no `interp` frames in this clip because detection never failed

## Review feedback applied

During code review, four fixes were applied before merge:

1. **EMA bbox float cast** — now converts input to float64 on every observation, not just the first, so subsequent integer inputs would not silently truncate.
2. **Landmark deep copy** — `BboxSmoother` now copies landmarks on storage to guard against caller-side mutation.
3. **Test name clarity** — renamed `test_gap_handling_on_faceless_frames` to `test_faceless_frames_return_none` since the test does not actually exercise gap expiry (the smoother stays in its unseeded state throughout).
4. **Orange color test** — added a test that explicitly verifies `draw_tracking_overlay` uses the orange (0,165,255) BGR color when `tracked.detected is False`.

Issues deferred to future milestones:
- `TestFaceTracker` smoke tests do not actually detect a face (they only verify the negative path on solid-color frames). A positive test would need a face fixture. Visual verification in this milestone's end-to-end run closes the gap in practice.
- `ctx_id=0` is hardcoded in `FaceTracker.__init__`. Same pattern as `swap.py`, works on M4 with CoreML. Revisit if multi-GPU ever matters.
- Landmarks are not smoothed (only bboxes are). Observed: landmarks show some micro-jitter relative to the bbox. This is expected and acceptable for debug visualization; if it matters for downstream mouth-region extraction in Milestone 3, we will add per-point EMA there.

## What was learned

- `insightface.FaceAnalysis(buffalo_l)` is fast enough on M4 with CoreML for per-frame detection at 24 fps on 1280x720 — the tracker ran end-to-end on 192 frames in well under a minute including model load.
- Exponential moving average with `alpha=0.3` and `max_gap=5` gives visibly stable bboxes without perceptible lag on a clip with moderate head motion.
- The `--debug-tracking` overlay is a genuinely useful debugging tool — being able to see confidence + landmarks + detected-vs-interpolated state per frame will help tune parameters in later milestones.

## Next milestone

Milestone 3: Wav2Lip integrated (baseline lip-sync). This is where the pipeline starts actually modifying the mouth region to match a dubbed audio track. Before that, Milestone 3 will also need:
- `core/mouth_region.py` — crop and align the mouth region using the landmarks from this milestone
- `core/lipsync_model.py` — thin wrapper around Wav2Lip
- `core/blender.py` — paste generated mouth back onto the frame

See `docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md` section 8 for the full roadmap.
