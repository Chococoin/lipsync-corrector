# Milestone 0: Face-Swap Quick Win

**Date completed:** 2026-04-12
**Track:** A
**Status:** Done

## What was built

A single CLI script (`swap.py`) that takes a reference face image and a video,
and produces a video with the face swapped using `inswapper_128` from InsightFace.

## How to run

```bash
uv run python swap.py \
  --face examples/reference.png \
  --video examples/input.mp4 \
  --output examples/output.mp4 \
  --max-seconds 15
```

## Environment on the test machine

- Apple M4, macOS
- Python 3.11.14, uv 0.10.7
- onnxruntime 1.24.4, insightface 0.7.3, opencv 4.11.0, numpy 1.26.4
- onnxruntime providers available: CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider
- Providers used by the script: CoreMLExecutionProvider, CPUExecutionProvider

## Measured performance

- Input clip: 8s, 1280x720, 24 fps
- Total frames processed: 192
- Processing time: 40.0s
- Effective fps: 4.80 fps
- Faces per frame: 1

## Observed quality and defects

- The swap is visible: facial features (nose, eyes, jawline, skin tone) change to match the reference.
- Blending is decent at this resolution — inswapper handles lighting adaptation reasonably well.
- The model runs on CoreML (Apple Silicon GPU), not CPU — confirmed by provider logs.
- At 4.8 fps, a 1-minute clip would take ~12.5 minutes to process. Acceptable for batch/offline use.
- FutureWarning from insightface (deprecated `estimate` and `rcond`): cosmetic, does not affect output.

### Defects to watch in longer/harder clips (not observed in this 8s test):
- Temporal flicker between frames (inswapper processes each frame independently).
- Boundary artifacts when head rotation exceeds ~30 degrees.
- Lighting mismatch when reference photo lighting differs sharply from scene lighting.
- No handling of occluded faces (hands, microphones).

## What was learned

- CoreML execution provider works out of the box on M4 with onnxruntime 1.24.4. No CUDA workarounds needed.
- The 8s AI-generated test clip (Veo) worked perfectly — clean face detection on every frame. Real-world clips with motion blur, occlusion, and varying angles will be harder.
- 4.8 fps on M4 with CoreML is a usable baseline for offline processing. Real-time is not feasible with this model at this resolution.

## Next milestone

Milestone 1: Track B setup and `video_io` module. See the design spec
`docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md` section 8.

The concrete next action when returning to the project:

```bash
cd ~/Projects/lipsync-corrector
cat docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md
# Then open a new plan session for Milestone 1.
```
