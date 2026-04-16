# Blending Pass (Mouth-Only Mask) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify `blend_back` to support a mouth-only mask that preserves the original pixels in the upper face (eyes, eyebrows, forehead) and only blends the Wav2Lip output in the lower face (mouth region), eliminating the unnecessary blur caused by the 96x96 resize cycle on parts of the face that Wav2Lip didn't change.

**Architecture:** A surgical change to one function (`blend_back` in `core/blender.py`) adding a `mouth_only` parameter that replaces the uniform feathered mask with a vertical gradient: zero at the top (original pixels), linear ramp in the middle (nose bridge transition), full opacity at the bottom (Wav2Lip mouth). The CLI passes `mouth_only=True` when `--lipsync` is active. No new modules, dependencies, or CLI flags.

**Tech Stack:** Python 3.11, numpy, opencv-python. No new dependencies.

**Branch:** `blending-pass` off `main` (currently at `1bfe564`).

**Design spec:** `docs/superpowers/specs/2026-04-17-blending-pass-design.md`.

---

## File Structure (end state)

```
lipsync-corrector/
├── core/
│   └── blender.py                 # MODIFIED: blend_back gains mouth_only + ratio params
├── cli/
│   └── main.py                    # MODIFIED: one line change, passes mouth_only=True
├── tests/
│   └── test_blender.py            # MODIFIED: ~7 new tests added
└── docs/milestones/
    └── blending-pass.md           # NEW: milestone notes with visual comparison
```

---

## Task 1: Modify `blend_back` with mouth-only mask + tests

**Files:**
- Modify: `~/Projects/lipsync-corrector/core/blender.py`
- Modify: `~/Projects/lipsync-corrector/tests/test_blender.py`

- [ ] **Step 1: Create the feature branch**

```bash
cd ~/Projects/lipsync-corrector
git checkout -b blending-pass
```

- [ ] **Step 2: Write failing tests**

Append the following new test class to the end of `tests/test_blender.py`:

```python
class TestBlendBackMouthOnly:
    def test_mouth_only_default_false_matches_original(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result_default = blend_back(frame, modified, face_crop, feather_pixels=0)
        result_explicit = blend_back(frame, modified, face_crop, feather_pixels=0, mouth_only=False)
        np.testing.assert_array_equal(result_default, result_explicit)

    def test_top_of_face_is_original(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop, feather_pixels=0, mouth_only=True)
        # Top 30% of bbox (rows 100-159) should be pure original (100).
        # Using 30% instead of 40% to stay safely above the transition zone.
        top_region = result[100:160, 150:250, 0]
        assert np.all(top_region == 100)

    def test_bottom_of_face_is_modified(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop, feather_pixels=0, mouth_only=True)
        # Bottom 30% of bbox (rows 241-299) should be pure modified (200).
        # Using 70% to stay safely below the transition zone.
        bottom_region = result[241:299, 150:250, 0]
        assert np.all(bottom_region >= 199)

    def test_transition_zone_is_blended(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop, feather_pixels=0, mouth_only=True)
        # Transition zone center (~47.5% of bbox = row 195) should be
        # between 100 and 200 (a blend of original and modified).
        # mouth_top_ratio=0.4 → transition starts at row 180
        # mouth_blend_ratio=0.15 → transition ends at row 210
        # Center of transition at row 195.
        transition_value = result[195, 200, 0]
        assert 110 < transition_value < 190

    def test_outside_bbox_unchanged(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop, feather_pixels=0, mouth_only=True)
        np.testing.assert_array_equal(result[0:100, :], frame[0:100, :])
        np.testing.assert_array_equal(result[300:, :], frame[300:, :])
        np.testing.assert_array_equal(result[:, 0:100], frame[:, 0:100])
        np.testing.assert_array_equal(result[:, 300:], frame[:, 300:])

    def test_small_bbox_does_not_crash(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((20, 20, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 120, 120, image_size=(20, 20))
        result = blend_back(frame, modified, face_crop, mouth_only=True)
        assert result.shape == frame.shape

    def test_returns_copy_not_original(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop, mouth_only=True)
        assert result is not frame
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_blender.py::TestBlendBackMouthOnly -v
```

Expected: `TypeError: blend_back() got an unexpected keyword argument 'mouth_only'` for most tests.

- [ ] **Step 4: Implement the mouth-only mask in `blend_back`**

Replace the entire content of `core/blender.py` with:

```python
from __future__ import annotations

import cv2
import numpy as np

from core.mouth_region import FaceCrop


def blend_back(
    frame: np.ndarray,
    modified_crop: np.ndarray,
    face_crop: FaceCrop,
    feather_pixels: int = 8,
    mouth_only: bool = False,
    mouth_top_ratio: float = 0.4,
    mouth_blend_ratio: float = 0.15,
) -> np.ndarray:
    """Paste a modified face crop back onto the frame with feathered edges.

    When mouth_only is False (default), the entire crop is blended with a
    uniform feathered mask — identical to the original behavior.

    When mouth_only is True, the mask uses a vertical gradient: zero opacity
    in the upper face (eyes, forehead stay original), linear ramp through
    the nose bridge, full opacity in the lower face (mouth from the model).
    This preserves the original resolution in the upper face while only
    applying the model's output where it matters.
    """
    result = frame.copy()
    x1, y1, x2, y2 = face_crop.bbox.astype(int)
    dst_w = x2 - x1
    dst_h = y2 - y1
    if dst_w <= 0 or dst_h <= 0:
        return result

    warped = cv2.resize(modified_crop, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    if mouth_only:
        mask = _build_mouth_only_mask(dst_h, dst_w, feather_pixels,
                                       mouth_top_ratio, mouth_blend_ratio)
    else:
        if feather_pixels > 0 and dst_w > 2 * feather_pixels and dst_h > 2 * feather_pixels:
            inner = np.zeros((dst_h, dst_w), dtype=np.float32)
            inner[feather_pixels:-feather_pixels, feather_pixels:-feather_pixels] = 1.0
            k = feather_pixels * 2 + 1
            mask = cv2.GaussianBlur(inner, (k, k), 0)
        else:
            mask = np.ones((dst_h, dst_w), dtype=np.float32)

    mask3 = np.stack([mask, mask, mask], axis=-1)

    region = result[y1:y2, x1:x2].astype(np.float32)
    blended = mask3 * warped.astype(np.float32) + (1.0 - mask3) * region
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def _build_mouth_only_mask(
    dst_h: int,
    dst_w: int,
    feather_pixels: int,
    mouth_top_ratio: float,
    mouth_blend_ratio: float,
) -> np.ndarray:
    """Build a mask that covers only the lower face (mouth region).

    The mask has three vertical zones:
    - Top (0 to transition_start): 0.0 — original pixels preserved
    - Transition (transition_start to transition_end): linear ramp 0→1
    - Bottom (transition_end to dst_h): 1.0 — model output used

    The vertical gradient is multiplied with lateral feathering (left/right
    edges fade to 0) so the blend is smooth in all directions.
    """
    transition_start = int(mouth_top_ratio * dst_h)
    transition_end = int((mouth_top_ratio + mouth_blend_ratio) * dst_h)
    transition_end = min(transition_end, dst_h)

    vertical = np.zeros((dst_h, dst_w), dtype=np.float32)
    if transition_end > transition_start:
        ramp_len = transition_end - transition_start
        ramp = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
        vertical[transition_start:transition_end, :] = ramp[:, np.newaxis]
    vertical[transition_end:, :] = 1.0

    if feather_pixels > 0 and dst_w > 2 * feather_pixels:
        lateral = np.zeros((dst_h, dst_w), dtype=np.float32)
        lateral[:, feather_pixels:-feather_pixels] = 1.0
        k = feather_pixels * 2 + 1
        lateral = cv2.GaussianBlur(lateral, (k, k), 0)
    else:
        lateral = np.ones((dst_h, dst_w), dtype=np.float32)

    return vertical * lateral
```

- [ ] **Step 5: Run mouth-only tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_blender.py::TestBlendBackMouthOnly -v
```

Expected: 7 passed.

- [ ] **Step 6: Run all blender tests (verify backward compat)**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_blender.py -v
```

Expected: 15 passed (8 existing + 7 new). The existing `TestBlendBack` tests all use `mouth_only=False` (default) and should produce identical results to before.

- [ ] **Step 7: Run full suite**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 182 passed (175 existing + 7 new), 1 skipped (translation integration test if no API key).

- [ ] **Step 8: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/blender.py tests/test_blender.py
git commit -m "feat: add mouth-only blend mask preserving original eyes/forehead"
```

---

## Task 2: Wire mouth_only in CLI + E2E visual verification + milestone notes

**Files:**
- Modify: `~/Projects/lipsync-corrector/cli/main.py`
- Create: `~/Projects/lipsync-corrector/docs/milestones/blending-pass.md`

- [ ] **Step 1: Modify `cli/main.py`**

In `cli/main.py`, find line 119 (the `blend_back` call inside the lipsync frame loop):

```python
                            frame = blend_back(frame, processed[0], face_crop)
```

Replace it with:

```python
                            frame = blend_back(frame, processed[0], face_crop, mouth_only=True)
```

This is the only code change in the CLI. No new flags, no new imports.

- [ ] **Step 2: Run the CLI tests to verify no regressions**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_cli.py -v
```

Expected: 19 passed. The existing `TestMainLipsync` tests use `IdentityModel` which produces identical crops — the mouth_only mask makes no visible difference when the modified crop equals the original. These tests are testing frame count and audio preservation, not pixel values, so they pass regardless.

- [ ] **Step 3: Run the full test suite**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 182 passed (same as after Task 1), 1 skipped.

- [ ] **Step 4: Run the Wav2Lip pipeline on the Veo clip**

```bash
cd ~/Projects/lipsync-corrector
time uv run python -m cli.main \
  --video ~/Downloads/Video_De_Mujer_Saludando_Generado.mp4 \
  --audio examples/veo_audio_16k_mono.wav \
  --output examples/blending_pass.mp4 \
  --lipsync --model wav2lip
```

Expected output: "Wrote 192 frames to intermediate" / "Audio preserved" / "Done."
Expected wall time: approximately the same as Milestone 3b (~35 seconds) — the mask construction adds microseconds of numpy per frame.

- [ ] **Step 5: Open the output video and compare visually**

```bash
open examples/blending_pass.mp4
```

Expected visual behavior — compare with the Milestone 3b output you remember:
- **Eyes, eyebrows, forehead:** pixel-sharp from the original video. This is the key improvement — in 3b these were blurry.
- **Mouth:** Wav2Lip-generated, lip-sync matches audio. Still at 96x96 upscaled resolution (some blur expected, but confined to a smaller area now).
- **Nose bridge transition zone:** smooth gradient, no visible line or seam between original-pixels and Wav2Lip-pixels.
- **Outside the face bbox:** unchanged (same as before).
- **Audio:** preserved.

Record your subjective assessment for the milestone notes (Step 7).

- [ ] **Step 6: Verify frame count and audio**

```bash
cd ~/Projects/lipsync-corrector
uv run python -c "
from pathlib import Path
from core.video_io import VideoReader, has_audio_stream
out = Path('examples/blending_pass.mp4')
with VideoReader(out) as r:
    print(f'Output: {r.frame_count} frames, {r.fps:.1f} fps, {r.width}x{r.height}')
print(f'Audio preserved: {has_audio_stream(out)}')
"
```

Expected: 192 frames, 24.0 fps, 1280x720, audio preserved.

- [ ] **Step 7: Write milestone notes**

Create `docs/milestones/blending-pass.md`:

```markdown
# Blending Pass: Mouth-Only Mask

**Date completed:** <YYYY-MM-DD>
**Track:** B (original milestone plan: Milestone 4)
**Status:** Done

## What was built

A mouth-only blend mask for the Wav2Lip pipeline. Instead of blending the
entire face crop (which degraded the eyes and forehead with unnecessary
96x96 blur), the new mask preserves the original pixels in the upper face
and only applies the Wav2Lip output in the lower face (mouth region). The
transition zone at the nose bridge uses a linear gradient for a seamless
blend.

**Key insight (from the user):** the original approach of restoring degraded
pixels with GFPGAN was solving a self-inflicted problem. The simpler fix is
to not degrade them in the first place.

**Changes:**
- `core/blender.py` — `blend_back()` gained `mouth_only`, `mouth_top_ratio`,
  and `mouth_blend_ratio` parameters. A new `_build_mouth_only_mask()` helper
  constructs the vertical gradient mask. Default `mouth_only=False` preserves
  backward compatibility.
- `cli/main.py` — one line changed: passes `mouth_only=True` when `--lipsync`
  is active.
- 7 new tests in `test_blender.py` verifying the mask behavior, for a total
  suite of <N> tests.

## How to run

Same command as Milestone 3b — no new flags:

```bash
uv run python -m cli.main \
  --video examples/input.mp4 \
  --audio examples/dubbed.wav \
  --output examples/out.mp4 \
  --lipsync --model wav2lip
```

## Measured results

End-to-end on the Veo clip (192 frames, 24 fps, 1280x720):

- Frame count preserved: <yes/no>
- Audio preserved: <yes/no>
- Wall time: <N> s (compare to ~35s in milestone 3b)

### Visual comparison vs Milestone 3b

- Eyes/forehead: <describe — should be "sharp, pixel-identical to original">
- Mouth region: <describe — should be "same as 3b, lip-sync works">
- Nose transition: <describe — should be "seamless, no visible line">
- Overall impression: <fill in>

## What was learned

- <fill in after running>

## Deferred

- GFPGAN face restoration on the mouth zone only (if the 96x96 mouth
  blur is still too noticeable).
- Landmark-based mask positioning (replaces fixed ratios with detected
  facial feature positions).
- Color correction at the transition boundary (if skin tone mismatch
  becomes visible in some clips).

## Next milestone

Evaluate whether GFPGAN is still needed for the mouth zone, or proceed
to the TTS sub-project to complete the auto-dub pipeline.
```

Fill in the `<placeholders>` after the visual inspection.

- [ ] **Step 8: Clean up the example output**

```bash
rm -f ~/Projects/lipsync-corrector/examples/blending_pass.mp4
```

- [ ] **Step 9: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add cli/main.py docs/milestones/blending-pass.md
git commit -m "feat: blending pass complete — mouth-only mask preserving original eyes"
```

---

## Done criteria

- `uv run pytest` passes all tests (182 + previous, plus 1 skipped if no API key).
- `uv run python -m cli.main --video <real-video> --audio <real-audio> --output <path> --lipsync --model wav2lip` produces a video where the upper face is sharp (original pixels) and the lower face has Wav2Lip lip-sync, with a seamless transition at the nose bridge.
- Existing behavior with `mouth_only=False` (default) is unchanged — all 8 original blender tests pass without modification.
- `cli/main.py` has one line changed (passes `mouth_only=True`). No new CLI flags.
- `docs/milestones/blending-pass.md` written with visual comparison notes.
- Everything committed on `blending-pass` branch, ready to merge to `main`.
- No new dependencies, no new model downloads.
