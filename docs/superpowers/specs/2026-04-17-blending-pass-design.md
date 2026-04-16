# Blending Pass (Mouth-Only Mask) — Design Spec

**Date:** 2026-04-17
**Status:** Approved for implementation
**Author:** chocos (with Claude)
**Predecessor:** `2026-04-11-lipsync-corrector-design.md` (Milestone 4 in the original plan)

## 1. Purpose

Improve the visual quality of the Wav2Lip pipeline output by blending only the mouth region of the model's output back into the frame, instead of blending the entire face. The upper portion of the face (eyes, eyebrows, forehead, nose bridge) stays pixel-perfect from the original video, eliminating the blur caused by the 96x96 resize cycle that those regions currently undergo unnecessarily.

This replaces the original plan's "GFPGAN face restoration" approach with a simpler, zero-dependency solution that addresses the root cause: do not degrade pixels that did not need to change.

## 2. Problem Statement

The current pipeline (Milestone 3b) does this:

1. Crop the full face bbox from the frame (e.g. 300x300 pixels).
2. Resize to 96x96 (Wav2Lip's native resolution).
3. Wav2Lip generates a new mouth in the 96x96 crop.
4. Resize back to 300x300 (now blurry due to the double interpolation).
5. Blend the **entire** 300x300 back into the frame with a feathered alpha mask.

Step 5 is the problem. Wav2Lip only modified the bottom half of the 96x96 crop (the mouth). The top half (eyes, forehead) traveled through the resize round-trip for nothing — they came back blurry and replaced the sharp originals in the frame.

The feathered alpha blend from Milestone 3a successfully hides the bbox boundary (no visible rectangle), but it does not help with the resolution loss inside the blended region.

## 3. Solution

Modify `blend_back` in `core/blender.py` to support a `mouth_only` mode where the alpha mask uses a vertical gradient: zero opacity in the upper portion of the face (preserving originals) transitioning smoothly to full opacity in the lower portion (using Wav2Lip's generated mouth).

This is a surgical change to one function in one file. No new modules, no new dependencies, no new model downloads, no new CLI flags.

## 4. Scope

**In scope:**

- Modify `core/blender.py`: add `mouth_only`, `mouth_top_ratio`, and `mouth_blend_ratio` parameters to `blend_back`.
- Modify `cli/main.py`: pass `mouth_only=True` when the lipsync pipeline is active.
- Add ~7 new tests to `tests/test_blender.py` covering the mouth-only mask behavior.
- Visual verification via E2E run on the Veo clip.
- Milestone notes in `docs/milestones/blending-pass.md`.

**Explicitly out of scope:**

- GFPGAN or any face restoration model. Deferred — may not be needed after this change.
- An `--enhance` CLI flag. Not needed because mouth-only blending is strictly better than full-face blending and is always active when `--lipsync` is used.
- Landmark-based mask positioning. Fixed ratios are sufficient for the conversational (near-frontal) use case. Landmark-driven masking can be added later if needed.
- Changes to `core/mouth_region.py`, `core/wav2lip_model.py`, or `core/wav2lip/`. The crop and inference stages are unchanged.

## 5. Constraints and Decisions

1. **Mouth-only blend is implicit, not a flag.** When `--lipsync` is active, `mouth_only=True` is always passed. There is no `--mouth-only` or `--enhance` flag. This is because mouth-only blending is strictly better than full-face blending for lip-sync — there is no scenario where a user would prefer the blurry-eyes version.

2. **Backward compatibility via default parameter.** `blend_back(..., mouth_only=False)` is the default. All existing tests and callers that do not pass the parameter get identical behavior to before. Only the `--lipsync` path in `cli/main.py` passes `True`.

3. **Fixed ratios, not landmarks.** The mask transition zone is defined by `mouth_top_ratio=0.4` (transition starts at 40% from top of bbox) and `mouth_blend_ratio=0.15` (transition zone is 15% of bbox height). These are based on average facial proportions in conversational framing. They are parameters on the function signature (not hardcoded constants), so a future landmark-based approach can compute them dynamically and pass them in.

4. **The crop sent to Wav2Lip remains the full face.** Wav2Lip's architecture requires the full face as input — the upper half is used as the identity reference (channels 0-2 of the 6-channel input have the lower half zeroed, channels 3-5 have the full face). We cannot reduce the crop without retraining the model. The change is only in what portion of the **output** we use.

5. **The vertical gradient multiplies with the existing lateral feathering.** The current `blend_back` already creates a feathered mask (Gaussian-blurred inner rectangle) for smooth edges on the left/right/top/bottom of the bbox. The mouth-only gradient is multiplied element-wise with this feathering, producing a mask with smooth falloff in all directions but biased toward the bottom half.

## 6. The Mask Construction

When `mouth_only=True`, the mask is built as follows:

```python
# dst_h = height of the bbox region in the frame
transition_start = int(mouth_top_ratio * dst_h)       # e.g. row 80 of 200
transition_end = int((mouth_top_ratio + mouth_blend_ratio) * dst_h)  # e.g. row 110

vertical_mask = np.zeros((dst_h, dst_w), dtype=np.float32)
# Transition zone: linear ramp from 0 to 1
if transition_end > transition_start:
    ramp = np.linspace(0.0, 1.0, transition_end - transition_start)
    vertical_mask[transition_start:transition_end, :] = ramp[:, np.newaxis]
# Below transition: full opacity
vertical_mask[transition_end:, :] = 1.0

# Multiply with lateral feathering (already computed)
mask = vertical_mask * lateral_feather_mask
```

With the default ratios (0.4 / 0.15) and a 200px-tall bbox:
- Rows 0-79: opacity 0.0 (eyes, eyebrows, forehead — original pixels)
- Rows 80-109: linear ramp 0.0→1.0 (nose bridge transition)
- Rows 110-199: opacity 1.0 (mouth — Wav2Lip output)

The lateral feathering (left/right edges) is preserved from the existing implementation.

## 7. File Changes

### `core/blender.py`

The `blend_back` function signature changes from:

```python
def blend_back(frame, modified_crop, face_crop, feather_pixels=8) -> np.ndarray:
```

to:

```python
def blend_back(frame, modified_crop, face_crop, feather_pixels=8,
               mouth_only=False, mouth_top_ratio=0.4, mouth_blend_ratio=0.15) -> np.ndarray:
```

When `mouth_only=False` (default): behavior is identical to the current implementation.

When `mouth_only=True`: the mask is constructed using the vertical gradient described in Section 6, multiplied with the existing lateral feathering.

### `cli/main.py`

One line changes. In the lipsync frame loop:

```python
# Before:
frame = blend_back(frame, processed[0], face_crop)

# After:
frame = blend_back(frame, processed[0], face_crop, mouth_only=True)
```

### `tests/test_blender.py`

~7 new tests added to the existing file:

- `test_mouth_only_false_matches_original_behavior`
- `test_mouth_only_top_of_face_is_original`
- `test_mouth_only_bottom_of_face_is_modified`
- `test_mouth_only_transition_zone_is_blended`
- `test_mouth_only_outside_bbox_unchanged`
- `test_mouth_only_small_bbox`
- `test_mouth_only_returns_copy`

Tests use synthetic numpy arrays with known values (frame=100, crop=200). Assertions check pixel value ranges in specific row zones of the result.

## 8. Testing Strategy

All tests are deterministic, use synthetic data, and require no model downloads.

**Existing 8 blender tests:** unchanged, continue to pass (they use the default `mouth_only=False`).

**New ~7 tests:** verify the mouth-only mask behavior:
- Upper rows of the bbox match the original frame value (≈100).
- Lower rows match the modified crop value (≈200).
- Transition rows have intermediate values (between 100 and 200).
- Outside the bbox is unchanged.
- Small bbox does not crash.
- The original frame is not mutated.

**Test count:** 175 existing → ~182 (without API key) or ~183 (with API key).

**Visual verification:** after all tests pass, run the full pipeline on the Veo clip and compare visually against the Milestone 3b output. The eyes should be noticeably sharper; the mouth should look the same; the nose/transition zone should be seamless.

## 9. What Success Looks Like

After implementation:

- Running `--lipsync --model wav2lip` on the Veo clip produces a video where the upper face (eyes, eyebrows, forehead) is pixel-sharp from the original, the mouth moves in sync with the audio (Wav2Lip output), and the transition at the nose bridge is visually seamless.
- All existing tests pass unchanged (backward compatibility).
- ~7 new tests verify the mouth-only mask.
- The wall time is negligibly different from Milestone 3b (~35s for the Veo clip — the mask construction is microseconds of numpy).
- No new dependencies, no new model downloads, no new CLI flags.

## 10. Future Work (Not in Scope)

- **GFPGAN / face restoration on just the mouth zone.** If the 96x96 upscaled mouth still looks too blurry after this change, a face restoration model can be applied to just the lower half of the crop before blending. This would be a much smaller restoration task than restoring the full face (Section 1 explains why).
- **Landmark-based mask positioning.** Replace fixed `mouth_top_ratio` / `mouth_blend_ratio` with values computed from InsightFace's 68-point landmarks. More precise for non-frontal or unusual framing.
- **Color correction at the transition boundary.** If Wav2Lip produces a slightly different skin tone than the original (lighting/color shift), the transition zone can reveal a subtle color seam. A histogram-matching step between the Wav2Lip output and the original in the transition zone would fix this. Not needed for the Veo clip (Wav2Lip's color is close enough).
