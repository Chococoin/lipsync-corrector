# Blending Pass: Mouth-Only Mask

**Date completed:** 2026-04-17
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
to not degrade them in the first place. Zero new dependencies, zero new
model downloads, zero additional wall time.

**Changes:**
- `core/blender.py` — `blend_back()` gained `mouth_only`, `mouth_top_ratio`,
  and `mouth_blend_ratio` parameters. A new `_build_mouth_only_mask()` helper
  constructs the vertical gradient mask. Default `mouth_only=False` preserves
  backward compatibility.
- `cli/main.py` — one line changed: passes `mouth_only=True` when `--lipsync`
  is active.
- 7 new tests in `test_blender.py` verifying the mask behavior, for a total
  suite of 182 tests (+ 1 skipped without API key).

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

- Frame count preserved: **yes** (192 → 192)
- Audio preserved: **yes**
- Wall time: **33 s** (vs ~35 s in milestone 3b — negligible difference,
  confirming the mask construction adds no measurable overhead)

### Visual comparison vs Milestone 3b

- **Eyes/forehead:** sharp, pixel-identical to the original video. This is
  the key improvement — in 3b these were noticeably blurry from the 96x96
  resize round-trip.
- **Mouth region:** same as 3b — Wav2Lip-generated lip-sync matching the
  audio. Still 96x96 upscaled resolution (some inherent blur from the model's
  native resolution), but now confined to a smaller area.
- **Nose transition zone:** seamless gradient — no visible line or boundary
  between original and Wav2Lip pixels.
- **Overall impression:** "excelente" (user's assessment). The combination of
  sharp original eyes + synced mouth + invisible transition makes the output
  look significantly more natural than the 3b version where the entire face
  was uniformly degraded.

## What was learned

- The biggest quality improvement came from the simplest possible change
  (a different mask shape) rather than the most complex one (adding a face
  restoration model). The user's question "why are we replacing the entire
  face when we only need the mouth?" cut through the over-engineering and
  identified the root cause: unnecessary pixel degradation.
- Fixed anatomical ratios (0.4 / 0.15) work well for conversational framing
  (webcam, YouTube, interviews). The nose is consistently at ~40% from the
  top of a face bbox in these scenarios. No landmark detection was needed.
- The wall time stayed constant (~33s vs ~35s) because the mask is a
  microsecond numpy operation — the bottleneck is tracker + Wav2Lip inference,
  not blending.
- Backward compatibility was trivially preserved via the default parameter
  (`mouth_only=False`). All 8 original blender tests pass unchanged.

## Deferred

- GFPGAN face restoration on the mouth zone only (if the 96x96 mouth
  blur is still too noticeable for specific use cases).
- Landmark-based mask positioning (replaces fixed ratios with detected
  facial feature positions for non-frontal faces).
- Color correction at the transition boundary (if skin tone mismatch
  becomes visible in some clips).

## Next milestone

Evaluate whether GFPGAN is still needed for the mouth zone, or proceed
to the TTS sub-project to complete the auto-dub pipeline.
