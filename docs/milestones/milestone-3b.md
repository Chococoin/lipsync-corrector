# Milestone 3b: Wav2Lip Integration

**Date completed:** 2026-04-14
**Track:** B
**Status:** Done

## What was built

- `core/wav2lip/` — vendored Wav2Lip nn.Module (`model.py`, `conv.py`) from
  the Rudrabha reference repo, plus audio preprocessing (`audio.py`) with
  the exact training hparams and frame↔mel alignment math (`frame_sync.py`).
- `core/wav2lip_model.py` — `Wav2LipModel(LipSyncModel)` that loads the
  pretrained `wav2lip_gan.pth` checkpoint and runs batched inference on MPS.
- `cli/main.py` — `--model {identity,wav2lip}` flag wiring the real model
  into the existing `--lipsync` pipeline built in Milestone 3a. Default is
  `identity` so tests pass without the checkpoint file present.
- 27 new tests passing (3 architecture + 9 audio + 7 frame sync + 4 inference
  + 4 CLI), for a total suite of 111 tests.

## How to run

1. Download `wav2lip_gan.pth` to `models/wav2lip_gan.pth` (see README).
2. Run:

```bash
uv run python -m cli.main \
  --video examples/input.mp4 \
  --audio examples/dubbed.wav \
  --output examples/out.mp4 \
  --lipsync --model wav2lip
```

## Measured results

End-to-end run on the Veo-generated clip (192 frames, 24 fps, 1280x720)
using the clip's own original audio re-encoded to 16 kHz mono as the
"dubbed" track:

- Frame count preserved: **yes** (192 → 192)
- fps preserved: **yes** (24.0 → 24.0)
- Resolution preserved: **yes** (1280x720 → 1280x720)
- Audio preserved: **yes**
- Wall time: **35.8 s** (~5.4 frames/sec effective)
- Device: **MPS** (PyTorch) for Wav2Lip; **CoreML** for InsightFace tracker
- Output file size: 4.6 MB

### Subjective quality

- Mouth movement is clearly in sync with the audio — the model is doing its
  job.
- No visible bbox seam at the face boundary — the feathered alpha blend from
  Milestone 3a cleanly hides the paste region.
- No artifacts outside the face region — blur/quality loss stays strictly
  within the bbox.
- The face is noticeably blurry compared to the original. This is the
  expected signature of Wav2Lip operating natively at 96x96 and being
  upscaled back to the ~300–400 px face bbox of the source video.
- Subjective quality didn't degrade outside the face, but the face clearly
  lost resolution.

## What was learned

- The whole pipeline works end-to-end with a real ML model. The scaffold
  from Milestone 3a (tracker → crop → model → blend → write) was the right
  bet — swapping `IdentityModel` for `Wav2LipModel` required zero changes
  outside of the new files and a single CLI flag.
- Torch ≥2.6 made `weights_only=True` the default for `torch.load`, which
  silently breaks loading legacy `.pth` dicts like `wav2lip_gan.pth`. Every
  Wav2Lip wrapper in the wild needs `weights_only=False` now.
- Vendoring the `nn.Module` from Rudrabha's master branch rather than
  installing it as a package was the right call: the class is stable
  enough to be a one-time copy, and it avoids pulling in the whole repo's
  CUDA-hardcoded training infrastructure.
- `monkeypatch.setattr` on a module-level default value doesn't affect
  function default arguments already bound at class definition time. The
  CLI had to read `DEFAULT_CHECKPOINT_PATH` from the module at call time
  rather than relying on the constructor's default argument for the
  error-path test to be patchable.
- MPS performance on M4 is fine for this size of pipeline — 5.4 fps on a
  1280x720 clip means a 1-minute video processes in roughly 3–4 minutes.
  Wav2Lip's small input (96x96) keeps the actual GAN forward cheap; the
  tracker and video I/O dominate wall time.
- Wav2Lip's resolution hit is real and visible. The next milestone (4) is
  specifically for improving perceptual quality — blending, possibly
  upscaling, maybe a face-restoration pass.

## Deferred to Milestone 4

- Resolution/sharpness recovery inside the face region — this is the main
  perceptual defect of the baseline.
- Face alignment via landmarks (currently bbox-only). Wav2Lip is forgiving
  on near-frontal faces but rotation alignment would reduce edge cases.
- Silence detection to skip frames where audio is silent (avoids
  unnecessary model runs and potential artifacts on silent passages).
- Optional face-restoration pass (GFPGAN / CodeFormer).

## Next milestone

**Milestone 4:** blending pass. Reduce the resolution defect and improve
subjective quality inside the face region. Likely approaches: super-resolution
on the Wav2Lip output before blending back, or a lightweight face-restoration
pass over the bbox region.
