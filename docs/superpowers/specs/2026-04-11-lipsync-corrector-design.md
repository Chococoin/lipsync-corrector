# Lip-Sync Corrector — Design Spec

**Date:** 2026-04-11
**Status:** Approved for implementation (Milestone 0 first)
**Author:** chocos (with Claude)

## 1. Purpose

Build a side project that lets the author experiment with modern face-swap and lip-sync ML, and grows into a shareable open-source tool that **corrects lip-sync on auto-dubbed YouTube videos**.

Target user: content creators who use auto-dubbing (YouTube Auto-Dub, ElevenLabs, Rask, HeyGen) to translate their videos into other languages. These tools produce good audio but leave the original mouth movements untouched, creating a visible uncanny valley. This project fills that gap: given the original video and the already-dubbed audio, produce a video where the mouth matches the new audio.

Explicit non-goals:
- We do not translate text.
- We do not clone or generate voices.
- We do not replace face identity (that belongs to the face-swap side track).
- We do not attempt cinema-grade VFX quality.

## 2. Constraints

- **Hardware:** Apple Silicon M4 only. Backend is Metal/MPS via PyTorch, ONNX Runtime with CoreML provider, or MLX. No CUDA.
- **Training:** out of scope locally. Inference only on M4. Any fine-tuning happens in rented cloud GPUs, not on the laptop.
- **Time budget:** bursty side project. Weekends and occasional evenings. Must survive multi-week gaps without losing context.
- **Language:** Python 3.11 for the core. Rust is deferred — only considered later for specific performance-critical components if a real bottleneck appears. Rewriting the whole pipeline in Rust gives near-zero speedup because GPU kernels dominate runtime.
- **Form factor:** local CLI first. Architecture must not foreclose a future SaaS/API wrapping.

## 3. Project Shape: Two Tracks, One Repo

### Track A — Face-swap quick win

Scope: a single CLI script that takes a reference face image and a video, and outputs the video with the face replaced. Built in one afternoon using off-the-shelf pre-trained models. No quality polish, no temporal coherence work, no restoration.

Purpose: scratch the face-swap itch, validate the M4 environment end-to-end, gain first-hand intuition for where face-swap breaks, and build the shared infrastructure (Python env, ffmpeg, face detection, video I/O) that Track B reuses.

### Track B — Lip-sync corrector (the real project)

Scope: a CLI tool that accepts `video_original.mp4` and `audio_dubbed.{wav,mp3,m4a}`, and produces `video_corrected.mp4` where the speaker's mouth matches the dubbed audio. Everything else in the frame is preserved.

Assumption: the dubbed audio already has the same duration as the original. Time-stretching to handle mismatched durations is out of scope for the initial release.

Future extensibility: the engine lives in `core/` and the CLI in `cli/`. Wrapping `core/` in a web API or SaaS later requires adding `api/` or `web/`, not rewriting the engine.

## 4. Technical Stack

**Shared across both tracks:**

- Python 3.11 in an isolated environment managed with `uv`.
- `ffmpeg` (system-level, via Homebrew) for video mux/demux and audio extraction.
- `opencv-python` for frame-level video I/O and basic image ops.
- `numpy` as the shared array format.
- `onnxruntime` with CoreML execution provider for ONNX model inference.
- `insightface` for face detection, alignment, and the face-swap model.

**Track A only:**

- `inswapper_128.onnx` (~550 MB), the standard open-source face-swap model. Downloaded once, cached locally, git-ignored.

**Track B additional:**

- `torch` with MPS backend (for Wav2Lip and successors).
- `librosa` or `soundfile` for audio loading and resampling.
- Initial lip-sync model: **Wav2Lip**. Chosen because it is stable, widely implemented, MPS-compatible with minor patching, and serves as a known baseline. It is not state-of-the-art — that is deliberate. We replace it in a later milestone with LatentSync or MuseTalk once the pipeline is solid.

## 5. Track A Architecture (Milestone 0)

A single script, `swap.py`, roughly 100–200 lines. Usage:

```
python swap.py --face reference.jpg --video input.mp4 --output output.mp4 [--max-seconds N]
```

Pipeline:

1. **Load reference face.** Read `reference.jpg`, run `insightface` detection, extract the 512-dim identity embedding from the first detected face. If no face is detected, exit with a clear error.
2. **Open input video.** Read metadata (fps, resolution, frame count, duration) via OpenCV. Respect `--max-seconds` by computing a frame limit.
3. **Extract audio.** Use ffmpeg to pull the original audio stream into a temp file. We will re-mux it at the end.
4. **Process frames.** For each frame up to the limit: detect faces, and for each detected face run the inswapper with the reference embedding. Write the modified frame to an intermediate video file (no audio yet).
5. **Re-mux.** Use ffmpeg to combine the processed video with the original audio into `output.mp4`.
6. **Report.** Print total time, frames processed, effective fps, and output path.

Error handling:
- Missing input files → clear error message, exit 1.
- No face in reference → clear error, exit 1.
- CoreML provider init failure → fall back to CPU provider with a warning. Do not crash.
- Frame with no detected face → pass through unchanged. Do not crash.

Explicitly out of scope for Milestone 0:
- Temporal consistency / flicker reduction.
- Face restoration (GFPGAN, CodeFormer).
- Relighting.
- Handling multiple target faces or identity matching.
- Progress bars, fancy output, config files.

## 6. Track B Architecture

### 6.1 Module layout

```
lipsync-corrector/
├── core/
│   ├── __init__.py
│   ├── video_io.py        # open/write video, frame iteration, audio extraction, re-mux
│   ├── face_tracker.py    # detection + temporal tracking of the target face
│   ├── mouth_region.py    # crop and align the mouth region for the model
│   ├── lipsync_model.py   # thin wrapper around Wav2Lip / LatentSync / MuseTalk
│   ├── blender.py         # paste generated mouth back onto the original frame
│   └── pipeline.py        # orchestrates the full flow end-to-end
├── cli/
│   └── main.py            # CLI entry point, argument parsing, progress reporting
├── models/                # downloaded weights, git-ignored
├── examples/              # short test clips and reference audios
├── benchmarks/            # throughput and quality measurement scripts
├── docs/
│   ├── superpowers/specs/ # design docs (this file lives here)
│   └── milestones/        # one markdown per milestone, see section 8
└── pyproject.toml
```

Each module has one clear responsibility, a small public interface, and can be tested in isolation with a short example clip. Consumers of `core/` import `pipeline.process(video_path, audio_path, output_path, options)` and nothing else from the orchestrator layer.

### 6.2 Data flow

1. `video_io` opens `video.mp4`, yields frames, extracts the original (unused) audio for preservation if needed.
2. `face_tracker` runs detection on each frame and produces a stable per-frame bounding box for the primary speaker. It must handle brief detection gaps without jumping.
3. `mouth_region` takes each frame plus the tracked bbox and produces an aligned mouth crop plus the affine transform needed to paste it back.
4. `lipsync_model` takes the mouth crop sequence plus the target audio (resampled to 16 kHz mono) and produces new mouth crops synchronized to that audio.
5. `blender` takes each generated mouth crop and pastes it back onto the original frame using the stored affine transform, with feathered alpha blending at the seam.
6. `video_io` writes the composited frames to an intermediate video file and re-muxes with the **dubbed** audio.

### 6.3 Key design decisions

- **Audio normalization.** Inside `lipsync_model`, audio is always resampled to 16 kHz mono WAV regardless of input. Input format (`.wav`, `.mp3`, `.m4a`) is handled at the CLI boundary.
- **Duration assumption.** The dubbed audio is assumed to match the original video duration. If it does not, the tool errors out with a clear message. Time-stretching is explicitly deferred.
- **Single-speaker assumption.** The initial release targets videos with one primary speaker facing camera. Multi-speaker support is a future milestone.
- **Silence passthrough (future).** Frames that fall in silent sections of the dubbed audio can be passed through unchanged, saving compute and avoiding artifacts. Deferred to a post-baseline milestone.
- **Model abstraction.** `lipsync_model.py` exposes a `LipSyncModel` interface with `process(mouth_crops, audio) -> new_mouth_crops`. Concrete implementations (`Wav2LipModel`, `LatentSyncModel`, `MuseTalkModel`) are swappable. The CLI picks one via `--model` flag, default Wav2Lip.
- **Quality-by-layers strategy.** We ship an ugly baseline first, then improve one layer at a time: tracking stability, mouth alignment, blending quality, model upgrade. Each improvement is a milestone with a before/after video in `docs/milestones/`.

## 7. M4 / MPS Risks and Mitigations

- **Missing MPS ops.** Some PyTorch ops fall back to CPU silently, tanking performance. Mitigation: benchmark each milestone, log per-stage timing, and isolate bottlenecks before blaming the model.
- **Repos assuming CUDA.** Wav2Lip and LatentSync originals hardcode `cuda`. Mitigation: maintain a small `core/device.py` that picks MPS, CPU, or CUDA and patches model loading accordingly.
- **Memory unification is a double-edged sword.** On M4, model weights and video frames share the same RAM. Long clips at 4K can exhaust memory. Mitigation: stream frames in fixed-size batches, never load a whole video into memory.
- **Flash attention / xformers absent.** Mitigation: use stock attention. If a model requires flash attention, prefer a different model.
- **Training is impractical.** Mitigation: never put training in the critical path. If we ever need fine-tuning, rent cloud GPUs for a few hours.

## 8. Milestones (Bursty-Work Friendly)

Each milestone is self-contained, produces a visible artifact, and ends with a markdown note in `docs/milestones/milestone-N.md` recording what was done, what remains, and the exact next step. This lets the author return after weeks without re-reading the whole repo.

| # | Milestone | Est. | Observable output |
|---|---|---|---|
| 0 | Track A: face-swap quick win | 1 afternoon | Swapped video on M4 + first-hand intuition |
| 1 | Track B setup + `video_io` | 3–4 h | CLI reads a video, extracts frames, re-muxes with audio identically |
| 2 | `face_tracker` stable | 4–6 h | Video with a tracked bounding box overlay, no jitter |
| 3 | Wav2Lip integrated (baseline) | 4–6 h | First real lip-sync output, visibly imperfect |
| 4 | Blending pass | 4–6 h | Same video with no visible seams around the mouth |
| 5 | Benchmark + quality report | 2–4 h | Markdown report: fps, memory, perceptual notes |
| 6 | Upgrade to LatentSync or MuseTalk | 6–8 h | Side-by-side comparison with baseline |
| 7 | CLI polish + error handling | 3–4 h | Usable by a non-author |
| 8+ | Targeted improvements | bursts | Driven by what earlier milestones reveal |

Total time to first shareable release: roughly 30–40 hours, spread over whatever calendar time the author's bursts allow.

**Milestone discipline:** no milestone starts until the previous one has a completed note. No milestone is declared done until its observable output is captured (video, report, screenshot) in `docs/milestones/`.

## 9. What Success Looks Like

- **Short term (today):** A working face-swap script on M4, producing a real video, with the author having seen the actual quality and defects first-hand.
- **Medium term (first few bursts):** A lip-sync pipeline that takes a real YouTube clip and a dubbed audio and produces a plausibly-corrected video, even if imperfect.
- **Long term:** An open-source tool a few creators actually use to improve their auto-dubbed videos, documented well enough that others can contribute.

What would make this project *fail*: trying to outrun cinema VFX, trying to invent a new model architecture, trying to build a SaaS before the engine works, or abandoning it because a single milestone felt too big. The milestone sizing and bursty-work format are explicit mitigations against each of those failure modes.

## 10. Open Questions (Deferred, Not Blocking)

- Which specific Wav2Lip fork to use as the base (there are several; picked during Milestone 3).
- Exact blending algorithm (Poisson, alpha feather, learned) — picked during Milestone 4 after seeing baseline artifacts.
- License choice for open-sourcing — decided before Milestone 7.
- Whether to add a GUI (Gradio or similar) — decided after Milestone 7 based on user feedback.
