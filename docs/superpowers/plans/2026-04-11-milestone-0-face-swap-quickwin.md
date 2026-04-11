# Milestone 0: Face-Swap Quick Win — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single Python CLI script (`swap.py`) on macOS M4 that takes a reference face image and a video and produces a video with that face swapped in, using `inswapper_128` from InsightFace. Target completion: one afternoon.

**Architecture:** One script, ~150 lines. Uses `insightface` (FaceAnalysis + inswapper model) for detection and swapping, `opencv-python` for frame I/O, `ffmpeg` (system binary) via `subprocess` for audio extraction and re-muxing. Runs `onnxruntime` with the CoreML execution provider on Apple Silicon, with automatic fallback to CPU. No training, no quality enhancement, no temporal coherence — raw baseline.

**Tech Stack:** Python 3.11, uv, onnxruntime (CoreML), insightface, opencv-python, numpy, ffmpeg.

**Repo:** `~/Projects/lipsync-corrector` (already initialized with the design spec and a first commit).

---

## File Structure (end state of this milestone)

```
lipsync-corrector/
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
├── swap.py                          # main script (Track A)
├── models/
│   ├── .gitkeep
│   └── inswapper_128.onnx           # downloaded, git-ignored
├── examples/
│   ├── .gitkeep
│   ├── reference.jpg                # user-provided, git-ignored
│   ├── input.mp4                    # user-provided, git-ignored
│   └── output.mp4                   # generated, git-ignored
├── tests/
│   └── test_swap.py
└── docs/
    ├── superpowers/
    │   ├── specs/
    │   │   └── 2026-04-11-lipsync-corrector-design.md   # already exists
    │   └── plans/
    │       └── 2026-04-11-milestone-0-face-swap-quickwin.md  # this file
    └── milestones/
        └── milestone-0.md            # written in the last task
```

**Responsibility per file:**

- `swap.py` — single-file implementation of the face-swap CLI. Contains: provider selection, argparse, reference embedding extraction, video frame iteration, frame-by-frame swap, audio extraction, re-mux, main().
- `tests/test_swap.py` — pytest unit tests for the non-ML-heavy helpers (provider selection, argument parsing, error handling).
- `pyproject.toml` — project metadata, Python version pin, dependencies managed by uv.
- `.gitignore` — excludes `.venv`, `__pycache__`, `models/*.onnx`, `examples/*.mp4`, `examples/*.jpg`, etc.
- `README.md` — one screen, just how to install and run.
- `docs/milestones/milestone-0.md` — written at the end. What was done, what the output looked like, observed defects, next steps.

---

## Prerequisites (before Task 1)

- `uv` installed. Check with `uv --version`. If missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- `ffmpeg` installed. Check with `ffmpeg -version`. If missing: `brew install ffmpeg`.
- Python 3.11 available via uv (uv will download if not present).
- You have two files ready to drop into `examples/` at Task 7:
  - `reference.jpg` — a clear frontal face photo of whoever you want to swap IN.
  - `input.mp4` — a short (10–30s) video clip with one person talking to camera.

---

## Task 1: Bootstrap project files

**Files:**
- Create: `~/Projects/lipsync-corrector/pyproject.toml`
- Create: `~/Projects/lipsync-corrector/.gitignore`
- Create: `~/Projects/lipsync-corrector/README.md`
- Create: `~/Projects/lipsync-corrector/models/.gitkeep`
- Create: `~/Projects/lipsync-corrector/examples/.gitkeep`
- Create: `~/Projects/lipsync-corrector/tests/__init__.py` (empty)

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "lipsync-corrector"
version = "0.0.1"
description = "Lip-sync corrector and face-swap experiments on Apple Silicon"
requires-python = "==3.11.*"
dependencies = [
    "insightface>=0.7.3",
    "onnxruntime>=1.17.0",
    "opencv-python>=4.9.0",
    "numpy>=1.26,<2.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Note: `numpy<2.0` because `insightface` and older onnxruntime builds do not yet play nicely with numpy 2.x. If you hit issues later, relax this.

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
*.egg-info/

# uv
.uv/

# Project artifacts
models/*.onnx
models/*.pt
models/*.pth
examples/*.mp4
examples/*.mov
examples/*.jpg
examples/*.jpeg
examples/*.png
examples/*.wav
examples/*.mp3

# macOS
.DS_Store
```

- [ ] **Step 3: Create `README.md`**

```markdown
# lipsync-corrector

Side project. Two tracks:

- **Track A (Milestone 0):** face-swap quick win. A single script that swaps a reference face into a video.
- **Track B (Milestone 1+):** lip-sync corrector for auto-dubbed YouTube videos.

See `docs/superpowers/specs/` for the design spec.

## Requirements

- macOS on Apple Silicon (tested on M4).
- `ffmpeg` (install with `brew install ffmpeg`).
- `uv` (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`).

## Setup

```bash
uv sync
# Download the inswapper model (see docs/milestones/milestone-0.md)
```

## Usage (Track A)

```bash
uv run python swap.py \
  --face examples/reference.jpg \
  --video examples/input.mp4 \
  --output examples/output.mp4 \
  --max-seconds 15
```
```

- [ ] **Step 4: Create empty placeholder files**

```bash
cd ~/Projects/lipsync-corrector
mkdir -p models examples tests
touch models/.gitkeep examples/.gitkeep tests/__init__.py
```

- [ ] **Step 5: Verify tree**

```bash
cd ~/Projects/lipsync-corrector
ls -la
```

Expected to see: `.git/`, `.gitignore`, `README.md`, `docs/`, `examples/`, `models/`, `pyproject.toml`, `tests/`.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add .gitignore README.md pyproject.toml models/.gitkeep examples/.gitkeep tests/__init__.py
git commit -m "chore: bootstrap project structure and pyproject.toml"
```

---

## Task 2: Install dependencies and download the inswapper model

**Files:**
- Modify: `~/Projects/lipsync-corrector/uv.lock` (generated)
- Create: `~/Projects/lipsync-corrector/models/inswapper_128.onnx` (downloaded, git-ignored)

- [ ] **Step 1: Sync Python environment with uv**

```bash
cd ~/Projects/lipsync-corrector
uv sync
```

Expected: uv downloads Python 3.11 if needed, creates `.venv/`, installs all dependencies, writes `uv.lock`. This may take 2–5 minutes the first time because `onnxruntime` and `insightface` are not tiny.

If you see a build error for `insightface` related to `Cython`, run `uv pip install cython` then retry `uv sync`. Some older `insightface` wheels need it at build time on Python 3.11 ARM.

- [ ] **Step 2: Verify the imports work**

```bash
cd ~/Projects/lipsync-corrector
uv run python -c "import insightface, onnxruntime, cv2, numpy; print('insightface', insightface.__version__); print('onnxruntime', onnxruntime.__version__); print('providers', onnxruntime.get_available_providers())"
```

Expected output: version strings and a providers list that includes `CoreMLExecutionProvider` and `CPUExecutionProvider`. If `CoreMLExecutionProvider` is missing, it is fine — the script will fall back to CPU. Note this in `docs/milestones/milestone-0.md` at the end.

- [ ] **Step 3: Download `inswapper_128.onnx`**

```bash
cd ~/Projects/lipsync-corrector
curl -L -o models/inswapper_128.onnx \
  https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
```

Expected: file of roughly 554 MB. Verify:

```bash
ls -lh models/inswapper_128.onnx
```

If the hugging face link 404s (mirrors move over time), alternatives:
- `https://huggingface.co/datasets/deepghs/inswapper/resolve/main/inswapper_128.onnx`
- Search "inswapper_128.onnx huggingface" and pick any copy of the same file.

Do **not** commit this file — it is git-ignored.

- [ ] **Step 4: Trigger insightface detector download**

The detection model (`buffalo_l`) auto-downloads on first use. Trigger it now so it is out of the way:

```bash
cd ~/Projects/lipsync-corrector
uv run python -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print('buffalo_l ready')
"
```

Expected: it downloads ~280 MB into `~/.insightface/models/buffalo_l/` the first time, then prints `buffalo_l ready`. Subsequent runs are instant.

- [ ] **Step 5: Commit the lockfile**

```bash
cd ~/Projects/lipsync-corrector
git add pyproject.toml uv.lock
git commit -m "chore: lock Python dependencies with uv"
```

---

## Task 3: Provider selection helper + unit test

**Files:**
- Create: `~/Projects/lipsync-corrector/swap.py`
- Create: `~/Projects/lipsync-corrector/tests/test_swap.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_swap.py`:

```python
from swap import select_providers


def test_select_providers_prefers_coreml_when_available():
    available = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert select_providers(available) == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_select_providers_falls_back_to_cpu_when_coreml_missing():
    available = ["CPUExecutionProvider"]
    assert select_providers(available) == ["CPUExecutionProvider"]


def test_select_providers_always_includes_cpu_as_last_resort():
    available = ["CoreMLExecutionProvider"]
    result = select_providers(available)
    assert result[-1] == "CPUExecutionProvider"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_swap.py -v
```

Expected: ImportError / collection error because `swap.py` does not exist yet.

- [ ] **Step 3: Create `swap.py` with the minimal implementation**

```python
"""Track A: Face-swap quick win. See docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md."""

from __future__ import annotations


def select_providers(available: list[str]) -> list[str]:
    preferred = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    result = [p for p in preferred if p in available]
    if "CPUExecutionProvider" not in result:
        result.append("CPUExecutionProvider")
    return result
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_swap.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add swap.py tests/test_swap.py
git commit -m "feat: add provider selection helper with tests"
```

---

## Task 4: Argument parsing with validation + unit test

**Files:**
- Modify: `~/Projects/lipsync-corrector/swap.py`
- Modify: `~/Projects/lipsync-corrector/tests/test_swap.py`

- [ ] **Step 1: Add failing tests for argparse**

Append to `tests/test_swap.py`:

```python
import pytest
from swap import parse_args


def test_parse_args_accepts_required_arguments():
    args = parse_args([
        "--face", "a.jpg",
        "--video", "b.mp4",
        "--output", "c.mp4",
    ])
    assert args.face == "a.jpg"
    assert args.video == "b.mp4"
    assert args.output == "c.mp4"
    assert args.max_seconds is None


def test_parse_args_accepts_max_seconds():
    args = parse_args([
        "--face", "a.jpg",
        "--video", "b.mp4",
        "--output", "c.mp4",
        "--max-seconds", "15",
    ])
    assert args.max_seconds == 15


def test_parse_args_rejects_missing_required(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--face", "a.jpg"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_swap.py -v
```

Expected: 3 new failures (`ImportError: cannot import name 'parse_args'`).

- [ ] **Step 3: Implement `parse_args` in `swap.py`**

Add these imports at the top of `swap.py`:

```python
import argparse
from typing import Optional
```

Then append this function to `swap.py`:

```python
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="swap.py",
        description="Face-swap a reference face into a video (Track A, Milestone 0).",
    )
    parser.add_argument("--face", required=True, help="Path to the reference face image.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--output", required=True, help="Path to the output video.")
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Optional cap: process only the first N seconds of the video.",
    )
    return parser.parse_args(argv)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_swap.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add swap.py tests/test_swap.py
git commit -m "feat: add CLI argument parser with tests"
```

---

## Task 5: Reference embedding + face analyzer + swapper loading

**Files:**
- Modify: `~/Projects/lipsync-corrector/swap.py`

No unit tests here: this code loads ML models and requires real image data, so we verify it manually at the end of the task with a one-line smoke check against `reference.jpg`. This is a deliberate departure from pure TDD for the ML layers, as discussed in the spec.

- [ ] **Step 1: Add the imports and constants**

At the top of `swap.py`, after the existing imports, add:

```python
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

INSWAPPER_PATH = Path(__file__).parent / "models" / "inswapper_128.onnx"
DET_SIZE = (640, 640)
```

- [ ] **Step 2: Add the `build_face_analyzer` function**

Append to `swap.py`:

```python
def build_face_analyzer(providers: list[str]) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    return app
```

- [ ] **Step 3: Add the `load_swapper` function**

Append to `swap.py`:

```python
def load_swapper(providers: list[str]):
    if not INSWAPPER_PATH.exists():
        raise FileNotFoundError(
            f"inswapper model not found at {INSWAPPER_PATH}. "
            "Download it with: curl -L -o models/inswapper_128.onnx "
            "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        )
    return get_model(str(INSWAPPER_PATH), providers=providers)
```

- [ ] **Step 4: Add the `extract_reference_face` function**

Append to `swap.py`:

```python
def extract_reference_face(face_analyzer: FaceAnalysis, image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read reference image: {image_path}")
    faces = face_analyzer.get(image)
    if not faces:
        raise ValueError(f"No face detected in reference image: {image_path}")
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0]
```

Rationale: sort by bounding-box area and pick the largest face, so if the reference photo happens to have more than one person the primary subject wins.

- [ ] **Step 5: Smoke-test the loaders manually**

You need a `reference.jpg` in `examples/` for this to succeed. Drop in any frontal face photo.

```bash
cd ~/Projects/lipsync-corrector
uv run python -c "
from pathlib import Path
from swap import build_face_analyzer, load_swapper, extract_reference_face, select_providers
import onnxruntime
providers = select_providers(onnxruntime.get_available_providers())
print('Using providers:', providers)
analyzer = build_face_analyzer(providers)
swapper = load_swapper(providers)
face = extract_reference_face(analyzer, Path('examples/reference.jpg'))
print('Detected face bbox:', face.bbox.tolist())
print('Embedding shape:', face.normed_embedding.shape)
print('OK')
"
```

Expected: prints the providers, a bounding box, `(512,)`, and `OK`. If you get `No face detected`, use a clearer frontal photo.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add swap.py
git commit -m "feat: load face analyzer, inswapper, and reference embedding"
```

---

## Task 6: Video I/O and frame swap helpers

**Files:**
- Modify: `~/Projects/lipsync-corrector/swap.py`

- [ ] **Step 1: Add the `open_video` helper**

Append to `swap.py`:

```python
def open_video(video_path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, width, height, total
```

- [ ] **Step 2: Add the `swap_one_frame` helper**

Append to `swap.py`:

```python
def swap_one_frame(frame, face_analyzer: FaceAnalysis, swapper, source_face) -> np.ndarray:
    faces = face_analyzer.get(frame)
    if not faces:
        return frame
    result = frame
    for target_face in faces:
        result = swapper.get(result, target_face, source_face, paste_back=True)
    return result
```

Note: `paste_back=True` composites the swapped face directly onto the original frame. Multiple faces per frame are all swapped to the same reference — crude but correct for the quick-win.

- [ ] **Step 3: Add `process_video` that writes frames to an intermediate mp4 (no audio yet)**

Append to `swap.py`:

```python
def process_video(
    video_path: Path,
    intermediate_path: Path,
    face_analyzer: FaceAnalysis,
    swapper,
    source_face,
    max_seconds: Optional[int],
) -> tuple[int, float]:
    cap, fps, width, height, total = open_video(video_path)

    frame_limit = total
    if max_seconds is not None:
        frame_limit = min(total, int(fps * max_seconds))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(intermediate_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for {intermediate_path}")

    import time
    start = time.perf_counter()
    processed = 0
    try:
        while processed < frame_limit:
            ok, frame = cap.read()
            if not ok:
                break
            swapped = swap_one_frame(frame, face_analyzer, swapper, source_face)
            writer.write(swapped)
            processed += 1
            if processed % 30 == 0:
                elapsed = time.perf_counter() - start
                print(f"  frame {processed}/{frame_limit}  ({processed / elapsed:.1f} fps)")
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start
    return processed, elapsed
```

- [ ] **Step 4: Run the tests to confirm nothing broke**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_swap.py -v
```

Expected: 6 passed, no collection errors (the new functions do not have tests, but they must import cleanly).

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add swap.py
git commit -m "feat: video open, frame swap, and intermediate-video writer"
```

---

## Task 7: Audio extraction and re-mux via ffmpeg

**Files:**
- Modify: `~/Projects/lipsync-corrector/swap.py`

- [ ] **Step 1: Add the ffmpeg helpers**

At the top of `swap.py`, add to the imports:

```python
import subprocess
import shutil
import tempfile
```

Append these functions to `swap.py`:

```python
def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install with: brew install ffmpeg")


def extract_audio(video_path: Path, audio_out: Path) -> bool:
    """Extract the audio track to a temp file. Returns False if the video has no audio."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    if "audio" not in probe.stdout:
        return False
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
         "-vn", "-acodec", "copy", str(audio_out)],
        check=True,
    )
    return True


def mux_video_audio(video_in: Path, audio_in: Path, output: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(video_in), "-i", str(audio_in),
         "-c:v", "copy", "-c:a", "copy",
         "-map", "0:v:0", "-map", "1:a:0",
         "-shortest", str(output)],
        check=True,
    )


def copy_video_only(video_in: Path, output: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_in), "-c", "copy", str(output)],
        check=True,
    )
```

Rationale: if the input video has no audio (some clips do not), we skip muxing and just copy the intermediate. `-shortest` on the mux protects us if the audio is slightly longer than the processed portion (which happens when `--max-seconds` is used).

- [ ] **Step 2: Smoke-test extract_audio against a real video**

You need `examples/input.mp4` for this step. Drop in a short clip (10–30s) with someone talking to camera.

```bash
cd ~/Projects/lipsync-corrector
uv run python -c "
from pathlib import Path
from swap import ensure_ffmpeg, extract_audio
ensure_ffmpeg()
had_audio = extract_audio(Path('examples/input.mp4'), Path('/tmp/test_audio.aac'))
print('had audio:', had_audio)
import os; print('size:', os.path.getsize('/tmp/test_audio.aac') if had_audio else 'N/A')
"
```

Expected: `had audio: True` and a nonzero file size. If False, the clip is silent and the mux step will be skipped later — still fine.

- [ ] **Step 3: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add swap.py
git commit -m "feat: audio extract and re-mux via ffmpeg"
```

---

## Task 8: Wire `main()` and run end-to-end

**Files:**
- Modify: `~/Projects/lipsync-corrector/swap.py`
- Create: `~/Projects/lipsync-corrector/docs/milestones/milestone-0.md`

- [ ] **Step 1: Add `main()` to `swap.py`**

Append to `swap.py`:

```python
def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    ensure_ffmpeg()

    face_path = Path(args.face)
    video_path = Path(args.video)
    output_path = Path(args.output)

    if not face_path.exists():
        print(f"error: reference face not found: {face_path}", file=sys.stderr)
        return 1
    if not video_path.exists():
        print(f"error: input video not found: {video_path}", file=sys.stderr)
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)

    providers = select_providers(onnxruntime.get_available_providers())
    print(f"Using providers: {providers}")

    print("Loading face analyzer...")
    face_analyzer = build_face_analyzer(providers)

    print("Loading swapper model...")
    swapper = load_swapper(providers)

    print(f"Extracting reference face from {face_path}...")
    source_face = extract_reference_face(face_analyzer, face_path)

    with tempfile.TemporaryDirectory(prefix="swap-") as tmp:
        tmp_dir = Path(tmp)
        intermediate = tmp_dir / "video_no_audio.mp4"
        audio = tmp_dir / "audio.aac"

        print(f"Processing {video_path} -> intermediate (no audio)...")
        processed, elapsed = process_video(
            video_path, intermediate, face_analyzer, swapper, source_face, args.max_seconds,
        )
        print(f"Processed {processed} frames in {elapsed:.1f}s ({processed / max(elapsed, 1e-6):.2f} fps)")

        print("Muxing audio...")
        had_audio = extract_audio(video_path, audio)
        if had_audio:
            mux_video_audio(intermediate, audio, output_path)
        else:
            print("  (input has no audio, copying video stream)")
            copy_video_only(intermediate, output_path)

    print(f"Done. Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the tests one more time to confirm imports still clean**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_swap.py -v
```

Expected: 6 passed.

- [ ] **Step 3: Run the script end-to-end on a short clip**

Make sure `examples/reference.jpg` and `examples/input.mp4` are in place. Then:

```bash
cd ~/Projects/lipsync-corrector
uv run python swap.py \
  --face examples/reference.jpg \
  --video examples/input.mp4 \
  --output examples/output.mp4 \
  --max-seconds 10
```

Expected: the script prints provider choice, model loading messages, per-30-frame progress, then `Done. Output: examples/output.mp4`. Total time for ~10s of 1080p video on M4: roughly 30–90 seconds depending on face count and provider.

- [ ] **Step 4: Open the output video and look at it**

```bash
open examples/output.mp4
```

Watch the clip. Confirm the face is swapped. Take mental notes on: flicker, boundary artifacts, lighting mismatch, failure modes (profile angles, occlusion, multiple faces).

If the script crashes or the output is empty, the most common causes are:
- `inswapper_128.onnx` not in `models/` → re-run Task 2 Step 3.
- `CoreMLExecutionProvider` crashing on a particular op → force CPU only with `CUDA_VISIBLE_DEVICES=""` or, simpler, modify `select_providers` temporarily to return `["CPUExecutionProvider"]` and re-run.
- Input video has an unusual codec the OpenCV writer cannot handle → re-encode the input first with `ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 examples/input.mp4`.

- [ ] **Step 5: Write milestone notes**

Create `docs/milestones/milestone-0.md`:

```markdown
# Milestone 0: Face-Swap Quick Win

**Date completed:** 2026-04-11
**Track:** A
**Status:** Done

## What was built

A single CLI script (`swap.py`) that takes a reference face image and a video,
and produces a video with the face swapped using `inswapper_128` from InsightFace.

## How to run

```bash
uv run python swap.py \
  --face examples/reference.jpg \
  --video examples/input.mp4 \
  --output examples/output.mp4 \
  --max-seconds 15
```

## Environment on the test machine

- Apple M4, macOS <version>
- Python 3.11, uv <version>
- onnxruntime providers available: <fill in from actual run>
- Providers used by the script: <fill in>

## Measured performance

- Input clip: <duration>s, <resolution>, <fps> fps
- Processing time: <seconds>
- Effective fps: <fps>
- Faces per frame: <mostly 1 / mostly 2 / variable>

## Observed quality and defects

<Fill in after watching the output video. Expected things to note:>
- Flicker between frames (temporal incoherence).
- Visible edges at the face boundary in some frames.
- Lighting of the swapped face does not match the scene.
- Failure modes: profile shots, occluded face, fast motion.

## What was learned

- <one or two bullets: what surprised you, what confirmed the spec's assumptions>

## Next milestone

Milestone 1: Track B setup and `video_io` module. See the design spec
`docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md` section 8.

The concrete next action when returning to the project:

```bash
cd ~/Projects/lipsync-corrector
cat docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md
# Then open a new plan session for Milestone 1.
```
```

Fill in the actual measurements and observations while they are fresh. This is the whole point of milestone notes: when you come back in three weeks, you need to know what worked and what did not without re-running the experiment.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add swap.py docs/milestones/milestone-0.md
git commit -m "feat: end-to-end face-swap pipeline and milestone-0 notes"
```

- [ ] **Step 7: Final sanity check**

```bash
cd ~/Projects/lipsync-corrector
git log --oneline
ls -la examples/ models/
```

Expected: a clean commit history of 7 commits on top of the initial spec commit, `examples/output.mp4` present, `models/inswapper_128.onnx` present but git-ignored.

---

## Done criteria for Milestone 0

- `uv run pytest` passes (6 tests).
- `uv run python swap.py --face ... --video ... --output ... --max-seconds 10` produces a real output video.
- You have watched the output video and recorded observations in `docs/milestones/milestone-0.md`.
- Everything is committed.

Milestone 1 (Track B: video I/O module) is out of scope for today. Do not start it in the same session — it deserves its own plan.
