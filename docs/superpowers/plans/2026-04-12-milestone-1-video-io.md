# Milestone 1: Track B Setup + video_io — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the `core/` and `cli/` package structure for Track B (lip-sync corrector), implement `core/video_io.py` with robust video reading, writing, and audio handling, and build a CLI that does a lossless identity pass-through (video in → video out, unchanged). This proves the I/O foundation works before any ML is layered on top.

**Architecture:** `core/video_io.py` exposes `VideoReader` and `VideoWriter` context-manager classes plus ffmpeg-based audio helpers. `core/device.py` centralizes hardware selection (ONNX providers, PyTorch device) for all future milestones. `cli/main.py` is a thin CLI that orchestrates `core/` modules. The identity pass-through test is the acceptance criterion: the output video must match the input frame-for-frame (within codec tolerance) with audio preserved.

**Tech Stack:** Python 3.11, uv, opencv-python, numpy, ffmpeg (subprocess), pytest. No new dependencies.

**Repo:** `~/Projects/lipsync-corrector` on `main`, branching to `milestone-1`.

---

## File Structure (end state of this milestone)

```
lipsync-corrector/
├── core/
│   ├── __init__.py
│   ├── device.py              # hardware/provider selection
│   └── video_io.py            # VideoReader, VideoWriter, ffmpeg audio helpers
├── cli/
│   ├── __init__.py
│   └── main.py                # lip-sync CLI entry point (passthrough for now)
├── tests/
│   ├── __init__.py             # already exists
│   ├── conftest.py             # shared test fixtures (tiny test videos)
│   ├── test_swap.py            # already exists, untouched
│   ├── test_device.py          # tests for core/device.py
│   ├── test_video_io.py        # tests for core/video_io.py
│   └── test_cli.py             # tests for cli/main.py
├── swap.py                     # Track A, untouched
├── docs/milestones/
│   └── milestone-1.md          # written at the end
└── (everything else unchanged)
```

**Responsibility per new file:**

- `core/__init__.py` — empty, marks `core` as a package.
- `core/device.py` — `get_onnx_providers()` returns ordered provider list for onnxruntime. `get_torch_device()` returns the best available PyTorch device string. Single source of truth for hardware selection across the project.
- `core/video_io.py` — `VideoReader` (context manager, yields frames, exposes metadata), `VideoWriter` (context manager, writes frames), `ensure_ffmpeg()`, `extract_audio()`, `mux_video_audio()`, `has_audio_stream()`. All video I/O for Track B flows through this module.
- `cli/__init__.py` — empty, marks `cli` as a package.
- `cli/main.py` — `parse_args()`, `main()`. For Milestone 1, accepts `--video` and `--output`, reads all frames, writes them back, preserves audio. Future milestones add `--audio` (dubbed audio) and `--model` (lip-sync model choice).
- `tests/conftest.py` — pytest fixtures that generate tiny test videos (10 frames, 64x64) with and without audio. Shared across all test files. Auto-cleaned after each test session.

---

## Task 1: Core package structure + test fixtures

**Files:**
- Create: `~/Projects/lipsync-corrector/core/__init__.py`
- Create: `~/Projects/lipsync-corrector/cli/__init__.py`
- Create: `~/Projects/lipsync-corrector/tests/conftest.py`

- [ ] **Step 1: Create branch**

```bash
cd ~/Projects/lipsync-corrector
git checkout -b milestone-1
```

- [ ] **Step 2: Create package directories**

```bash
cd ~/Projects/lipsync-corrector
mkdir -p core cli
touch core/__init__.py cli/__init__.py
```

- [ ] **Step 3: Create `tests/conftest.py` with video fixtures**

```python
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def tmp_video(tmp_path: Path) -> Path:
    """Create a tiny 10-frame 64x64 video with no audio. Each frame is a different solid color."""
    video_path = tmp_path / "test_no_audio.mp4"
    fps = 10.0
    size = (64, 64)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, size)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 128),
    ]
    for color in colors:
        frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


@pytest.fixture
def tmp_video_with_audio(tmp_video: Path, tmp_path: Path) -> Path:
    """Take the no-audio video and add a silent audio track via ffmpeg."""
    output = tmp_path / "test_with_audio.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(tmp_video),
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(output),
        ],
        check=True,
    )
    return output
```

- [ ] **Step 4: Verify fixtures work**

```bash
cd ~/Projects/lipsync-corrector
uv run python -c "
from pathlib import Path
import tempfile, subprocess, cv2, numpy as np

tmp = Path(tempfile.mkdtemp())
video_path = tmp / 'test.mp4'
writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (64, 64))
for i in range(10):
    writer.write(np.full((64, 64, 3), (i * 25, 0, 0), dtype=np.uint8))
writer.release()
print('video created:', video_path.stat().st_size, 'bytes')

audio_path = tmp / 'with_audio.mp4'
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video_path), '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono', '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(audio_path)], check=True)
print('video+audio created:', audio_path.stat().st_size, 'bytes')
"
```

Expected: two files created, both nonzero.

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/__init__.py cli/__init__.py tests/conftest.py
git commit -m "chore: create core/ and cli/ packages with test video fixtures"
```

---

## Task 2: core/device.py with tests

**Files:**
- Create: `~/Projects/lipsync-corrector/core/device.py`
- Create: `~/Projects/lipsync-corrector/tests/test_device.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_device.py`:

```python
from core.device import get_onnx_providers, get_torch_device


def test_get_onnx_providers_returns_list():
    providers = get_onnx_providers()
    assert isinstance(providers, list)
    assert len(providers) >= 1


def test_get_onnx_providers_always_ends_with_cpu():
    providers = get_onnx_providers()
    assert providers[-1] == "CPUExecutionProvider"


def test_get_torch_device_returns_known_string():
    device = get_torch_device()
    assert device in ("mps", "cuda", "cpu")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_device.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `core/device.py`**

```python
from __future__ import annotations


def get_onnx_providers() -> list[str]:
    """Return ordered list of ONNX Runtime execution providers available on this machine."""
    import onnxruntime
    available = onnxruntime.get_available_providers()
    preferred = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    result = [p for p in preferred if p in available]
    if "CPUExecutionProvider" not in result:
        result.append("CPUExecutionProvider")
    return result


def get_torch_device() -> str:
    """Return the best available PyTorch device string: 'mps', 'cuda', or 'cpu'."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_device.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run all tests to confirm nothing broke**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: 9 passed (6 from test_swap + 3 new).

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/device.py tests/test_device.py
git commit -m "feat: add core/device.py for hardware/provider selection"
```

---

## Task 3: core/video_io.py — VideoReader

**Files:**
- Create: `~/Projects/lipsync-corrector/core/video_io.py`
- Create: `~/Projects/lipsync-corrector/tests/test_video_io.py`

- [ ] **Step 1: Write failing tests for VideoReader**

Create `tests/test_video_io.py`:

```python
import numpy as np
import pytest

from core.video_io import VideoReader


class TestVideoReader:
    def test_opens_valid_video(self, tmp_video):
        with VideoReader(tmp_video) as reader:
            assert reader.fps == pytest.approx(10.0, abs=0.5)
            assert reader.width == 64
            assert reader.height == 64
            assert reader.frame_count == 10

    def test_duration(self, tmp_video):
        with VideoReader(tmp_video) as reader:
            assert reader.duration == pytest.approx(1.0, abs=0.2)

    def test_iterates_all_frames(self, tmp_video):
        with VideoReader(tmp_video) as reader:
            frames = list(reader)
            assert len(frames) == 10
            assert all(f.shape == (64, 64, 3) for f in frames)
            assert all(f.dtype == np.uint8 for f in frames)

    def test_frames_have_different_content(self, tmp_video):
        with VideoReader(tmp_video) as reader:
            frames = list(reader)
            first = frames[0]
            last = frames[-1]
            assert not np.array_equal(first, last)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VideoReader(tmp_path / "nonexistent.mp4")

    def test_len(self, tmp_video):
        with VideoReader(tmp_video) as reader:
            assert len(reader) == 10

    def test_context_manager_releases(self, tmp_video):
        reader = VideoReader(tmp_video)
        reader.close()
        # After close, iterating should yield nothing or raise
        frames = list(reader)
        assert frames == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_video_io.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement VideoReader in `core/video_io.py`**

```python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    """Context manager for reading video frames and metadata."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video not found: {self._path}")
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self._path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration(self) -> float:
        return self._frame_count / self._fps if self._fps > 0 else 0.0

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def __len__(self) -> int:
        return self._frame_count

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_video_io.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Run all tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: 16 passed (6 swap + 3 device + 7 video_io).

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/video_io.py tests/test_video_io.py
git commit -m "feat: add VideoReader with frame iteration and metadata"
```

---

## Task 4: core/video_io.py — VideoWriter

**Files:**
- Modify: `~/Projects/lipsync-corrector/core/video_io.py`
- Modify: `~/Projects/lipsync-corrector/tests/test_video_io.py`

- [ ] **Step 1: Add failing tests for VideoWriter**

Append to `tests/test_video_io.py`:

```python
from core.video_io import VideoWriter


class TestVideoWriter:
    def test_writes_frames(self, tmp_path):
        output = tmp_path / "output.mp4"
        with VideoWriter(output, fps=10.0, width=64, height=64) as writer:
            for i in range(5):
                frame = np.full((64, 64, 3), (i * 50, 0, 0), dtype=np.uint8)
                writer.write(frame)
            assert writer.frames_written == 5
        assert output.exists()
        assert output.stat().st_size > 0

    def test_written_video_is_readable(self, tmp_path):
        output = tmp_path / "roundtrip.mp4"
        original_frames = []
        for i in range(5):
            original_frames.append(np.full((64, 64, 3), (i * 50, 0, 0), dtype=np.uint8))

        with VideoWriter(output, fps=10.0, width=64, height=64) as writer:
            for f in original_frames:
                writer.write(f)

        with VideoReader(output) as reader:
            read_frames = list(reader)
            assert len(read_frames) == 5

    def test_raises_on_invalid_path(self, tmp_path):
        bad_path = tmp_path / "nonexistent_dir" / "output.mp4"
        with pytest.raises(RuntimeError):
            VideoWriter(bad_path, fps=10.0, width=64, height=64)
```

- [ ] **Step 2: Run tests to verify new ones fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_video_io.py::TestVideoWriter -v
```

Expected: ImportError for VideoWriter.

- [ ] **Step 3: Implement VideoWriter in `core/video_io.py`**

Append to `core/video_io.py`:

```python
class VideoWriter:
    """Context manager for writing video frames to an mp4 file."""

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        self._path = Path(path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {self._path}")
        self._frames_written = 0

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
        self._frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_video_io.py -v
```

Expected: 10 passed (7 reader + 3 writer).

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/video_io.py tests/test_video_io.py
git commit -m "feat: add VideoWriter for frame-by-frame video output"
```

---

## Task 5: core/video_io.py — ffmpeg audio helpers

**Files:**
- Modify: `~/Projects/lipsync-corrector/core/video_io.py`
- Modify: `~/Projects/lipsync-corrector/tests/test_video_io.py`

- [ ] **Step 1: Add failing tests for audio helpers**

Append to `tests/test_video_io.py`:

```python
from core.video_io import ensure_ffmpeg, has_audio_stream, extract_audio, mux_video_audio


class TestFfmpegHelpers:
    def test_ensure_ffmpeg_does_not_raise(self):
        ensure_ffmpeg()

    def test_has_audio_stream_false_for_no_audio(self, tmp_video):
        assert has_audio_stream(tmp_video) is False

    def test_has_audio_stream_true_for_audio(self, tmp_video_with_audio):
        assert has_audio_stream(tmp_video_with_audio) is True

    def test_extract_audio_returns_false_for_no_audio(self, tmp_video, tmp_path):
        result = extract_audio(tmp_video, tmp_path / "audio.aac")
        assert result is False

    def test_extract_audio_returns_true_for_audio(self, tmp_video_with_audio, tmp_path):
        audio_out = tmp_path / "audio.aac"
        result = extract_audio(tmp_video_with_audio, audio_out)
        assert result is True
        assert audio_out.exists()
        assert audio_out.stat().st_size > 0

    def test_mux_video_audio_produces_output(self, tmp_video, tmp_video_with_audio, tmp_path):
        audio_out = tmp_path / "extracted.aac"
        extract_audio(tmp_video_with_audio, audio_out)
        muxed = tmp_path / "muxed.mp4"
        mux_video_audio(tmp_video, audio_out, muxed)
        assert muxed.exists()
        assert muxed.stat().st_size > 0
        assert has_audio_stream(muxed) is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_video_io.py::TestFfmpegHelpers -v
```

Expected: ImportError for the new functions.

- [ ] **Step 3: Implement ffmpeg helpers in `core/video_io.py`**

Append to `core/video_io.py`:

```python
def ensure_ffmpeg() -> None:
    """Raise RuntimeError if ffmpeg is not in PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install with: brew install ffmpeg")


def has_audio_stream(video_path: Path) -> bool:
    """Check whether a video file contains an audio stream."""
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    return "audio" in probe.stdout


def extract_audio(video_path: Path, audio_out: Path) -> bool:
    """Extract the audio track from a video. Returns False if video has no audio."""
    if not has_audio_stream(video_path):
        return False
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path), "-vn", "-acodec", "copy",
            str(audio_out),
        ],
        check=True,
    )
    return True


def mux_video_audio(video_in: Path, audio_in: Path, output: Path) -> None:
    """Combine a video file (no audio) with an audio file into a single output."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_in), "-i", str(audio_in),
            "-c:v", "copy", "-c:a", "copy",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", str(output),
        ],
        check=True,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_video_io.py -v
```

Expected: 16 passed (7 reader + 3 writer + 6 ffmpeg).

- [ ] **Step 5: Run all tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: 25 passed (6 swap + 3 device + 16 video_io).

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/video_io.py tests/test_video_io.py
git commit -m "feat: add ffmpeg audio helpers (extract, mux, has_audio_stream)"
```

---

## Task 6: cli/main.py — passthrough pipeline

**Files:**
- Create: `~/Projects/lipsync-corrector/cli/main.py`
- Create: `~/Projects/lipsync-corrector/tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cli.py`:

```python
import pytest

from cli.main import parse_args, main


class TestParseArgs:
    def test_accepts_required_args(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4"])
        assert args.video == "in.mp4"
        assert args.output == "out.mp4"

    def test_rejects_missing_video(self):
        with pytest.raises(SystemExit):
            parse_args(["--output", "out.mp4"])

    def test_rejects_missing_output(self):
        with pytest.raises(SystemExit):
            parse_args(["--video", "in.mp4"])

    def test_accepts_audio_arg(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4", "--audio", "dub.wav"])
        assert args.audio == "dub.wav"

    def test_audio_defaults_to_none(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4"])
        assert args.audio is None


class TestMainPassthrough:
    def test_passthrough_preserves_frame_count(self, tmp_video, tmp_path):
        output = tmp_path / "output.mp4"
        result = main(["--video", str(tmp_video), "--output", str(output)])
        assert result == 0
        assert output.exists()
        from core.video_io import VideoReader
        with VideoReader(output) as reader:
            assert reader.frame_count == 10

    def test_passthrough_preserves_audio(self, tmp_video_with_audio, tmp_path):
        output = tmp_path / "output.mp4"
        result = main(["--video", str(tmp_video_with_audio), "--output", str(output)])
        assert result == 0
        from core.video_io import has_audio_stream
        assert has_audio_stream(output) is True

    def test_returns_1_for_missing_video(self, tmp_path):
        result = main(["--video", str(tmp_path / "nope.mp4"), "--output", str(tmp_path / "out.mp4")])
        assert result == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_cli.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `cli/main.py`**

```python
"""Lip-sync corrector CLI — Track B entry point."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional

from core.video_io import (
    VideoReader,
    VideoWriter,
    ensure_ffmpeg,
    extract_audio,
    has_audio_stream,
    mux_video_audio,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lipsync",
        description="Lip-sync corrector for auto-dubbed videos.",
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--output", required=True, help="Path to the output video.")
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to the dubbed audio (not yet implemented, reserved for Milestone 3).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    ensure_ffmpeg()

    video_path = Path(args.video)
    output_path = Path(args.output)

    if not video_path.exists():
        print(f"error: video not found: {video_path}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with VideoReader(video_path) as reader:
        print(f"Input: {video_path} ({reader.frame_count} frames, {reader.fps:.1f} fps, {reader.width}x{reader.height})")

        with tempfile.TemporaryDirectory(prefix="lipsync-") as tmp:
            tmp_dir = Path(tmp)
            intermediate = tmp_dir / "video_only.mp4"

            with VideoWriter(intermediate, fps=reader.fps, width=reader.width, height=reader.height) as writer:
                for frame in reader:
                    writer.write(frame)
                print(f"Wrote {writer.frames_written} frames to intermediate.")

            if has_audio_stream(video_path):
                audio_tmp = tmp_dir / "audio.aac"
                extract_audio(video_path, audio_tmp)
                mux_video_audio(intermediate, audio_tmp, output_path)
                print("Audio preserved.")
            else:
                import shutil
                shutil.copy2(str(intermediate), str(output_path))
                print("No audio stream in input, copied video only.")

    print(f"Done. Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_cli.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Run ALL tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: 33 passed (6 swap + 3 device + 16 video_io + 8 cli).

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add cli/main.py tests/test_cli.py
git commit -m "feat: add lip-sync CLI with passthrough pipeline"
```

---

## Task 7: End-to-end verification + milestone notes

**Files:**
- Create: `~/Projects/lipsync-corrector/docs/milestones/milestone-1.md`

- [ ] **Step 1: Run the passthrough CLI on the real test clip**

```bash
cd ~/Projects/lipsync-corrector
uv run python -m cli.main \
  --video examples/input.mp4 \
  --output examples/passthrough_output.mp4
```

Expected: prints frame count, fps, dimensions, "Audio preserved.", "Done." without errors.

- [ ] **Step 2: Verify the output visually**

```bash
open examples/passthrough_output.mp4
```

The output video should look identical to `examples/input.mp4` — same content, same audio, same duration. The passthrough must not introduce visible artifacts.

- [ ] **Step 3: Verify frame count matches**

```bash
cd ~/Projects/lipsync-corrector
uv run python -c "
from core.video_io import VideoReader, has_audio_stream
from pathlib import Path
with VideoReader(Path('examples/input.mp4')) as orig:
    print(f'Original: {orig.frame_count} frames, {orig.fps:.1f} fps, {orig.width}x{orig.height}')
with VideoReader(Path('examples/passthrough_output.mp4')) as out:
    print(f'Output:   {out.frame_count} frames, {out.fps:.1f} fps, {out.width}x{out.height}')
print(f'Original has audio: {has_audio_stream(Path(\"examples/input.mp4\"))}')
print(f'Output has audio:   {has_audio_stream(Path(\"examples/passthrough_output.mp4\"))}')
"
```

Expected: frame count, fps, and dimensions match. Audio stream preserved.

- [ ] **Step 4: Clean up the passthrough output (git-ignored but tidy)**

```bash
rm -f ~/Projects/lipsync-corrector/examples/passthrough_output.mp4
```

- [ ] **Step 5: Write milestone notes**

Create `docs/milestones/milestone-1.md`:

```markdown
# Milestone 1: Track B Setup + video_io

**Date completed:** 2026-04-12
**Track:** B
**Status:** Done

## What was built

- `core/` package with `device.py` (hardware selection) and `video_io.py` (VideoReader, VideoWriter, ffmpeg audio helpers).
- `cli/main.py` — lip-sync CLI entry point with passthrough pipeline.
- Test fixtures: auto-generated tiny test videos (with and without audio) in `conftest.py`.
- Full test suite: <N> tests passing.

## How to run the passthrough

```bash
uv run python -m cli.main \
  --video examples/input.mp4 \
  --output examples/passthrough_output.mp4
```

## Measured results

- Frame count preserved: <yes/no>
- Audio preserved: <yes/no>
- FPS match: <yes/no>
- Visual quality: <identical / minor codec artifacts>

## What was learned

- <fill in after running>

## Next milestone

Milestone 2: `face_tracker` — stable face detection and tracking across frames.
See `docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md` section 8.
```

Fill in the `<placeholders>` with actual values from the end-to-end run.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add docs/milestones/milestone-1.md
git commit -m "feat: milestone-1 complete — Track B setup and video_io passthrough"
```

- [ ] **Step 7: Run full test suite one last time**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: 33 passed, no warnings, no failures.

---

## Done criteria for Milestone 1

- `uv run pytest` passes all tests (target: 33).
- `uv run python -m cli.main --video examples/input.mp4 --output examples/passthrough_output.mp4` produces a video with identical frame count and preserved audio.
- `docs/milestones/milestone-1.md` is written with actual measurements.
- Everything is committed on `milestone-1` branch, ready to merge to `main`.
- `swap.py` and its 6 tests remain untouched and passing.

Milestone 2 (face_tracker) is out of scope. Do not start it in the same session.
