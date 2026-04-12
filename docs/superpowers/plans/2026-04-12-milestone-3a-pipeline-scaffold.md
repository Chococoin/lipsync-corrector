# Milestone 3a: Lip-Sync Pipeline Scaffold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete lip-sync pipeline end-to-end with a placeholder `IdentityModel` that passes face crops through unchanged, proving the geometry (crop → model interface → blend-back) works before integrating a real ML model in Milestone 3b.

**Architecture:** Three new modules — `core/mouth_region.py` (bbox-based face crop with padding, resize to target size), `core/lipsync_model.py` (abstract `LipSyncModel` interface + `IdentityModel` implementation), `core/blender.py` (paste modified crop back with feathered alpha blending). The CLI gains a `--lipsync` flag that wires everything together: read frame → track face → crop → process via model → blend back → write. With `IdentityModel`, the output matches the input except for the expected double-resize blur inside the face bbox (which is a useful visual signature that the pipeline is actually running).

**Tech Stack:** Python 3.11, opencv-python, numpy. No new dependencies. Reuses `FaceTracker` from Milestone 2 and `VideoReader`/`VideoWriter` from Milestone 1.

**Repo:** `~/Projects/lipsync-corrector` on `main`, branching to `milestone-3a`.

---

## File Structure (end state of this milestone)

```
lipsync-corrector/
├── core/
│   ├── __init__.py
│   ├── device.py                 # unchanged
│   ├── video_io.py               # unchanged
│   ├── face_tracker.py           # unchanged
│   ├── mouth_region.py           # NEW: FaceCrop dataclass + crop_face_region
│   ├── lipsync_model.py          # NEW: LipSyncModel ABC + IdentityModel
│   └── blender.py                # NEW: blend_back function
├── cli/
│   ├── __init__.py
│   └── main.py                   # MODIFIED: add --lipsync flag + pipeline path
├── tests/
│   ├── conftest.py               # unchanged
│   ├── test_swap.py              # unchanged
│   ├── test_device.py            # unchanged
│   ├── test_video_io.py          # unchanged
│   ├── test_face_tracker.py      # unchanged
│   ├── test_cli.py               # MODIFIED: add --lipsync tests
│   ├── test_mouth_region.py      # NEW
│   ├── test_lipsync_model.py     # NEW
│   └── test_blender.py           # NEW
├── docs/milestones/
│   └── milestone-3a.md           # written at the end
└── (everything else unchanged)
```

**Responsibility per new/changed file:**

- `core/mouth_region.py` — `FaceCrop` dataclass (holds the resized face image + the bbox used in original frame coordinates + the target size). `crop_face_region(frame, tracked, target_size, padding)` expands the tracker bbox by `padding` fraction on each side, clamps to frame bounds, crops, and resizes to `target_size`. Pure image/geometry, no ML.
- `core/lipsync_model.py` — Abstract `LipSyncModel` class with `process(face_crops, audio_path) -> list[np.ndarray]`. Concrete `IdentityModel` returns copies of the input crops. This establishes the interface that `Wav2LipModel` / `LatentSyncModel` / `MuseTalkModel` will implement in future milestones.
- `core/blender.py` — `blend_back(frame, modified_crop, face_crop, feather_pixels)` resizes `modified_crop` to the bbox size, builds a feathered alpha mask (soft edges), and alpha-blends it back into a copy of the frame. Returns the composited frame.
- `cli/main.py` — Adds `--lipsync` flag (store_true). When set, runs the full pipeline: frame → FaceTracker → crop_face_region → IdentityModel.process → blend_back → write. When not set, behavior is unchanged from earlier milestones (passthrough or `--debug-tracking`).
- `tests/test_mouth_region.py`, `tests/test_lipsync_model.py`, `tests/test_blender.py` — Unit tests for each new module using the existing `tmp_video` fixture and synthetic numpy arrays.

---

## Task 1: core/mouth_region.py — FaceCrop + crop_face_region

**Files:**
- Create: `~/Projects/lipsync-corrector/core/mouth_region.py`
- Create: `~/Projects/lipsync-corrector/tests/test_mouth_region.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_mouth_region.py`:

```python
import numpy as np
import pytest

from core.face_tracker import TrackedFace
from core.mouth_region import FaceCrop, crop_face_region


def _tracked(x1, y1, x2, y2):
    return TrackedFace(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
        landmarks=None,
        confidence=0.9,
        detected=True,
    )


class TestFaceCropDataclass:
    def test_fields(self):
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        bbox = np.array([10, 20, 100, 200], dtype=np.float64)
        fc = FaceCrop(image=img, bbox=bbox, target_size=(96, 96))
        assert fc.image.shape == (96, 96, 3)
        np.testing.assert_array_equal(fc.bbox, bbox)
        assert fc.target_size == (96, 96)


class TestCropFaceRegion:
    def test_returns_face_crop_with_target_size(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        tracked = _tracked(100, 100, 300, 300)
        fc = crop_face_region(frame, tracked, target_size=(96, 96), padding=0.0)
        assert fc.image.shape == (96, 96, 3)
        assert fc.target_size == (96, 96)

    def test_padding_expands_bbox(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        tracked = _tracked(200, 200, 300, 300)
        fc = crop_face_region(frame, tracked, target_size=(96, 96), padding=0.2)
        x1, y1, x2, y2 = fc.bbox
        assert x1 < 200
        assert y1 < 200
        assert x2 > 300
        assert y2 > 300

    def test_padding_clamps_at_frame_edges(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        tracked = _tracked(0, 0, 50, 50)
        fc = crop_face_region(frame, tracked, target_size=(96, 96), padding=0.5)
        x1, y1, _, _ = fc.bbox
        assert x1 >= 0
        assert y1 >= 0

    def test_padding_clamps_right_and_bottom(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        tracked = _tracked(590, 430, 640, 480)
        fc = crop_face_region(frame, tracked, target_size=(96, 96), padding=0.5)
        _, _, x2, y2 = fc.bbox
        assert x2 <= 640
        assert y2 <= 480

    def test_custom_target_size(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        tracked = _tracked(100, 100, 300, 300)
        fc = crop_face_region(frame, tracked, target_size=(128, 64), padding=0.0)
        assert fc.image.shape == (64, 128, 3)
        assert fc.target_size == (128, 64)

    def test_no_padding_matches_input_bbox(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        tracked = _tracked(100, 150, 300, 350)
        fc = crop_face_region(frame, tracked, target_size=(96, 96), padding=0.0)
        np.testing.assert_array_almost_equal(fc.bbox, [100, 150, 300, 350])

    def test_preserves_content_at_zero_padding(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[50:100, 50:100] = [255, 0, 0]
        tracked = _tracked(50, 50, 100, 100)
        fc = crop_face_region(frame, tracked, target_size=(50, 50), padding=0.0)
        assert np.all(fc.image[:, :, 0] == 255)
        assert np.all(fc.image[:, :, 1] == 0)
        assert np.all(fc.image[:, :, 2] == 0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_mouth_region.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `core/mouth_region.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.face_tracker import TrackedFace


@dataclass
class FaceCrop:
    """A cropped and resized face region with the bbox used in original frame coordinates."""
    image: np.ndarray
    bbox: np.ndarray
    target_size: tuple[int, int]


def crop_face_region(
    frame: np.ndarray,
    tracked: TrackedFace,
    target_size: tuple[int, int] = (96, 96),
    padding: float = 0.2,
) -> FaceCrop:
    """Crop the face region with padding and resize to target_size.

    The tracker bbox is expanded by `padding` fraction on each side,
    clamped to frame bounds, then cropped and resized. The returned
    FaceCrop carries both the resized image and the exact bbox used,
    so the blender can paste the modified result back into the same spot.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = tracked.bbox
    bw = x2 - x1
    bh = y2 - y1
    ex1 = max(0, int(round(x1 - padding * bw)))
    ey1 = max(0, int(round(y1 - padding * bh)))
    ex2 = min(w, int(round(x2 + padding * bw)))
    ey2 = min(h, int(round(y2 + padding * bh)))
    crop = frame[ey1:ey2, ex1:ex2]
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    return FaceCrop(
        image=resized,
        bbox=np.array([ex1, ey1, ex2, ey2], dtype=np.float64),
        target_size=target_size,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_mouth_region.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Run all tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: 61 passed (54 existing + 7 new).

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/mouth_region.py tests/test_mouth_region.py
git commit -m "feat: add FaceCrop dataclass and crop_face_region helper"
```

---

## Task 2: core/lipsync_model.py — LipSyncModel interface + IdentityModel

**Files:**
- Create: `~/Projects/lipsync-corrector/core/lipsync_model.py`
- Create: `~/Projects/lipsync-corrector/tests/test_lipsync_model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_lipsync_model.py`:

```python
import numpy as np
import pytest

from core.lipsync_model import IdentityModel, LipSyncModel


class TestLipSyncModelABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            LipSyncModel()


class TestIdentityModel:
    def test_is_lipsync_model_subclass(self):
        assert issubclass(IdentityModel, LipSyncModel)

    def test_returns_same_number_of_crops(self):
        model = IdentityModel()
        crops = [np.zeros((96, 96, 3), dtype=np.uint8) for _ in range(5)]
        result = model.process(crops, None)
        assert len(result) == 5

    def test_returns_same_shapes(self):
        model = IdentityModel()
        crops = [np.zeros((96, 96, 3), dtype=np.uint8) for _ in range(3)]
        result = model.process(crops, None)
        for original, processed in zip(crops, result):
            assert processed.shape == original.shape
            assert processed.dtype == original.dtype

    def test_returns_same_pixel_values(self):
        model = IdentityModel()
        crop = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        result = model.process([crop], None)
        np.testing.assert_array_equal(result[0], crop)

    def test_returns_copies_not_same_object(self):
        model = IdentityModel()
        crop = np.zeros((96, 96, 3), dtype=np.uint8)
        result = model.process([crop], None)
        assert result[0] is not crop

    def test_handles_empty_list(self):
        model = IdentityModel()
        result = model.process([], None)
        assert result == []

    def test_audio_path_can_be_none(self):
        model = IdentityModel()
        crops = [np.zeros((96, 96, 3), dtype=np.uint8)]
        result = model.process(crops, None)
        assert len(result) == 1

    def test_audio_path_is_ignored_by_identity(self):
        from pathlib import Path
        model = IdentityModel()
        crops = [np.zeros((96, 96, 3), dtype=np.uint8)]
        result = model.process(crops, Path("/nonexistent/fake.wav"))
        assert len(result) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_lipsync_model.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `core/lipsync_model.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class LipSyncModel(ABC):
    """Abstract interface for lip-sync models.

    Concrete implementations take a sequence of face crops and an audio signal
    and return modified face crops where the mouth region matches the audio.

    The interface is deliberately minimal: the pipeline caller handles video
    I/O, face tracking, and cropping. The model only cares about crops and audio.
    """

    @abstractmethod
    def process(
        self,
        face_crops: list[np.ndarray],
        audio_path: Optional[Path],
    ) -> list[np.ndarray]:
        """Process face crops with the given audio.

        Args:
            face_crops: list of face images, each of shape (H, W, 3) uint8.
            audio_path: path to the audio file to sync to. May be None for
                placeholder/testing models that do not use audio.

        Returns:
            List of modified face crops of the same length and shapes as the input.
        """
        ...


class IdentityModel(LipSyncModel):
    """Placeholder lip-sync model that returns crops unchanged.

    Used during Milestone 3a to validate the crop → model → blend pipeline
    geometry before integrating a real ML model in Milestone 3b.
    """

    def process(
        self,
        face_crops: list[np.ndarray],
        audio_path: Optional[Path],
    ) -> list[np.ndarray]:
        return [crop.copy() for crop in face_crops]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_lipsync_model.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Run all tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: 70 passed (61 existing + 9 new).

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/lipsync_model.py tests/test_lipsync_model.py
git commit -m "feat: add LipSyncModel ABC and IdentityModel placeholder"
```

---

## Task 3: core/blender.py — blend_back

**Files:**
- Create: `~/Projects/lipsync-corrector/core/blender.py`
- Create: `~/Projects/lipsync-corrector/tests/test_blender.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_blender.py`:

```python
import numpy as np
import pytest

from core.blender import blend_back
from core.mouth_region import FaceCrop


def _face_crop_at(x1, y1, x2, y2, image_size=(96, 96)):
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    return FaceCrop(
        image=img,
        bbox=np.array([x1, y1, x2, y2], dtype=np.float64),
        target_size=image_size,
    )


class TestBlendBack:
    def test_output_shape_matches_frame(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_returns_copy_not_original(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop)
        assert result is not frame

    def test_outside_bbox_is_unchanged(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop, feather_pixels=0)
        np.testing.assert_array_equal(result[0:100, :], frame[0:100, :])
        np.testing.assert_array_equal(result[300:, :], frame[300:, :])
        np.testing.assert_array_equal(result[:, 0:100], frame[:, 0:100])
        np.testing.assert_array_equal(result[:, 300:], frame[:, 300:])

    def test_inside_bbox_is_modified_without_feather(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop, feather_pixels=0)
        center_value = result[200, 200, 0]
        assert center_value >= 199

    def test_feathering_blends_at_edges(self):
        frame = np.full((480, 640, 3), 0, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop, feather_pixels=10)
        center_value = result[200, 200, 0]
        edge_value = result[100, 200, 0]
        assert center_value > edge_value

    def test_zero_size_bbox_returns_copy_of_frame(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.zeros((96, 96, 3), dtype=np.uint8)
        face_crop = _face_crop_at(200, 200, 200, 200)
        result = blend_back(frame, modified, face_crop)
        np.testing.assert_array_equal(result, frame)

    def test_bbox_at_frame_edge(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(0, 0, 100, 100)
        result = blend_back(frame, modified, face_crop, feather_pixels=0)
        assert result[50, 50, 0] >= 199

    def test_small_bbox_with_large_feather(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((20, 20, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 120, 120, image_size=(20, 20))
        result = blend_back(frame, modified, face_crop, feather_pixels=50)
        assert result.shape == frame.shape
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_blender.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `core/blender.py`**

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
) -> np.ndarray:
    """Paste a modified face crop back onto the frame with feathered edges.

    Resizes `modified_crop` to the bbox size stored in `face_crop`, builds a
    soft alpha mask (inner region = 1, edges fade to 0 over `feather_pixels`
    pixels), and alpha-blends the result into a copy of the frame.

    Returns a new frame; the input frame is not modified.
    """
    result = frame.copy()
    x1, y1, x2, y2 = face_crop.bbox.astype(int)
    dst_w = x2 - x1
    dst_h = y2 - y1
    if dst_w <= 0 or dst_h <= 0:
        return result

    warped = cv2.resize(modified_crop, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_blender.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Run all tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: 78 passed (70 existing + 8 new).

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/blender.py tests/test_blender.py
git commit -m "feat: add blend_back with feathered alpha blending"
```

---

## Task 4: cli/main.py — integrate --lipsync pipeline

**Files:**
- Modify: `~/Projects/lipsync-corrector/cli/main.py`
- Modify: `~/Projects/lipsync-corrector/tests/test_cli.py`

- [ ] **Step 1: Add failing tests for --lipsync flag**

Append to the existing `TestParseArgs` class in `tests/test_cli.py`:

```python
    def test_lipsync_flag_accepted(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4", "--lipsync"])
        assert args.lipsync is True

    def test_lipsync_default_false(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4"])
        assert args.lipsync is False
```

Append to the existing `TestMainPassthrough` class (or as a new test class after it) in `tests/test_cli.py`:

```python
class TestMainLipsync:
    def test_lipsync_preserves_frame_count(self, tmp_video_with_audio, tmp_path):
        output = tmp_path / "output.mp4"
        result = main([
            "--video", str(tmp_video_with_audio),
            "--output", str(output),
            "--lipsync",
        ])
        assert result == 0
        assert output.exists()
        from core.video_io import VideoReader
        with VideoReader(output) as reader:
            assert reader.frame_count == 10

    def test_lipsync_preserves_audio(self, tmp_video_with_audio, tmp_path):
        output = tmp_path / "output.mp4"
        result = main([
            "--video", str(tmp_video_with_audio),
            "--output", str(output),
            "--lipsync",
        ])
        assert result == 0
        from core.video_io import has_audio_stream
        assert has_audio_stream(output) is True

    def test_lipsync_on_faceless_video_still_writes_output(self, tmp_video, tmp_path):
        output = tmp_path / "output.mp4"
        result = main([
            "--video", str(tmp_video),
            "--output", str(output),
            "--lipsync",
        ])
        assert result == 0
        assert output.exists()
```

Note: the last test exercises the "no face detected, frame passes through untouched" branch of the lipsync pipeline. The `tmp_video` fixture produces solid-color frames that never contain a face, so every frame should fall through the `if tracked is None` branch.

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_cli.py -v
```

Expected: failures — `lipsync` attribute missing on the argparse result, `TestMainLipsync` tests fail because the flag is not wired up.

- [ ] **Step 3: Read the current cli/main.py**

Read the existing file to understand the current structure. It should currently have: argparse for `--video`, `--output`, `--audio`, `--debug-tracking`; a `main()` that opens a VideoReader, iterates frames, optionally runs the tracker + overlay, writes to an intermediate, and muxes audio.

- [ ] **Step 4: Add --lipsync flag to parse_args**

In `parse_args()`, after the existing `add_argument` calls (the last one being `--debug-tracking`), add:

```python
    parser.add_argument(
        "--lipsync",
        action="store_true",
        default=False,
        help="Run the full lipsync pipeline (track → crop → model → blend → write). "
             "Milestone 3a uses an IdentityModel placeholder.",
    )
```

- [ ] **Step 5: Add lipsync pipeline branch to main()**

In `main()`, after the existing tracker initialization block (the one that runs when `args.debug_tracking` is set) and before `with VideoReader(video_path) as reader:`, add a parallel initialization block:

```python
    lipsync_tracker = None
    lipsync_model = None
    crop_face_region = None
    blend_back = None
    if args.lipsync:
        from core.face_tracker import FaceTracker
        from core.lipsync_model import IdentityModel
        from core.mouth_region import crop_face_region
        from core.blender import blend_back
        print("Loading lip-sync pipeline (placeholder IdentityModel)...")
        lipsync_tracker = FaceTracker()
        lipsync_model = IdentityModel()
```

Then, inside the frame loop, REPLACE the current block:

```python
                for frame in reader:
                    if tracker is not None:
                        tracked = tracker.track(frame)
                        frame = draw_tracking_overlay(frame, tracked)
                    writer.write(frame)
                print(f"Wrote {writer.frames_written} frames to intermediate.")
```

with:

```python
                for frame in reader:
                    if tracker is not None:
                        tracked = tracker.track(frame)
                        frame = draw_tracking_overlay(frame, tracked)
                    elif lipsync_tracker is not None:
                        tracked = lipsync_tracker.track(frame)
                        if tracked is not None:
                            face_crop = crop_face_region(frame, tracked)
                            processed = lipsync_model.process([face_crop.image], args.audio)
                            frame = blend_back(frame, processed[0], face_crop)
                    writer.write(frame)
                print(f"Wrote {writer.frames_written} frames to intermediate.")
```

Note: `args.audio` may be None (the --audio flag is optional). `IdentityModel` ignores the audio argument, so this works for Milestone 3a. In Milestone 3b, the real model will consume the audio.

- [ ] **Step 6: Run the CLI tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_cli.py -v
```

Expected: all CLI tests pass (13 total: existing + 2 parse_args + 3 lipsync).

- [ ] **Step 7: Run all tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: 83 passed (78 existing + 5 new).

- [ ] **Step 8: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add cli/main.py tests/test_cli.py
git commit -m "feat: add --lipsync flag wiring tracker + IdentityModel + blender"
```

---

## Task 5: End-to-end run + milestone notes

**Files:**
- Create: `~/Projects/lipsync-corrector/docs/milestones/milestone-3a.md`

- [ ] **Step 1: Run the pipeline on the Veo test clip**

```bash
cd ~/Projects/lipsync-corrector
uv run python -m cli.main \
  --video ~/Downloads/Video_De_Mujer_Saludando_Generado.mp4 \
  --output examples/lipsync_scaffold.mp4 \
  --lipsync
```

Expected: prints provider selection, loads the face tracker and IdentityModel, processes all frames through the tracker → crop → identity → blend pipeline, writes output with preserved audio.

- [ ] **Step 2: Open the output video**

```bash
open examples/lipsync_scaffold.mp4
```

Expected visual behavior:
- Outside the face region: pixel-identical to the input.
- Inside the face region: slightly softer (blurred) because of the 96x96 → bbox-size double resize cycle. This is the expected visual signature that the pipeline is actually running — not a bug.
- At the face bbox boundary: smooth transition (no visible rectangle edge) thanks to feathered alpha.
- Audio: preserved from the input.

- [ ] **Step 3: Verify frame count and audio**

```bash
cd ~/Projects/lipsync-corrector
uv run python -c "
from pathlib import Path
from core.video_io import VideoReader, has_audio_stream
input_path = Path.home() / 'Downloads/Video_De_Mujer_Saludando_Generado.mp4'
output_path = Path('examples/lipsync_scaffold.mp4')
with VideoReader(input_path) as orig:
    print(f'Input:  {orig.frame_count} frames, {orig.fps:.1f} fps, {orig.width}x{orig.height}')
with VideoReader(output_path) as out:
    print(f'Output: {out.frame_count} frames, {out.fps:.1f} fps, {out.width}x{out.height}')
print(f'Audio preserved: {has_audio_stream(output_path)}')
"
```

Expected: matching frame counts, matching fps, matching dimensions, audio preserved.

- [ ] **Step 4: Clean up the output**

```bash
rm -f ~/Projects/lipsync-corrector/examples/lipsync_scaffold.mp4
```

- [ ] **Step 5: Write milestone notes**

Create `docs/milestones/milestone-3a.md`:

```markdown
# Milestone 3a: Lip-Sync Pipeline Scaffold

**Date completed:** 2026-04-12
**Track:** B
**Status:** Done

## What was built

- `core/mouth_region.py` — `FaceCrop` dataclass + `crop_face_region()` that expands the tracker bbox by a padding fraction, clamps to frame bounds, and resizes to a target size (default 96x96 for Wav2Lip compatibility).
- `core/lipsync_model.py` — `LipSyncModel` abstract base class with `process(face_crops, audio_path)` interface. `IdentityModel` concrete implementation returns crops unchanged. This establishes the API that real models (Wav2Lip, LatentSync, MuseTalk) will plug into.
- `core/blender.py` — `blend_back()` resizes the modified crop to the original bbox size, builds a feathered alpha mask (soft edges via Gaussian blur), and alpha-blends into a copy of the frame.
- `cli/main.py` — new `--lipsync` flag that runs the full pipeline: track face → crop region → run model → blend back → write.
- Test suite: <N> tests passing.

## How to run

```bash
uv run python -m cli.main \
  --video examples/input.mp4 \
  --output examples/lipsync_scaffold.mp4 \
  --lipsync
```

## Measured results

End-to-end run on the Veo-generated clip (192 frames, 24 fps, 1280x720):

- Frame count preserved: <yes/no>
- Audio preserved: <yes/no>
- Visual behavior: the face region is slightly blurred due to the 96x96 double-resize cycle; the rest of the frame is unchanged. Feathered blend produces no visible bbox rectangle at the boundary.

## What was learned

- <fill in after running>

## Deferred to Milestone 3b

- Real lip-sync model integration (Wav2Lip or a modern alternative).
- Audio preprocessing (mel spectrogram, 16 kHz mono resampling).
- Batch processing of crops (current per-frame call works for IdentityModel but will need refactoring for models that require temporal context).
- Face alignment via landmarks (currently bbox-only; Wav2Lip will work fine without rotation alignment on near-frontal faces).

## Next milestone

Milestone 3b: replace `IdentityModel` with a real Wav2Lip wrapper. See
`docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md` section 8.
```

Fill in the `<placeholders>` after running.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add docs/milestones/milestone-3a.md
git commit -m "feat: milestone-3a complete — pipeline scaffold with IdentityModel"
```

- [ ] **Step 7: Run the full test suite one last time**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: 83 passed, no warnings beyond the pre-existing insightface FutureWarnings.

---

## Done criteria for Milestone 3a

- `uv run pytest` passes all tests (target: 83).
- `uv run python -m cli.main --video <real-video> --output <path> --lipsync` produces a video with preserved frame count, preserved audio, and a visually softer face region as the signature of the pipeline running.
- `core/mouth_region.py`, `core/lipsync_model.py`, and `core/blender.py` exist with their respective tests.
- `cli/main.py` has a `--lipsync` flag that wires the three new modules together.
- `docs/milestones/milestone-3a.md` written with actual measurements.
- Everything committed on `milestone-3a` branch, ready to merge to `main`.
- Existing behavior (passthrough, `--debug-tracking`, face-swap `swap.py`) untouched.

Milestone 3b (real Wav2Lip integration) is out of scope. Do not start it in the same session.
