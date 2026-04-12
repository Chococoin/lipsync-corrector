# Milestone 2: Face Tracker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a face tracker that detects and tracks the primary speaker's face across video frames with temporal smoothing (no jitter) and gap interpolation, producing a debug overlay video as the observable output.

**Architecture:** `core/face_tracker.py` contains three layers: `TrackedFace` (dataclass for per-frame results), `BboxSmoother` (exponential moving average + gap handling, pure logic), and `FaceTracker` (wraps insightface detection + smoother). The debug overlay is a helper function in the same module. `cli/main.py` gains a `--debug-tracking` flag that runs the tracker on each frame and renders bboxes/landmarks on the output video.

**Tech Stack:** Python 3.11, insightface (buffalo_l, already downloaded), opencv-python, numpy. No new dependencies.

**Repo:** `~/Projects/lipsync-corrector` on `main`, branching to `milestone-2`.

---

## File Structure (end state of this milestone)

```
lipsync-corrector/
├── core/
│   ├── __init__.py
│   ├── device.py                 # unchanged
│   ├── video_io.py               # unchanged
│   └── face_tracker.py           # NEW: TrackedFace, BboxSmoother, FaceTracker, draw_tracking_overlay
├── cli/
│   ├── __init__.py
│   └── main.py                   # MODIFIED: add --debug-tracking flag
├── tests/
│   ├── conftest.py               # unchanged
│   ├── test_swap.py              # unchanged
│   ├── test_device.py            # unchanged
│   ├── test_video_io.py          # unchanged
│   ├── test_cli.py               # MODIFIED: add debug-tracking tests
│   └── test_face_tracker.py      # NEW: BboxSmoother unit tests + FaceTracker smoke tests
├── docs/milestones/
│   └── milestone-2.md            # written at the end
└── (everything else unchanged)
```

**Responsibility per new/changed file:**

- `core/face_tracker.py` — All face tracking logic. `TrackedFace` dataclass holds per-frame results (bbox, landmarks, confidence, detected-vs-interpolated flag). `BboxSmoother` handles EMA smoothing and gap interpolation (pure numpy, no ML). `FaceTracker` wraps insightface's FaceAnalysis + BboxSmoother into a stateful `track(frame)` API. `draw_tracking_overlay()` renders debug bboxes/landmarks onto a frame.
- `cli/main.py` — Gains `--debug-tracking` flag. When set, imports FaceTracker, runs `track()` per frame, and draws overlay on each frame before writing. Without the flag, behavior is unchanged (passthrough).
- `tests/test_face_tracker.py` — Unit tests for BboxSmoother (pure logic: smoothing, gap handling, reset) and light integration tests for FaceTracker (loads without crashing, returns None on faceless frames).

---

## Task 1: TrackedFace dataclass + BboxSmoother with tests

**Files:**
- Create: `~/Projects/lipsync-corrector/core/face_tracker.py`
- Create: `~/Projects/lipsync-corrector/tests/test_face_tracker.py`

- [ ] **Step 1: Write failing tests for BboxSmoother**

Create `tests/test_face_tracker.py`:

```python
import numpy as np
import pytest

from core.face_tracker import BboxSmoother, TrackedFace


class TestTrackedFace:
    def test_dataclass_fields(self):
        tf = TrackedFace(
            bbox=np.array([10, 20, 100, 200], dtype=np.float64),
            landmarks=None,
            confidence=0.95,
            detected=True,
        )
        assert tf.bbox.shape == (4,)
        assert tf.confidence == 0.95
        assert tf.detected is True
        assert tf.landmarks is None


class TestBboxSmoother:
    def test_first_observation_passes_through(self):
        smoother = BboxSmoother(alpha=0.3, max_gap=5)
        bbox = np.array([10.0, 20.0, 100.0, 200.0])
        result = smoother.update(bbox, confidence=0.9)
        assert result is not None
        np.testing.assert_array_almost_equal(result.bbox, bbox)
        assert result.detected is True
        assert result.confidence == 0.9

    def test_smoothing_blends_observations(self):
        smoother = BboxSmoother(alpha=0.3, max_gap=5)
        bbox1 = np.array([10.0, 20.0, 100.0, 200.0])
        bbox2 = np.array([20.0, 30.0, 110.0, 210.0])
        smoother.update(bbox1, confidence=0.9)
        result = smoother.update(bbox2, confidence=0.9)
        assert result is not None
        expected = 0.3 * bbox2 + 0.7 * bbox1
        np.testing.assert_array_almost_equal(result.bbox, expected)

    def test_gap_holds_last_bbox(self):
        smoother = BboxSmoother(alpha=0.3, max_gap=5)
        bbox = np.array([10.0, 20.0, 100.0, 200.0])
        smoother.update(bbox, confidence=0.9)
        result = smoother.update(None)
        assert result is not None
        np.testing.assert_array_almost_equal(result.bbox, bbox)
        assert result.detected is False
        assert result.confidence == 0.0

    def test_gap_expires_after_max_gap(self):
        smoother = BboxSmoother(alpha=0.3, max_gap=3)
        bbox = np.array([10.0, 20.0, 100.0, 200.0])
        smoother.update(bbox, confidence=0.9)
        for _ in range(3):
            result = smoother.update(None)
            assert result is not None
        result = smoother.update(None)
        assert result is None

    def test_recovery_after_gap_smooths(self):
        smoother = BboxSmoother(alpha=0.3, max_gap=5)
        bbox1 = np.array([10.0, 20.0, 100.0, 200.0])
        bbox2 = np.array([15.0, 25.0, 105.0, 205.0])
        smoother.update(bbox1, confidence=0.9)
        smoother.update(None)
        smoother.update(None)
        result = smoother.update(bbox2, confidence=0.8)
        assert result is not None
        assert result.detected is True
        expected = 0.3 * bbox2 + 0.7 * bbox1
        np.testing.assert_array_almost_equal(result.bbox, expected)

    def test_reset_clears_state(self):
        smoother = BboxSmoother(alpha=0.3, max_gap=5)
        bbox = np.array([10.0, 20.0, 100.0, 200.0])
        smoother.update(bbox, confidence=0.9)
        smoother.reset()
        result = smoother.update(None)
        assert result is None

    def test_landmarks_passed_through(self):
        smoother = BboxSmoother(alpha=0.3, max_gap=5)
        bbox = np.array([10.0, 20.0, 100.0, 200.0])
        lmk = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], dtype=np.float64)
        result = smoother.update(bbox, landmarks=lmk, confidence=0.9)
        assert result is not None
        assert result.landmarks is not None
        assert result.landmarks.shape == (5, 2)

    def test_gap_preserves_last_landmarks(self):
        smoother = BboxSmoother(alpha=0.3, max_gap=5)
        bbox = np.array([10.0, 20.0, 100.0, 200.0])
        lmk = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], dtype=np.float64)
        smoother.update(bbox, landmarks=lmk, confidence=0.9)
        result = smoother.update(None)
        assert result is not None
        assert result.landmarks is not None
        np.testing.assert_array_equal(result.landmarks, lmk)

    def test_none_before_any_observation(self):
        smoother = BboxSmoother(alpha=0.3, max_gap=5)
        result = smoother.update(None)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_face_tracker.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement TrackedFace and BboxSmoother**

Create `core/face_tracker.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class TrackedFace:
    """Per-frame face tracking result."""
    bbox: np.ndarray
    landmarks: Optional[np.ndarray]
    confidence: float
    detected: bool


class BboxSmoother:
    """Exponential moving average smoother for bounding boxes with gap interpolation."""

    def __init__(self, alpha: float = 0.3, max_gap: int = 5) -> None:
        self._alpha = alpha
        self._max_gap = max_gap
        self._last_bbox: Optional[np.ndarray] = None
        self._last_landmarks: Optional[np.ndarray] = None
        self._gap_count: int = 0

    def update(
        self,
        bbox: Optional[np.ndarray],
        landmarks: Optional[np.ndarray] = None,
        confidence: float = 0.0,
    ) -> Optional[TrackedFace]:
        if bbox is not None:
            if self._last_bbox is None:
                smoothed = bbox.copy().astype(np.float64)
            else:
                smoothed = self._alpha * bbox + (1 - self._alpha) * self._last_bbox
            self._last_bbox = smoothed
            self._last_landmarks = landmarks
            self._gap_count = 0
            return TrackedFace(bbox=smoothed, landmarks=landmarks, confidence=confidence, detected=True)

        self._gap_count += 1
        if self._gap_count <= self._max_gap and self._last_bbox is not None:
            return TrackedFace(
                bbox=self._last_bbox.copy(),
                landmarks=self._last_landmarks,
                confidence=0.0,
                detected=False,
            )
        return None

    def reset(self) -> None:
        self._last_bbox = None
        self._last_landmarks = None
        self._gap_count = 0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_face_tracker.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Run all tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: 43 passed (33 existing + 10 new).

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/face_tracker.py tests/test_face_tracker.py
git commit -m "feat: add TrackedFace and BboxSmoother with tests"
```

---

## Task 2: FaceTracker class

**Files:**
- Modify: `~/Projects/lipsync-corrector/core/face_tracker.py`
- Modify: `~/Projects/lipsync-corrector/tests/test_face_tracker.py`

- [ ] **Step 1: Add failing tests for FaceTracker**

Append to `tests/test_face_tracker.py`. Add `FaceTracker` to the import at the top:

```python
from core.face_tracker import BboxSmoother, FaceTracker, TrackedFace
```

Then append:

```python
class TestFaceTracker:
    def test_loads_without_crashing(self):
        tracker = FaceTracker()
        assert tracker is not None

    def test_returns_none_for_solid_color_frame(self):
        tracker = FaceTracker()
        frame = np.full((480, 640, 3), (128, 128, 128), dtype=np.uint8)
        result = tracker.track(frame)
        assert result is None

    def test_gap_handling_on_faceless_frames(self):
        tracker = FaceTracker(max_gap=2)
        frame_with_no_face = np.full((480, 640, 3), (0, 0, 0), dtype=np.uint8)
        results = [tracker.track(frame_with_no_face) for _ in range(5)]
        assert all(r is None for r in results)

    def test_reset_clears_state(self):
        tracker = FaceTracker()
        frame = np.full((480, 640, 3), (128, 128, 128), dtype=np.uint8)
        tracker.track(frame)
        tracker.reset()
        result = tracker.track(frame)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_face_tracker.py::TestFaceTracker -v
```

Expected: ImportError for FaceTracker.

- [ ] **Step 3: Implement FaceTracker**

Append to `core/face_tracker.py`, after the BboxSmoother class:

```python
class FaceTracker:
    """Detects and tracks the primary face across video frames using insightface."""

    def __init__(
        self,
        providers: Optional[list[str]] = None,
        det_size: tuple[int, int] = (640, 640),
        alpha: float = 0.3,
        max_gap: int = 5,
    ) -> None:
        from insightface.app import FaceAnalysis
        from core.device import get_onnx_providers

        self._providers = providers or get_onnx_providers()
        self._app = FaceAnalysis(name="buffalo_l", providers=self._providers)
        self._app.prepare(ctx_id=0, det_size=det_size)
        self._smoother = BboxSmoother(alpha=alpha, max_gap=max_gap)

    def track(self, frame: np.ndarray) -> Optional[TrackedFace]:
        faces = self._app.get(frame)
        if faces:
            faces.sort(
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )
            primary = faces[0]
            kps = getattr(primary, "kps", None)
            return self._smoother.update(primary.bbox, landmarks=kps, confidence=float(primary.det_score))
        return self._smoother.update(None)

    def reset(self) -> None:
        self._smoother.reset()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_face_tracker.py -v
```

Expected: 14 passed (10 smoother + 1 dataclass + 4 tracker). Note: FaceTracker tests are slower (~2-5s) because they load insightface models.

- [ ] **Step 5: Run all tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: 47 passed.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/face_tracker.py tests/test_face_tracker.py
git commit -m "feat: add FaceTracker wrapping insightface with temporal smoothing"
```

---

## Task 3: Debug overlay + CLI --debug-tracking flag

**Files:**
- Modify: `~/Projects/lipsync-corrector/core/face_tracker.py`
- Modify: `~/Projects/lipsync-corrector/cli/main.py`
- Modify: `~/Projects/lipsync-corrector/tests/test_face_tracker.py`
- Modify: `~/Projects/lipsync-corrector/tests/test_cli.py`

- [ ] **Step 1: Add failing test for draw_tracking_overlay**

Add `draw_tracking_overlay` to the import in `tests/test_face_tracker.py`:

```python
from core.face_tracker import BboxSmoother, FaceTracker, TrackedFace, draw_tracking_overlay
```

Append to `tests/test_face_tracker.py`:

```python
class TestDrawTrackingOverlay:
    def test_returns_copy_not_original(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracked = TrackedFace(
            bbox=np.array([10, 10, 50, 50], dtype=np.float64),
            landmarks=None,
            confidence=0.9,
            detected=True,
        )
        result = draw_tracking_overlay(frame, tracked)
        assert result is not frame
        assert result.shape == frame.shape

    def test_modifies_frame_when_tracked(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracked = TrackedFace(
            bbox=np.array([10, 10, 50, 50], dtype=np.float64),
            landmarks=None,
            confidence=0.9,
            detected=True,
        )
        result = draw_tracking_overlay(frame, tracked)
        assert not np.array_equal(result, frame)

    def test_returns_unchanged_when_none(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = draw_tracking_overlay(frame, None)
        np.testing.assert_array_equal(result, frame)

    def test_handles_landmarks(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        lmk = np.array([[20, 25], [40, 25], [30, 35], [22, 45], [38, 45]], dtype=np.float64)
        tracked = TrackedFace(
            bbox=np.array([10, 10, 50, 50], dtype=np.float64),
            landmarks=lmk,
            confidence=0.9,
            detected=True,
        )
        result = draw_tracking_overlay(frame, tracked)
        assert not np.array_equal(result, frame)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_face_tracker.py::TestDrawTrackingOverlay -v
```

Expected: ImportError for draw_tracking_overlay.

- [ ] **Step 3: Implement draw_tracking_overlay**

Append to `core/face_tracker.py`:

```python
def draw_tracking_overlay(frame: np.ndarray, tracked: Optional[TrackedFace]) -> np.ndarray:
    """Draw a bounding box and landmarks overlay on a frame copy."""
    result = frame.copy()
    if tracked is None:
        return result
    x1, y1, x2, y2 = tracked.bbox.astype(int)
    color = (0, 255, 0) if tracked.detected else (0, 165, 255)
    cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
    label = f"{tracked.confidence:.2f}" if tracked.detected else "interp"
    cv2.putText(result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if tracked.landmarks is not None:
        for lx, ly in tracked.landmarks.astype(int):
            cv2.circle(result, (int(lx), int(ly)), 2, (255, 0, 0), -1)
    return result
```

- [ ] **Step 4: Run overlay tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_face_tracker.py -v
```

Expected: 18 passed (10 smoother + 1 dataclass + 4 tracker + 4 overlay). Note: the TestTrackedFace test was counted earlier in the 10, so actually it should be 10 + 4 + 4 = 18. Let me recount: TestTrackedFace(1) + TestBboxSmoother(9) + TestFaceTracker(4) + TestDrawTrackingOverlay(4) = 18.

- [ ] **Step 5: Add --debug-tracking flag to cli/main.py**

Read `cli/main.py` first, then modify it. Add to `parse_args()`:

```python
    parser.add_argument(
        "--debug-tracking",
        action="store_true",
        default=False,
        help="Overlay face-tracking bounding boxes on the output video.",
    )
```

Modify `main()` to conditionally import and use the tracker. After `output_path.parent.mkdir(...)` and before `with VideoReader(...)`, add tracker initialization:

```python
    tracker = None
    if args.debug_tracking:
        from core.face_tracker import FaceTracker, draw_tracking_overlay
        print("Loading face tracker...")
        tracker = FaceTracker()
```

Inside the frame loop (where `writer.write(frame)` is), replace with:

```python
                for frame in reader:
                    if tracker is not None:
                        tracked = tracker.track(frame)
                        frame = draw_tracking_overlay(frame, tracked)
                    writer.write(frame)
```

- [ ] **Step 6: Add CLI test for --debug-tracking**

Append to `tests/test_cli.py`:

```python
    def test_debug_tracking_flag_accepted(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4", "--debug-tracking"])
        assert args.debug_tracking is True

    def test_debug_tracking_default_false(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4"])
        assert args.debug_tracking is False
```

Add these inside the existing `TestParseArgs` class.

- [ ] **Step 7: Run all tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: 53 passed (33 existing + 18 face_tracker + 2 new cli).

- [ ] **Step 8: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/face_tracker.py cli/main.py tests/test_face_tracker.py tests/test_cli.py
git commit -m "feat: add debug tracking overlay and --debug-tracking CLI flag"
```

---

## Task 4: End-to-end run + milestone notes

**Files:**
- Create: `~/Projects/lipsync-corrector/docs/milestones/milestone-2.md`

- [ ] **Step 1: Run the tracker on the real test clip**

```bash
cd ~/Projects/lipsync-corrector
uv run python -m cli.main \
  --video examples/input.mp4 \
  --output examples/tracking_debug.mp4 \
  --debug-tracking
```

Expected: prints loading messages, processes all frames with face detection + smoothing per frame, writes output with green bounding boxes overlaid on detected faces. This will be significantly slower than passthrough due to face detection per frame.

- [ ] **Step 2: Open and inspect the debug video**

```bash
open examples/tracking_debug.mp4
```

Watch the clip. Verify:
- Green bounding boxes appear around the speaker's face.
- Boxes are stable (no jitter between frames).
- When the face is briefly occluded or turned, orange "interp" boxes appear instead of the box jumping or disappearing.
- Landmarks (blue dots) visible on eyes, nose, mouth corners.

- [ ] **Step 3: Clean up output**

```bash
rm -f ~/Projects/lipsync-corrector/examples/tracking_debug.mp4
```

- [ ] **Step 4: Write milestone notes**

Create `docs/milestones/milestone-2.md`:

```markdown
# Milestone 2: Face Tracker

**Date completed:** 2026-04-12
**Track:** B
**Status:** Done

## What was built

- `core/face_tracker.py` with:
  - `TrackedFace` dataclass (bbox, landmarks, confidence, detected flag)
  - `BboxSmoother` (EMA smoothing + gap interpolation, pure numpy)
  - `FaceTracker` (wraps insightface FaceAnalysis + BboxSmoother)
  - `draw_tracking_overlay()` for debug visualization
- `cli/main.py` updated with `--debug-tracking` flag.
- Test suite: <N> tests passing.

## How to run

```bash
uv run python -m cli.main \
  --video examples/input.mp4 \
  --output examples/tracking_debug.mp4 \
  --debug-tracking
```

## Measured results

- Detection fps: <fill in>
- Smoothing visible: <yes/no — are boxes stable?>
- Gap handling: <observed or not — did the face disappear briefly?>
- Landmarks visible: <yes/no>

## What was learned

- <fill in after running>

## Next milestone

Milestone 3: Wav2Lip integration (baseline lip-sync).
See `docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md` section 8.
```

Fill in actual measurements and observations.

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add docs/milestones/milestone-2.md
git commit -m "feat: milestone-2 complete — face tracker with debug overlay"
```

- [ ] **Step 6: Run full test suite one last time**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest -v
```

Expected: all tests pass.

---

## Done criteria for Milestone 2

- `uv run pytest` passes all tests (target: ~53).
- `uv run python -m cli.main --video examples/input.mp4 --output examples/tracking_debug.mp4 --debug-tracking` produces a video with visible, stable bounding boxes.
- BboxSmoother has thorough unit tests covering: first observation, smoothing blend, gap holding, gap expiration, recovery after gap, reset, landmarks passthrough.
- `docs/milestones/milestone-2.md` written with actual observations.
- Everything committed on `milestone-2` branch, ready to merge to `main`.
- All existing tests (33) remain passing and untouched (except test_cli.py which gains 2 new tests).

Milestone 3 (Wav2Lip integration) is out of scope. Do not start it in the same session.
