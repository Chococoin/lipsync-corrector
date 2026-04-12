import numpy as np
import pytest

from core.face_tracker import BboxSmoother, FaceTracker, TrackedFace, draw_tracking_overlay


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
