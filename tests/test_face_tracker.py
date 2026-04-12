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
