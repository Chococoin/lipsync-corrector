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
