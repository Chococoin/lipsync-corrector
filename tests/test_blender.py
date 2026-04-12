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
