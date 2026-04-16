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


class TestBlendBackMouthOnly:
    def test_mouth_only_default_false_matches_original(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result_default = blend_back(frame, modified, face_crop, feather_pixels=0)
        result_explicit = blend_back(frame, modified, face_crop, feather_pixels=0, mouth_only=False)
        np.testing.assert_array_equal(result_default, result_explicit)

    def test_top_of_face_is_original(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop, feather_pixels=0, mouth_only=True)
        # Top 30% of bbox (rows 100-159) should be pure original (100).
        # Using 30% instead of 40% to stay safely above the transition zone.
        top_region = result[100:160, 150:250, 0]
        assert np.all(top_region == 100)

    def test_bottom_of_face_is_modified(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop, feather_pixels=0, mouth_only=True)
        # Bottom 30% of bbox (rows 241-299) should be pure modified (200).
        # Using 70% to stay safely below the transition zone.
        bottom_region = result[241:299, 150:250, 0]
        assert np.all(bottom_region >= 199)

    def test_transition_zone_is_blended(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop, feather_pixels=0, mouth_only=True)
        # Transition zone center (~47.5% of bbox = row 195) should be
        # between 100 and 200 (a blend of original and modified).
        # mouth_top_ratio=0.4 → transition starts at row 180
        # mouth_blend_ratio=0.15 → transition ends at row 210
        # Center of transition at row 195.
        transition_value = result[195, 200, 0]
        assert 110 < transition_value < 190

    def test_outside_bbox_unchanged(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop, feather_pixels=0, mouth_only=True)
        np.testing.assert_array_equal(result[0:100, :], frame[0:100, :])
        np.testing.assert_array_equal(result[300:, :], frame[300:, :])
        np.testing.assert_array_equal(result[:, 0:100], frame[:, 0:100])
        np.testing.assert_array_equal(result[:, 300:], frame[:, 300:])

    def test_small_bbox_does_not_crash(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((20, 20, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 120, 120, image_size=(20, 20))
        result = blend_back(frame, modified, face_crop, mouth_only=True)
        assert result.shape == frame.shape

    def test_returns_copy_not_original(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        modified = np.full((96, 96, 3), 200, dtype=np.uint8)
        face_crop = _face_crop_at(100, 100, 300, 300)
        result = blend_back(frame, modified, face_crop, mouth_only=True)
        assert result is not frame
