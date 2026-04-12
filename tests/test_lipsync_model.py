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
