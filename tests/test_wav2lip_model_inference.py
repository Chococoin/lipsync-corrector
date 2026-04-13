from pathlib import Path

import numpy as np
import pytest

from core.wav2lip_model import Wav2LipModel, DEFAULT_CHECKPOINT_PATH


WEIGHTS_EXIST = DEFAULT_CHECKPOINT_PATH.exists()
requires_weights = pytest.mark.skipif(
    not WEIGHTS_EXIST,
    reason=f"wav2lip_gan.pth not present at {DEFAULT_CHECKPOINT_PATH}",
)


class TestConstructor:
    def test_missing_checkpoint_raises_filenotfound(self, tmp_path):
        missing = tmp_path / "not_there.pth"
        with pytest.raises(FileNotFoundError) as exc:
            Wav2LipModel(checkpoint_path=missing, fps=25.0)
        assert "not_there.pth" in str(exc.value)


@requires_weights
class TestWav2LipInference:
    def test_returns_same_number_of_crops(self, tmp_path):
        model = Wav2LipModel(fps=25.0)
        crops = [np.full((96, 96, 3), 128, dtype=np.uint8) for _ in range(5)]
        audio = _make_silence_wav(tmp_path, duration_s=1.0)
        result = model.process(crops, audio)
        assert len(result) == 5

    def test_returns_same_shapes_and_dtype(self, tmp_path):
        model = Wav2LipModel(fps=25.0)
        crops = [np.full((96, 96, 3), 128, dtype=np.uint8) for _ in range(3)]
        audio = _make_silence_wav(tmp_path, duration_s=1.0)
        result = model.process(crops, audio)
        for original, processed in zip(crops, result):
            assert processed.shape == (96, 96, 3)
            assert processed.dtype == np.uint8

    def test_output_differs_from_input(self, tmp_path):
        model = Wav2LipModel(fps=25.0)
        crop = np.full((96, 96, 3), 128, dtype=np.uint8)
        audio = _make_silence_wav(tmp_path, duration_s=1.0)
        result = model.process([crop], audio)
        # Wav2Lip should *not* return the input unchanged — that's the
        # visual signature that distinguishes it from IdentityModel.
        assert not np.array_equal(result[0], crop)


def _make_silence_wav(tmp_path: Path, duration_s: float) -> Path:
    import soundfile as sf
    n = int(16000 * duration_s)
    y = np.zeros(n, dtype=np.float32)
    path = tmp_path / "silence.wav"
    sf.write(str(path), y, 16000)
    return path
