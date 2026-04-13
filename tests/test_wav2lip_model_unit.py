import pytest
import torch

from core.wav2lip import Wav2Lip


class TestWav2LipArchitecture:
    def test_can_instantiate_on_cpu(self):
        model = Wav2Lip()
        assert isinstance(model, torch.nn.Module)

    def test_expected_input_output_shapes(self):
        model = Wav2Lip().eval()
        face = torch.zeros(1, 6, 96, 96)
        mel = torch.zeros(1, 1, 80, 16)
        with torch.no_grad():
            out = model(mel, face)
        assert out.shape == (1, 3, 96, 96)

    def test_batch_inference(self):
        model = Wav2Lip().eval()
        face = torch.zeros(4, 6, 96, 96)
        mel = torch.zeros(4, 1, 80, 16)
        with torch.no_grad():
            out = model(mel, face)
        assert out.shape == (4, 3, 96, 96)
