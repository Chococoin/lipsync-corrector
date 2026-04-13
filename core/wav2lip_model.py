"""Wav2Lip concrete implementation of LipSyncModel."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.device import get_torch_device
from core.lipsync_model import LipSyncModel
from core.wav2lip import Wav2Lip
from core.wav2lip.audio import load_wav_mono_16k, melspectrogram
from core.wav2lip.frame_sync import MEL_STEP_SIZE, get_mel_chunk_for_frame

DEFAULT_CHECKPOINT_PATH = Path(__file__).parent.parent / "models" / "wav2lip_gan.pth"

IMG_SIZE = 96
INFERENCE_BATCH_SIZE = 16


class Wav2LipModel(LipSyncModel):
    """Real lip-sync model using the pretrained Wav2Lip GAN checkpoint.

    The constructor loads the checkpoint and prepares the model on the
    best available device (MPS on Apple Silicon, else CUDA, else CPU).
    The `fps` argument is the source video fps, needed for frame↔mel
    alignment during inference.
    """

    def __init__(
        self,
        fps: float,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        device: Optional[str] = None,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"wav2lip checkpoint not found: {checkpoint_path}. "
                f"Download wav2lip_gan.pth and place it at that path. "
                f"See README for the download link."
            )
        self.fps = float(fps)
        self.device = device or get_torch_device()
        self.model = self._load_model(checkpoint_path, self.device)

    @staticmethod
    def _load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
        # weights_only=False is required for legacy checkpoints like
        # wav2lip_gan.pth that contain a dict rather than a bare state_dict.
        # Torch >=2.6 defaults to True and will refuse to load them.
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=device,
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Some forks save with "module." prefix from DataParallel
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.removeprefix("module.")] = v
        model = Wav2Lip()
        model.load_state_dict(cleaned)
        model = model.to(device)
        model.eval()
        return model

    def process(
        self,
        face_crops: list[np.ndarray],
        audio_path: Optional[Path],
    ) -> list[np.ndarray]:
        if audio_path is None:
            raise ValueError("Wav2LipModel requires an audio_path")
        if not face_crops:
            return []

        wav = load_wav_mono_16k(audio_path)
        mel = melspectrogram(wav)

        out_crops: list[np.ndarray] = [None] * len(face_crops)  # type: ignore[list-item]
        for batch_start in range(0, len(face_crops), INFERENCE_BATCH_SIZE):
            batch_end = min(batch_start + INFERENCE_BATCH_SIZE, len(face_crops))
            face_batch = []
            mel_batch = []
            for i in range(batch_start, batch_end):
                face_batch.append(face_crops[i])
                mel_batch.append(get_mel_chunk_for_frame(mel, i, self.fps))

            face_tensor, mel_tensor = self._build_batch(face_batch, mel_batch)
            face_tensor = face_tensor.to(self.device)
            mel_tensor = mel_tensor.to(self.device)

            with torch.no_grad():
                pred = self.model(mel_tensor, face_tensor)

            pred_np = pred.detach().cpu().numpy()
            for local_i, frame_i in enumerate(range(batch_start, batch_end)):
                out_crops[frame_i] = self._tensor_to_crop(pred_np[local_i])

        return out_crops

    @staticmethod
    def _build_batch(
        face_batch: list[np.ndarray],
        mel_batch: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        faces = np.stack(face_batch, axis=0).astype(np.float32) / 255.0
        masked = faces.copy()
        masked[:, IMG_SIZE // 2:, :, :] = 0.0
        six_channel = np.concatenate([masked, faces], axis=3)
        six_channel = np.transpose(six_channel, (0, 3, 1, 2))
        face_tensor = torch.from_numpy(six_channel).float()

        mels = np.stack(mel_batch, axis=0).astype(np.float32)
        mels = mels[:, np.newaxis, :, :]
        mel_tensor = torch.from_numpy(mels).float()
        return face_tensor, mel_tensor

    @staticmethod
    def _tensor_to_crop(pred_chw: np.ndarray) -> np.ndarray:
        img = np.transpose(pred_chw, (1, 2, 0))
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img
