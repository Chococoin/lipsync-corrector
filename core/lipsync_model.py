from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class LipSyncModel(ABC):
    """Abstract interface for lip-sync models.

    Concrete implementations take a sequence of face crops and an audio signal
    and return modified face crops where the mouth region matches the audio.

    The interface is deliberately minimal: the pipeline caller handles video
    I/O, face tracking, and cropping. The model only cares about crops and audio.
    """

    @abstractmethod
    def process(
        self,
        face_crops: list[np.ndarray],
        audio_path: Optional[Path],
    ) -> list[np.ndarray]:
        """Process face crops with the given audio.

        Args:
            face_crops: list of face images, each of shape (H, W, 3) uint8.
            audio_path: path to the audio file to sync to. May be None for
                placeholder/testing models that do not use audio.

        Returns:
            List of modified face crops of the same length and shapes as the input.
        """
        ...


class IdentityModel(LipSyncModel):
    """Placeholder lip-sync model that returns crops unchanged.

    Used during Milestone 3a to validate the crop → model → blend pipeline
    geometry before integrating a real ML model in Milestone 3b.
    """

    def process(
        self,
        face_crops: list[np.ndarray],
        audio_path: Optional[Path],
    ) -> list[np.ndarray]:
        return [crop.copy() for crop in face_crops]
