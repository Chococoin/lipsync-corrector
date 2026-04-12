from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class TrackedFace:
    """Per-frame face tracking result."""
    bbox: np.ndarray
    landmarks: Optional[np.ndarray]
    confidence: float
    detected: bool


class BboxSmoother:
    """Exponential moving average smoother for bounding boxes with gap interpolation."""

    def __init__(self, alpha: float = 0.3, max_gap: int = 5) -> None:
        self._alpha = alpha
        self._max_gap = max_gap
        self._last_bbox: Optional[np.ndarray] = None
        self._last_landmarks: Optional[np.ndarray] = None
        self._gap_count: int = 0

    def update(
        self,
        bbox: Optional[np.ndarray],
        landmarks: Optional[np.ndarray] = None,
        confidence: float = 0.0,
    ) -> Optional[TrackedFace]:
        if bbox is not None:
            if self._last_bbox is None:
                smoothed = bbox.copy().astype(np.float64)
            else:
                smoothed = self._alpha * bbox + (1 - self._alpha) * self._last_bbox
            self._last_bbox = smoothed
            self._last_landmarks = landmarks
            self._gap_count = 0
            return TrackedFace(bbox=smoothed, landmarks=landmarks, confidence=confidence, detected=True)

        self._gap_count += 1
        if self._gap_count <= self._max_gap and self._last_bbox is not None:
            return TrackedFace(
                bbox=self._last_bbox.copy(),
                landmarks=self._last_landmarks,
                confidence=0.0,
                detected=False,
            )
        return None

    def reset(self) -> None:
        self._last_bbox = None
        self._last_landmarks = None
        self._gap_count = 0
