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


class FaceTracker:
    """Detects and tracks the primary face across video frames using insightface."""

    def __init__(
        self,
        providers: Optional[list[str]] = None,
        det_size: tuple[int, int] = (640, 640),
        alpha: float = 0.3,
        max_gap: int = 5,
    ) -> None:
        from insightface.app import FaceAnalysis
        from core.device import get_onnx_providers

        self._providers = providers or get_onnx_providers()
        self._app = FaceAnalysis(name="buffalo_l", providers=self._providers)
        self._app.prepare(ctx_id=0, det_size=det_size)
        self._smoother = BboxSmoother(alpha=alpha, max_gap=max_gap)

    def track(self, frame: np.ndarray) -> Optional[TrackedFace]:
        faces = self._app.get(frame)
        if faces:
            faces.sort(
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )
            primary = faces[0]
            kps = getattr(primary, "kps", None)
            return self._smoother.update(primary.bbox, landmarks=kps, confidence=float(primary.det_score))
        return self._smoother.update(None)

    def reset(self) -> None:
        self._smoother.reset()
