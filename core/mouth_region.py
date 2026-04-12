from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.face_tracker import TrackedFace


@dataclass
class FaceCrop:
    """A cropped and resized face region with the bbox used in original frame coordinates."""
    image: np.ndarray
    bbox: np.ndarray
    target_size: tuple[int, int]


def crop_face_region(
    frame: np.ndarray,
    tracked: TrackedFace,
    target_size: tuple[int, int] = (96, 96),
    padding: float = 0.2,
) -> FaceCrop:
    """Crop the face region with padding and resize to target_size.

    The tracker bbox is expanded by `padding` fraction on each side,
    clamped to frame bounds, then cropped and resized. The returned
    FaceCrop carries both the resized image and the exact bbox used,
    so the blender can paste the modified result back into the same spot.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = tracked.bbox
    bw = x2 - x1
    bh = y2 - y1
    ex1 = max(0, int(round(x1 - padding * bw)))
    ey1 = max(0, int(round(y1 - padding * bh)))
    ex2 = min(w, int(round(x2 + padding * bw)))
    ey2 = min(h, int(round(y2 + padding * bh)))
    crop = frame[ey1:ey2, ex1:ex2]
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    return FaceCrop(
        image=resized,
        bbox=np.array([ex1, ey1, ex2, ey2], dtype=np.float64),
        target_size=target_size,
    )
