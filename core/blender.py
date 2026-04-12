from __future__ import annotations

import cv2
import numpy as np

from core.mouth_region import FaceCrop


def blend_back(
    frame: np.ndarray,
    modified_crop: np.ndarray,
    face_crop: FaceCrop,
    feather_pixels: int = 8,
) -> np.ndarray:
    """Paste a modified face crop back onto the frame with feathered edges.

    Resizes `modified_crop` to the bbox size stored in `face_crop`, builds a
    soft alpha mask (inner region = 1, edges fade to 0 over `feather_pixels`
    pixels), and alpha-blends the result into a copy of the frame.

    Returns a new frame; the input frame is not modified.
    """
    result = frame.copy()
    x1, y1, x2, y2 = face_crop.bbox.astype(int)
    dst_w = x2 - x1
    dst_h = y2 - y1
    if dst_w <= 0 or dst_h <= 0:
        return result

    warped = cv2.resize(modified_crop, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    if feather_pixels > 0 and dst_w > 2 * feather_pixels and dst_h > 2 * feather_pixels:
        inner = np.zeros((dst_h, dst_w), dtype=np.float32)
        inner[feather_pixels:-feather_pixels, feather_pixels:-feather_pixels] = 1.0
        k = feather_pixels * 2 + 1
        mask = cv2.GaussianBlur(inner, (k, k), 0)
    else:
        mask = np.ones((dst_h, dst_w), dtype=np.float32)

    mask3 = np.stack([mask, mask, mask], axis=-1)

    region = result[y1:y2, x1:x2].astype(np.float32)
    blended = mask3 * warped.astype(np.float32) + (1.0 - mask3) * region
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result
