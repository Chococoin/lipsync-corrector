from __future__ import annotations

import cv2
import numpy as np

from core.mouth_region import FaceCrop


def blend_back(
    frame: np.ndarray,
    modified_crop: np.ndarray,
    face_crop: FaceCrop,
    feather_pixels: int = 8,
    mouth_only: bool = False,
    mouth_top_ratio: float = 0.4,
    mouth_bottom_ratio: float = 0.75,
    mouth_blend_ratio: float = 0.15,
) -> np.ndarray:
    """Paste a modified face crop back onto the frame with feathered edges.

    When mouth_only is False (default), the entire crop is blended with a
    uniform feathered mask — identical to the original behavior.

    When mouth_only is True, the mask uses a vertical gradient: zero opacity
    in the upper face (eyes, forehead stay original), linear ramp through
    the nose bridge, full opacity in the lower face (mouth from the model).
    This preserves the original resolution in the upper face while only
    applying the model's output where it matters.
    """
    result = frame.copy()
    x1, y1, x2, y2 = face_crop.bbox.astype(int)
    dst_w = x2 - x1
    dst_h = y2 - y1
    if dst_w <= 0 or dst_h <= 0:
        return result

    warped = cv2.resize(modified_crop, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    if mouth_only:
        mask = _build_mouth_only_mask(dst_h, dst_w, feather_pixels,
                                       mouth_top_ratio, mouth_bottom_ratio,
                                       mouth_blend_ratio)
    else:
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


def _build_mouth_only_mask(
    dst_h: int,
    dst_w: int,
    feather_pixels: int,
    mouth_top_ratio: float,
    mouth_bottom_ratio: float,
    mouth_blend_ratio: float,
) -> np.ndarray:
    """Build a mask that covers only the mouth region of the face.

    The mask has five vertical zones:
    - Top: 0.0 — original pixels (eyes, forehead)
    - Upper ramp: linear 0→1 (nose bridge transition)
    - Middle: 1.0 — model output (mouth)
    - Lower ramp: linear 1→0 (chin transition)
    - Bottom: 0.0 — original pixels (jaw, neck)

    The vertical gradient is multiplied with lateral feathering (left/right
    edges fade to 0) so the blend is smooth in all directions.
    """
    top_start = int(mouth_top_ratio * dst_h)
    top_end = int((mouth_top_ratio + mouth_blend_ratio) * dst_h)
    top_end = min(top_end, dst_h)

    bot_start = int(mouth_bottom_ratio * dst_h)
    bot_end = int((mouth_bottom_ratio + mouth_blend_ratio) * dst_h)
    bot_end = min(bot_end, dst_h)

    vertical = np.zeros((dst_h, dst_w), dtype=np.float32)

    if top_end > top_start:
        ramp_up = np.linspace(0.0, 1.0, top_end - top_start, dtype=np.float32)
        vertical[top_start:top_end, :] = ramp_up[:, np.newaxis]

    vertical[top_end:bot_start, :] = 1.0

    if bot_end > bot_start:
        ramp_down = np.linspace(1.0, 0.0, bot_end - bot_start, dtype=np.float32)
        vertical[bot_start:bot_end, :] = ramp_down[:, np.newaxis]

    if feather_pixels > 0 and dst_w > 2 * feather_pixels:
        lateral = np.zeros((dst_h, dst_w), dtype=np.float32)
        lateral[:, feather_pixels:-feather_pixels] = 1.0
        k = feather_pixels * 2 + 1
        lateral = cv2.GaussianBlur(lateral, (k, k), 0)
    else:
        lateral = np.ones((dst_h, dst_w), dtype=np.float32)

    return vertical * lateral
