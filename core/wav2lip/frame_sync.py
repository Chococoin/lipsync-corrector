"""Frame index → mel chunk alignment for Wav2Lip inference."""

from __future__ import annotations

import numpy as np

MEL_STEP_SIZE = 16
MEL_COLS_PER_SECOND = 80  # 16000 / 200 (hop_size)


def get_mel_chunk_for_frame(
    mel: np.ndarray,
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    """Return the 80x16 mel chunk aligned to video frame `frame_idx`.

    The chunk start column is int(80 * frame_idx / fps). The chunk is 16
    columns wide. If the video extends past the end of the audio, the
    last valid window is returned (clamped to the tail of `mel`). The
    result is always shape (80, 16).
    """
    n_mel, n_cols = mel.shape
    start = int(MEL_COLS_PER_SECOND * frame_idx / fps)
    end = start + MEL_STEP_SIZE
    if end > n_cols:
        # Shift the window left so the chunk stays inside the mel
        end = n_cols
        start = max(0, end - MEL_STEP_SIZE)
    chunk = mel[:, start:end]
    if chunk.shape[1] < MEL_STEP_SIZE:
        # Audio is shorter than 16 mel columns total: right-pad with edge
        pad = MEL_STEP_SIZE - chunk.shape[1]
        chunk = np.pad(chunk, ((0, 0), (0, pad)), mode="edge")
    return chunk
