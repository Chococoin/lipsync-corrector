from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def tmp_video(tmp_path: Path) -> Path:
    """Create a tiny 10-frame 64x64 video with no audio. Each frame is a different solid color."""
    video_path = tmp_path / "test_no_audio.mp4"
    fps = 10.0
    size = (64, 64)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, size)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 128),
    ]
    for color in colors:
        frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


@pytest.fixture
def tmp_video_with_audio(tmp_video: Path, tmp_path: Path) -> Path:
    """Take the no-audio video and add a silent audio track via ffmpeg."""
    output = tmp_path / "test_with_audio.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(tmp_video),
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(output),
        ],
        check=True,
    )
    return output
