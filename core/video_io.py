from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    """Context manager for reading video frames and metadata."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video not found: {self._path}")
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self._path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration(self) -> float:
        return self._frame_count / self._fps if self._fps > 0 else 0.0

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._cap is None:
            return
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def __len__(self) -> int:
        return self._frame_count

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class VideoWriter:
    """Context manager for writing video frames to an mp4 file."""

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        self._path = Path(path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {self._path}")
        self._frames_written = 0

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
        self._frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
