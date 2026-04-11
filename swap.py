"""Track A: Face-swap quick win. See docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

INSWAPPER_PATH = Path(__file__).parent / "models" / "inswapper_128.onnx"
DET_SIZE = (640, 640)


def select_providers(available: list[str]) -> list[str]:
    preferred = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    result = [p for p in preferred if p in available]
    if "CPUExecutionProvider" not in result:
        result.append("CPUExecutionProvider")
    return result


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="swap.py",
        description="Face-swap a reference face into a video (Track A, Milestone 0).",
    )
    parser.add_argument("--face", required=True, help="Path to the reference face image.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--output", required=True, help="Path to the output video.")
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Optional cap: process only the first N seconds of the video.",
    )
    return parser.parse_args(argv)


def build_face_analyzer(providers: list[str]) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    return app


def load_swapper(providers: list[str]):
    if not INSWAPPER_PATH.exists():
        raise FileNotFoundError(
            f"inswapper model not found at {INSWAPPER_PATH}. "
            "Download it with: curl -L -o models/inswapper_128.onnx "
            "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        )
    return get_model(str(INSWAPPER_PATH), providers=providers)


def extract_reference_face(face_analyzer: FaceAnalysis, image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read reference image: {image_path}")
    faces = face_analyzer.get(image)
    if not faces:
        raise ValueError(f"No face detected in reference image: {image_path}")
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0]


def open_video(video_path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, width, height, total


def swap_one_frame(frame, face_analyzer: FaceAnalysis, swapper, source_face) -> np.ndarray:
    faces = face_analyzer.get(frame)
    if not faces:
        return frame
    result = frame
    for target_face in faces:
        result = swapper.get(result, target_face, source_face, paste_back=True)
    return result


def process_video(
    video_path: Path,
    intermediate_path: Path,
    face_analyzer: FaceAnalysis,
    swapper,
    source_face,
    max_seconds: Optional[int],
) -> tuple[int, float]:
    cap, fps, width, height, total = open_video(video_path)

    frame_limit = total
    if max_seconds is not None:
        frame_limit = min(total, int(fps * max_seconds))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(intermediate_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for {intermediate_path}")

    import time
    start = time.perf_counter()
    processed = 0
    try:
        while processed < frame_limit:
            ok, frame = cap.read()
            if not ok:
                break
            swapped = swap_one_frame(frame, face_analyzer, swapper, source_face)
            writer.write(swapped)
            processed += 1
            if processed % 30 == 0:
                elapsed = time.perf_counter() - start
                print(f"  frame {processed}/{frame_limit}  ({processed / elapsed:.1f} fps)")
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start
    return processed, elapsed


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install with: brew install ffmpeg")


def extract_audio(video_path: Path, audio_out: Path) -> bool:
    """Extract the audio track to a temp file. Returns False if the video has no audio."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    if "audio" not in probe.stdout:
        return False
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
         "-vn", "-acodec", "copy", str(audio_out)],
        check=True,
    )
    return True


def mux_video_audio(video_in: Path, audio_in: Path, output: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(video_in), "-i", str(audio_in),
         "-c:v", "copy", "-c:a", "copy",
         "-map", "0:v:0", "-map", "1:a:0",
         "-shortest", str(output)],
        check=True,
    )


def copy_video_only(video_in: Path, output: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_in), "-c", "copy", str(output)],
        check=True,
    )
