"""Track A: Face-swap quick win. See docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md."""

from __future__ import annotations

import argparse
import sys
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
