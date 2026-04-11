"""Track A: Face-swap quick win. See docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md."""

from __future__ import annotations

import argparse
from typing import Optional


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
