"""Lip-sync corrector CLI — Track B entry point."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from core.video_io import (
    VideoReader,
    VideoWriter,
    ensure_ffmpeg,
    extract_audio,
    has_audio_stream,
    mux_video_audio,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lipsync",
        description="Lip-sync corrector for auto-dubbed videos.",
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--output", required=True, help="Path to the output video.")
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to the dubbed audio (not yet implemented, reserved for Milestone 3).",
    )
    parser.add_argument(
        "--debug-tracking",
        action="store_true",
        default=False,
        help="Overlay face-tracking bounding boxes on the output video.",
    )
    parser.add_argument(
        "--lipsync",
        action="store_true",
        default=False,
        help="Run the full lipsync pipeline (track → crop → model → blend → write). "
             "Milestone 3a uses an IdentityModel placeholder.",
    )
    parser.add_argument(
        "--model",
        choices=["identity", "wav2lip"],
        default="identity",
        help="Which LipSyncModel to use. 'identity' is the placeholder that passes "
             "crops through unchanged (default). 'wav2lip' runs the real pretrained "
             "Wav2Lip GAN model and requires models/wav2lip_gan.pth to be present.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    ensure_ffmpeg()

    video_path = Path(args.video)
    output_path = Path(args.output)

    if not video_path.exists():
        print(f"error: video not found: {video_path}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = None
    if args.debug_tracking:
        from core.face_tracker import FaceTracker, draw_tracking_overlay
        print("Loading face tracker...")
        tracker = FaceTracker()

    lipsync_tracker = None
    lipsync_model = None
    crop_face_region = None
    blend_back = None
    if args.lipsync:
        from core.face_tracker import FaceTracker
        from core.mouth_region import crop_face_region
        from core.blender import blend_back
        lipsync_tracker = FaceTracker()
        if args.model == "identity":
            from core.lipsync_model import IdentityModel
            print("Loading lip-sync pipeline (placeholder IdentityModel)...")
            lipsync_model = IdentityModel()
        else:
            import core.wav2lip_model as _w2l_mod
            from core.wav2lip_model import Wav2LipModel
            print("Loading lip-sync pipeline (Wav2LipModel)...")
            with VideoReader(video_path) as _probe:
                _fps = _probe.fps
            try:
                lipsync_model = Wav2LipModel(fps=_fps, checkpoint_path=_w2l_mod.DEFAULT_CHECKPOINT_PATH)
            except FileNotFoundError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1

    with VideoReader(video_path) as reader:
        print(f"Input: {video_path} ({reader.frame_count} frames, {reader.fps:.1f} fps, {reader.width}x{reader.height})")

        with tempfile.TemporaryDirectory(prefix="lipsync-") as tmp:
            tmp_dir = Path(tmp)
            intermediate = tmp_dir / "video_only.mp4"

            with VideoWriter(intermediate, fps=reader.fps, width=reader.width, height=reader.height) as writer:
                for frame in reader:
                    if tracker is not None:
                        tracked = tracker.track(frame)
                        frame = draw_tracking_overlay(frame, tracked)
                    elif lipsync_tracker is not None:
                        tracked = lipsync_tracker.track(frame)
                        if tracked is not None:
                            face_crop = crop_face_region(frame, tracked)
                            processed = lipsync_model.process([face_crop.image], args.audio)
                            frame = blend_back(frame, processed[0], face_crop, mouth_only=True)
                    writer.write(frame)
                print(f"Wrote {writer.frames_written} frames to intermediate.")

            if has_audio_stream(video_path):
                audio_tmp = tmp_dir / "audio.aac"
                extract_audio(video_path, audio_tmp)
                mux_video_audio(intermediate, audio_tmp, output_path)
                print("Audio preserved.")
            else:
                shutil.copy2(str(intermediate), str(output_path))
                print("No audio stream in input, copied video only.")

    print(f"Done. Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
