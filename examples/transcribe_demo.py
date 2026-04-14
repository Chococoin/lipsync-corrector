"""Demo: transcribe a video file end-to-end.

Usage:
    uv run python examples/transcribe_demo.py <video_path> [<output_stem>]

Produces <output_stem>.json and <output_stem>.srt next to the video (or
at the provided stem). Also extracts the audio track as <output_stem>.wav.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.transcription import transcribe, write_json, write_srt  # noqa: E402
from core.video_io import ensure_ffmpeg, extract_audio_as_pcm_wav  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    video = Path(argv[1])
    stem = Path(argv[2]) if len(argv) > 2 else video.with_suffix("")
    if not video.exists():
        print(f"error: video not found: {video}", file=sys.stderr)
        return 1

    ensure_ffmpeg()
    audio = stem.with_suffix(".wav")
    extract_audio_as_pcm_wav(video, audio)
    print(f"Extracted audio to {audio}")

    result = transcribe(audio, model_size="medium")
    print(
        f"Transcribed {result.duration:.1f}s of {result.language} "
        f"in {len(result.segments)} segments"
    )

    json_path = stem.with_suffix(".json")
    srt_path = stem.with_suffix(".srt")
    write_json(result, json_path)
    write_srt(result, srt_path)
    print(f"Wrote {json_path} and {srt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
