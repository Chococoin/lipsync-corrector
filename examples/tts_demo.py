"""Demo: generate dubbed audio from a translated transcription.

Usage:
    uv run python examples/tts_demo.py <translated_json> <source_audio_wav> [<output_wav>]

Reads a translated Transcription JSON, clones the speaker's voice from
the source audio, and generates a dubbed WAV in the target language.
First run downloads the XTTS-v2 model (~1.8 GB).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.transcription.models import Segment, Transcription, Word  # noqa: E402
from core.tts import synthesize  # noqa: E402


def _load_transcription(path: Path) -> Transcription:
    """Deserialize a Transcription JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = tuple(
        Segment(
            text=s["text"],
            start=s["start"],
            end=s["end"],
            words=tuple(
                Word(
                    text=w["text"],
                    start=w["start"],
                    end=w["end"],
                    probability=w["probability"],
                )
                for w in s.get("words", [])
            ),
            avg_logprob=s["avg_logprob"],
            no_speech_prob=s["no_speech_prob"],
        )
        for s in data["segments"]
    )
    return Transcription(
        language=data["language"],
        segments=segments,
        duration=data["duration"],
        model_size=data["model_size"],
    )


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    translated_json = Path(argv[1])
    source_audio = Path(argv[2])
    output_wav = (
        Path(argv[3])
        if len(argv) > 3
        else translated_json.with_suffix(".dubbed.wav")
    )

    if not translated_json.exists():
        print(f"error: JSON not found: {translated_json}", file=sys.stderr)
        return 1
    if not source_audio.exists():
        print(f"error: source audio not found: {source_audio}", file=sys.stderr)
        return 1

    transcription = _load_transcription(translated_json)
    print(
        f"Loaded {len(transcription.segments)} segments "
        f"in {transcription.language} ({transcription.duration:.1f}s)"
    )

    print("Synthesizing (first run downloads XTTS-v2 model, ~1.8 GB)...")
    result = synthesize(transcription, source_audio, output_wav)

    import soundfile as sf
    data, sr = sf.read(str(result))
    print(f"Wrote {result} ({len(data)/sr:.1f}s at {sr} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
