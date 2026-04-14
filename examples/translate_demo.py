"""Demo: translate a transcription JSON into another language.

Usage:
    uv run python examples/translate_demo.py <input_json> <target_language> [<output_json>]

Reads a Transcription JSON produced by transcribe_demo.py, translates it
via Claude API, and writes the result as a new JSON. Requires
ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.transcription.models import Segment, Transcription, Word  # noqa: E402
from core.transcription.serializers import write_json  # noqa: E402
from core.translation import translate  # noqa: E402


def _load_transcription(path: Path) -> Transcription:
    """Deserialize a Transcription JSON written by transcribe_demo.py."""
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
    input_json = Path(argv[1])
    target_language = argv[2]
    output_json = (
        Path(argv[3])
        if len(argv) > 3
        else input_json.with_name(f"{input_json.stem}.{target_language}.json")
    )

    if not input_json.exists():
        print(f"error: input not found: {input_json}", file=sys.stderr)
        return 1

    source = _load_transcription(input_json)
    print(f"Loaded {len(source.segments)} segments in {source.language}")

    result = translate(source, target_language=target_language)
    print(f"Translated to {result.language}")

    write_json(result, output_json)
    print(f"Wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
