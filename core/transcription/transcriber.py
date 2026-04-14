"""Whisper transcription wrapper built on mlx-whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import mlx_whisper

from core.transcription.models import Segment, Transcription, Word

_ALLOWED_MODEL_SIZES = frozenset({"tiny", "base", "small", "medium", "large-v3"})


def transcribe(
    audio_path: Path | str,
    model_size: str = "medium",
    language: Optional[str] = None,
) -> Transcription:
    """Transcribe an audio file with Whisper via MLX.

    Args:
        audio_path: Path to an audio file readable by ffmpeg
            (wav, mp3, m4a, ogg, etc.).
        model_size: One of "tiny", "base", "small", "medium", "large-v3".
        language: Optional ISO language code (e.g. "es", "en"). If None,
            the model auto-detects from the first ~30 seconds of audio.

    Returns:
        An immutable Transcription with segment- and word-level timestamps.

    Raises:
        FileNotFoundError: if audio_path does not exist.
        ValueError: if model_size is not one of the allowed values.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")
    if model_size not in _ALLOWED_MODEL_SIZES:
        raise ValueError(
            f"unknown model_size: {model_size!r}. "
            f"Expected one of: {sorted(_ALLOWED_MODEL_SIZES)}"
        )

    repo_id = f"mlx-community/whisper-{model_size}-mlx"
    raw = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=repo_id,
        word_timestamps=True,
        language=language,
    )
    return _adapt_result(raw, model_size=model_size)


def _adapt_result(raw: dict[str, Any], model_size: str) -> Transcription:
    """Convert the mlx-whisper result dict into our Transcription dataclass.

    Only the fields we declare are copied; extra fields from future
    mlx-whisper versions are silently ignored.
    """
    segments = tuple(_adapt_segment(s) for s in raw.get("segments", ()))
    duration = segments[-1].end if segments else 0.0
    return Transcription(
        language=raw.get("language", ""),
        segments=segments,
        duration=float(duration),
        model_size=model_size,
    )


def _adapt_segment(raw: dict[str, Any]) -> Segment:
    words = tuple(_adapt_word(w) for w in raw.get("words", ()))
    return Segment(
        text=raw.get("text", ""),
        start=float(raw.get("start", 0.0)),
        end=float(raw.get("end", 0.0)),
        words=words,
        avg_logprob=float(raw.get("avg_logprob", 0.0)),
        no_speech_prob=float(raw.get("no_speech_prob", 0.0)),
    )


def _adapt_word(raw: dict[str, Any]) -> Word:
    # mlx-whisper uses the key "word" (singular) in the raw dict.
    return Word(
        text=raw.get("word", raw.get("text", "")),
        start=float(raw.get("start", 0.0)),
        end=float(raw.get("end", 0.0)),
        probability=float(raw.get("probability", 0.0)),
    )
