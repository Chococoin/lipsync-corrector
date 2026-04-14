"""Immutable data types for speech-to-text transcription results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    end: float
    words: tuple[Word, ...]
    avg_logprob: float
    no_speech_prob: float


@dataclass(frozen=True)
class Transcription:
    language: str
    segments: tuple[Segment, ...]
    duration: float
    model_size: str
