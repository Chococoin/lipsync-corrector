import json

import pytest

from core.transcription.models import Segment, Transcription, Word
from core.transcription.serializers import (
    _format_srt_timestamp,
    write_json,
    write_srt,
)


def _transcription():
    return Transcription(
        language="es",
        segments=(
            Segment(
                text="Hola, ¿cómo están?",
                start=0.0,
                end=1.5,
                words=(
                    Word(text="Hola", start=0.0, end=0.42, probability=0.98),
                    Word(text="¿cómo", start=0.5, end=0.8, probability=0.97),
                    Word(text="están?", start=0.85, end=1.5, probability=0.96),
                ),
                avg_logprob=-0.3,
                no_speech_prob=0.01,
            ),
            Segment(
                text="Hoy hablamos.",
                start=2.0,
                end=3.2,
                words=(
                    Word(text="Hoy", start=2.0, end=2.3, probability=0.95),
                    Word(text="hablamos.", start=2.4, end=3.2, probability=0.94),
                ),
                avg_logprob=-0.25,
                no_speech_prob=0.02,
            ),
        ),
        duration=3.2,
        model_size="medium",
    )


class TestWriteJson:
    def test_creates_file(self, tmp_path):
        t = _transcription()
        out = tmp_path / "out.json"
        write_json(t, out)
        assert out.exists()

    def test_round_trip_top_level_fields(self, tmp_path):
        t = _transcription()
        out = tmp_path / "out.json"
        write_json(t, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["language"] == "es"
        assert data["duration"] == 3.2
        assert data["model_size"] == "medium"

    def test_contains_segments_and_words(self, tmp_path):
        t = _transcription()
        out = tmp_path / "out.json"
        write_json(t, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["segments"]) == 2
        assert len(data["segments"][0]["words"]) == 3
        assert data["segments"][0]["words"][0]["text"] == "Hola"

    def test_preserves_non_ascii(self, tmp_path):
        t = _transcription()
        out = tmp_path / "out.json"
        write_json(t, out)
        raw = out.read_text(encoding="utf-8")
        assert "¿cómo" in raw  # must not be escaped to \u00bf\u00f3

    def test_accepts_str_path(self, tmp_path):
        t = _transcription()
        out = tmp_path / "out.json"
        write_json(t, str(out))
        assert out.exists()


class TestWriteSrt:
    def test_creates_file(self, tmp_path):
        t = _transcription()
        out = tmp_path / "out.srt"
        write_srt(t, out)
        assert out.exists()

    def test_is_one_indexed(self, tmp_path):
        t = _transcription()
        out = tmp_path / "out.srt"
        write_srt(t, out)
        content = out.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert lines[0] == "1"

    def test_uses_comma_timestamp_format(self, tmp_path):
        t = _transcription()
        out = tmp_path / "out.srt"
        write_srt(t, out)
        content = out.read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:01,500" in content

    def test_second_subtitle_indexed_two(self, tmp_path):
        t = _transcription()
        out = tmp_path / "out.srt"
        write_srt(t, out)
        content = out.read_text(encoding="utf-8")
        assert "\n2\n" in content

    def test_separates_subtitles_with_blank_line(self, tmp_path):
        t = _transcription()
        out = tmp_path / "out.srt"
        write_srt(t, out)
        content = out.read_text(encoding="utf-8")
        assert "\n\n" in content

    def test_contains_segment_text(self, tmp_path):
        t = _transcription()
        out = tmp_path / "out.srt"
        write_srt(t, out)
        content = out.read_text(encoding="utf-8")
        assert "Hola, ¿cómo están?" in content
        assert "Hoy hablamos." in content


class TestFormatSrtTimestamp:
    def test_zero(self):
        assert _format_srt_timestamp(0.0) == "00:00:00,000"

    def test_sub_second(self):
        assert _format_srt_timestamp(1.234) == "00:00:01,234"

    def test_over_one_minute(self):
        assert _format_srt_timestamp(65.5) == "00:01:05,500"

    def test_exact_hour(self):
        assert _format_srt_timestamp(3600.0) == "01:00:00,000"

    def test_fractional_hour(self):
        assert _format_srt_timestamp(3665.789) == "01:01:05,789"
