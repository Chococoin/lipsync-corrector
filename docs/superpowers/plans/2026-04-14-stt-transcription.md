# STT Transcription Sub-project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Python-API-only speech-to-text module (`core/transcription/`) built on `mlx-whisper` that produces a `Transcription` dataclass from an audio file, plus JSON and SRT serializers and a standalone demo script.

**Architecture:** A new `core/transcription/` subpackage with three single-responsibility files — `models.py` (frozen dataclasses), `serializers.py` (JSON + SRT writers), `transcriber.py` (the `transcribe` function that wraps `mlx_whisper.transcribe` and adapts the result into our dataclasses). No CLI changes. An `examples/transcribe_demo.py` script chains the existing `extract_audio` helper with the new transcriber for end-to-end verification.

**Tech Stack:** Python 3.11, `mlx-whisper` (MLX-native Whisper for Apple Silicon, which pulls in `mlx`). Default model `medium` (~1.5 GB downloaded automatically to the HuggingFace Hub cache on first run). No changes to PyTorch, librosa, or the existing Wav2Lip stack.

**Branch:** `stt-transcription` off `main` (currently at `23f10f5`).

**Design spec:** `docs/superpowers/specs/2026-04-14-stt-transcription-design.md`.

---

## File Structure (end state of this sub-project)

```
lipsync-corrector/
├── core/
│   └── transcription/
│       ├── __init__.py                # NEW: re-exports 6 public names
│       ├── models.py                  # NEW: Word, Segment, Transcription dataclasses
│       ├── serializers.py             # NEW: write_json, write_srt, _format_srt_timestamp
│       └── transcriber.py             # NEW: transcribe function + _adapt_result helper
├── tests/
│   ├── test_transcription_models.py       # NEW
│   ├── test_transcription_serializers.py  # NEW
│   └── test_transcriber.py                # NEW (inference tests marked skipif)
├── examples/
│   └── transcribe_demo.py             # NEW: standalone script
├── docs/milestones/
│   └── stt-transcription.md           # NEW: written at end of task 5
├── pyproject.toml                     # MODIFIED: adds mlx-whisper
└── README.md                          # MODIFIED: adds transcription demo section
```

---

## Task 1: Add mlx-whisper dependency

**Files:**
- Modify: `~/Projects/lipsync-corrector/pyproject.toml`

- [ ] **Step 1: Create the feature branch**

```bash
cd ~/Projects/lipsync-corrector
git checkout -b stt-transcription
```

- [ ] **Step 2: Edit `pyproject.toml`**

Replace the `dependencies = [...]` block with:

```toml
dependencies = [
    "insightface>=0.7.3",
    "onnxruntime>=1.17.0",
    "opencv-python>=4.9.0",
    "numpy>=1.26,<2.0",
    "torch>=2.1,<3.0",
    "librosa>=0.10,<0.11",
    "soundfile>=0.12",
    "mlx-whisper>=0.4,<1.0",
]
```

- [ ] **Step 3: Sync env**

```bash
cd ~/Projects/lipsync-corrector
uv sync
```

Expected: installs `mlx`, `mlx-whisper`, and their transitive deps (numba already present from librosa, tiktoken may be added). Adds ~300-500 MB to `.venv`.

- [ ] **Step 4: Smoke test mlx-whisper import**

```bash
uv run python -c "
import mlx_whisper
import mlx.core as mx
print('mlx_whisper:', mlx_whisper.__name__)
print('mlx default device:', mx.default_device())
print('has transcribe:', hasattr(mlx_whisper, 'transcribe'))
"
```

Expected: prints `mlx_whisper: mlx_whisper`, `mlx default device: Device(gpu, 0)` (or similar), and `has transcribe: True`.

- [ ] **Step 5: Run existing suite — verify no regressions**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 111 passed. The new dep should not affect any existing test.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add pyproject.toml uv.lock
git commit -m "deps: add mlx-whisper for speech-to-text transcription"
```

---

## Task 2: Dataclasses (`core/transcription/models.py`)

**Files:**
- Create: `~/Projects/lipsync-corrector/core/transcription/__init__.py`
- Create: `~/Projects/lipsync-corrector/core/transcription/models.py`
- Create: `~/Projects/lipsync-corrector/tests/test_transcription_models.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_transcription_models.py`:

```python
from dataclasses import FrozenInstanceError

import pytest

from core.transcription.models import Segment, Transcription, Word


def _word(text="hola", start=0.0, end=0.5, probability=0.95):
    return Word(text=text, start=start, end=end, probability=probability)


def _segment(
    text="hola mundo",
    start=0.0,
    end=1.0,
    words=None,
    avg_logprob=-0.3,
    no_speech_prob=0.01,
):
    if words is None:
        words = (_word(),)
    return Segment(
        text=text,
        start=start,
        end=end,
        words=words,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


class TestWord:
    def test_construction(self):
        w = _word()
        assert w.text == "hola"
        assert w.start == 0.0
        assert w.end == 0.5
        assert w.probability == 0.95

    def test_is_frozen(self):
        w = _word()
        with pytest.raises(FrozenInstanceError):
            w.text = "otra"  # type: ignore[misc]

    def test_is_hashable(self):
        w = _word()
        hash(w)


class TestSegment:
    def test_construction(self):
        s = _segment()
        assert s.text == "hola mundo"
        assert len(s.words) == 1
        assert s.avg_logprob == -0.3

    def test_is_frozen(self):
        s = _segment()
        with pytest.raises(FrozenInstanceError):
            s.text = "otro"  # type: ignore[misc]

    def test_is_hashable(self):
        s = _segment()
        hash(s)

    def test_words_is_tuple(self):
        s = _segment()
        assert isinstance(s.words, tuple)

    def test_empty_words_allowed(self):
        s = _segment(words=())
        assert s.words == ()


class TestTranscription:
    def test_construction(self):
        t = Transcription(
            language="es",
            segments=(_segment(),),
            duration=1.0,
            model_size="medium",
        )
        assert t.language == "es"
        assert len(t.segments) == 1
        assert t.duration == 1.0
        assert t.model_size == "medium"

    def test_is_frozen(self):
        t = Transcription(
            language="es", segments=(), duration=0.0, model_size="medium"
        )
        with pytest.raises(FrozenInstanceError):
            t.language = "en"  # type: ignore[misc]

    def test_is_hashable(self):
        t = Transcription(
            language="es",
            segments=(_segment(),),
            duration=1.0,
            model_size="medium",
        )
        hash(t)

    def test_segments_is_tuple(self):
        t = Transcription(
            language="es", segments=(), duration=0.0, model_size="medium"
        )
        assert isinstance(t.segments, tuple)

    def test_empty_segments_allowed(self):
        t = Transcription(
            language="es", segments=(), duration=0.0, model_size="medium"
        )
        assert t.segments == ()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_transcription_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.transcription'`.

- [ ] **Step 3: Create `core/transcription/models.py`**

```python
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
```

- [ ] **Step 4: Create `core/transcription/__init__.py`**

```python
"""Speech-to-text transcription subpackage.

Exposes a Python API: transcribe an audio file with Whisper (via MLX),
receive an immutable Transcription, and optionally serialize to JSON or SRT.
"""

from core.transcription.models import Segment, Transcription, Word

__all__ = ["Segment", "Transcription", "Word"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_transcription_models.py -v
```

Expected: 13 passed.

- [ ] **Step 6: Run full suite**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 124 passed (111 existing + 13 new).

- [ ] **Step 7: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/transcription/__init__.py core/transcription/models.py tests/test_transcription_models.py
git commit -m "feat: add Word, Segment, Transcription dataclasses"
```

---

## Task 3: Serializers (`core/transcription/serializers.py`)

**Files:**
- Create: `~/Projects/lipsync-corrector/core/transcription/serializers.py`
- Modify: `~/Projects/lipsync-corrector/core/transcription/__init__.py`
- Create: `~/Projects/lipsync-corrector/tests/test_transcription_serializers.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_transcription_serializers.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_transcription_serializers.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.transcription.serializers'`.

- [ ] **Step 3: Create `core/transcription/serializers.py`**

```python
"""JSON and SRT serializers for Transcription objects."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from core.transcription.models import Transcription


def write_json(transcription: Transcription, out_path: Path | str) -> None:
    """Write a Transcription to a pretty-printed UTF-8 JSON file.

    Tuples are serialized as JSON arrays. Non-ASCII characters are preserved
    in the output (ensure_ascii=False).
    """
    data = asdict(transcription)
    out = Path(out_path)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_srt(transcription: Transcription, out_path: Path | str) -> None:
    """Write a Transcription as a SubRip (.srt) subtitle file.

    One subtitle per segment. Indices are 1-based. Timestamps use the
    standard HH:MM:SS,mmm format.
    """
    out = Path(out_path)
    blocks = []
    for i, seg in enumerate(transcription.segments, start=1):
        start_ts = _format_srt_timestamp(seg.start)
        end_ts = _format_srt_timestamp(seg.end)
        blocks.append(f"{i}\n{start_ts} --> {end_ts}\n{seg.text}\n")
    out.write_text("\n".join(blocks), encoding="utf-8")


def _format_srt_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to SRT timestamp HH:MM:SS,mmm."""
    total_ms = int(round(seconds * 1000))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
```

- [ ] **Step 4: Extend `core/transcription/__init__.py`**

Replace the current file with:

```python
"""Speech-to-text transcription subpackage.

Exposes a Python API: transcribe an audio file with Whisper (via MLX),
receive an immutable Transcription, and optionally serialize to JSON or SRT.
"""

from core.transcription.models import Segment, Transcription, Word
from core.transcription.serializers import write_json, write_srt

__all__ = ["Segment", "Transcription", "Word", "write_json", "write_srt"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_transcription_serializers.py -v
```

Expected: 16 passed.

- [ ] **Step 6: Run full suite**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 140 passed (124 existing + 16 new).

- [ ] **Step 7: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/transcription/serializers.py core/transcription/__init__.py tests/test_transcription_serializers.py
git commit -m "feat: add JSON and SRT serializers for Transcription"
```

---

## Task 4: Transcriber (`core/transcription/transcriber.py`)

**Files:**
- Create: `~/Projects/lipsync-corrector/core/transcription/transcriber.py`
- Modify: `~/Projects/lipsync-corrector/core/transcription/__init__.py`
- Create: `~/Projects/lipsync-corrector/tests/test_transcriber.py`

**Background:** This task wires `mlx_whisper.transcribe` into our dataclasses. The function must validate its inputs before touching mlx-whisper (so validation errors raise fast without attempting a model download). The adapter extracts only the fields our dataclasses declare, ignoring anything extra that `mlx-whisper` emits. The inference tests require the `medium` model cached in `~/.cache/huggingface/hub/models--mlx-community--whisper-medium-mlx/` AND `examples/veo_audio_16k_mono.wav` on disk; they skip when either is absent.

- [ ] **Step 1: Write failing tests**

Create `tests/test_transcriber.py`:

```python
from pathlib import Path

import pytest

from core.transcription.models import Transcription
from core.transcription.transcriber import transcribe

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
WEIGHTS_CACHED = (HF_CACHE / "models--mlx-community--whisper-medium-mlx").exists()
requires_weights = pytest.mark.skipif(
    not WEIGHTS_CACHED,
    reason="mlx-community/whisper-medium-mlx not in HF cache",
)

VEO_AUDIO = Path(__file__).parent.parent / "examples" / "veo_audio_16k_mono.wav"
requires_audio = pytest.mark.skipif(
    not VEO_AUDIO.exists(),
    reason="examples/veo_audio_16k_mono.wav not present",
)


class TestUnitValidation:
    def test_missing_audio_raises_filenotfound(self, tmp_path):
        missing = tmp_path / "not_there.wav"
        with pytest.raises(FileNotFoundError) as exc:
            transcribe(missing)
        assert "not_there.wav" in str(exc.value)

    def test_invalid_model_size_raises_valueerror(self, tmp_path):
        fake = tmp_path / "fake.wav"
        fake.write_bytes(b"RIFF0000")
        with pytest.raises(ValueError) as exc:
            transcribe(fake, model_size="notreal")
        assert "notreal" in str(exc.value)


@requires_weights
@requires_audio
class TestInference:
    def test_returns_transcription(self):
        result = transcribe(VEO_AUDIO)
        assert isinstance(result, Transcription)

    def test_detects_spanish(self):
        result = transcribe(VEO_AUDIO)
        assert result.language == "es"

    def test_has_at_least_one_segment(self):
        result = transcribe(VEO_AUDIO)
        assert len(result.segments) >= 1

    def test_duration_is_positive(self):
        result = transcribe(VEO_AUDIO)
        assert result.duration > 0.0

    def test_model_size_stored(self):
        result = transcribe(VEO_AUDIO)
        assert result.model_size == "medium"

    def test_words_are_monotonic_within_segment(self):
        result = transcribe(VEO_AUDIO)
        for seg in result.segments:
            for w in seg.words:
                assert w.start < w.end
            for a, b in zip(seg.words, seg.words[1:]):
                assert a.start <= b.start

    def test_forced_language_es_still_returns_es(self):
        result = transcribe(VEO_AUDIO, language="es")
        assert result.language == "es"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_transcriber.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.transcription.transcriber'`.

- [ ] **Step 3: Create `core/transcription/transcriber.py`**

```python
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
```

- [ ] **Step 4: Extend `core/transcription/__init__.py`**

Replace the current file with:

```python
"""Speech-to-text transcription subpackage.

Exposes a Python API: transcribe an audio file with Whisper (via MLX),
receive an immutable Transcription, and optionally serialize to JSON or SRT.
"""

from core.transcription.models import Segment, Transcription, Word
from core.transcription.serializers import write_json, write_srt
from core.transcription.transcriber import transcribe

__all__ = [
    "Segment",
    "Transcription",
    "Word",
    "transcribe",
    "write_json",
    "write_srt",
]
```

- [ ] **Step 5: Run unit tests (no weights needed)**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_transcriber.py::TestUnitValidation -v
```

Expected: 2 passed.

- [ ] **Step 6: Run full transcriber tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_transcriber.py -v
```

Expected, if weights are NOT cached (first time on this machine): 2 passed, 7 skipped with the reason `mlx-community/whisper-medium-mlx not in HF cache`.

This is the expected outcome for this task. The actual inference tests will run in Task 5 after the first real transcription triggers the download.

- [ ] **Step 7: Run full suite**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 142 passed, 7 skipped (if no weights yet).

- [ ] **Step 8: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/transcription/transcriber.py core/transcription/__init__.py tests/test_transcriber.py
git commit -m "feat: add transcribe function wrapping mlx-whisper"
```

---

## Task 5: Demo script + first-run download + milestone notes

**Files:**
- Create: `~/Projects/lipsync-corrector/examples/transcribe_demo.py`
- Modify: `~/Projects/lipsync-corrector/README.md`
- Create: `~/Projects/lipsync-corrector/docs/milestones/stt-transcription.md`

**Background:** This task produces the end-to-end observable output of the sub-project: a real transcription of the Veo clip with JSON + SRT files. It also triggers the first-time download of the `medium` Whisper model (~1.5 GB) into the HuggingFace Hub cache. After the download, the inference tests in `test_transcriber.py` stop skipping and start passing.

- [ ] **Step 1: Create `examples/transcribe_demo.py`**

```python
"""Demo: transcribe a video file end-to-end.

Usage:
    uv run python examples/transcribe_demo.py <video_path> [<output_stem>]

Produces <output_stem>.json and <output_stem>.srt next to the video (or
at the provided stem). Also extracts the audio track as <output_stem>.wav.
"""

from __future__ import annotations

import sys
from pathlib import Path

from core.transcription import transcribe, write_json, write_srt
from core.video_io import ensure_ffmpeg, extract_audio


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
    extract_audio(video, audio)
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
```

- [ ] **Step 2: Run the demo on the Veo clip (first-time download)**

```bash
cd ~/Projects/lipsync-corrector
uv run python examples/transcribe_demo.py \
  ~/Downloads/Video_De_Mujer_Saludando_Generado.mp4 \
  examples/veo_stt
```

Expected first-time output: mlx-whisper prints progress as it downloads `mlx-community/whisper-medium-mlx` weights (~1.5 GB, takes a few minutes depending on network). Then the script prints:

```
Extracted audio to examples/veo_stt.wav
Transcribed 8.0s of es in <N> segments
Wrote examples/veo_stt.json and examples/veo_stt.srt
```

(The exact duration and segment count depend on the clip.)

- [ ] **Step 3: Inspect the JSON and SRT outputs**

```bash
cd ~/Projects/lipsync-corrector
head -40 examples/veo_stt.json
echo "---"
cat examples/veo_stt.srt
```

Expected JSON head: pretty-printed with `language`, `segments`, `duration`, `model_size` at the top level, segments contain `text` and `words` arrays.

Expected SRT: numbered blocks like `1\n00:00:00,000 --> 00:00:02,500\n<spanish text>\n`.

Record the actual Spanish transcription text for the milestone notes.

- [ ] **Step 4: Re-run inference tests now that weights are cached**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_transcriber.py -v
```

Expected: all 9 tests pass (2 unit + 7 inference). No more skips.

- [ ] **Step 5: Run the full suite**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 149 passed (142 existing + 7 inference tests that were previously skipped).

- [ ] **Step 6: Add a README section for transcription**

Open `README.md` and append at the end:

```markdown
## Transcription demo (sub-project: auto-dub STT)

The `core.transcription` module exposes a Python API for speech-to-text
via `mlx-whisper`. A standalone demo script transcribes a video end-to-end:

```bash
uv run python examples/transcribe_demo.py <video_path> [<output_stem>]
```

Produces `<stem>.wav`, `<stem>.json` (canonical data with word-level
timestamps), and `<stem>.srt` (human-readable subtitles).

First run downloads the default `medium` Whisper checkpoint (~1.5 GB) to
`~/.cache/huggingface/hub/`. Subsequent runs use the cached weights.
```

- [ ] **Step 7: Clean up the example outputs**

```bash
cd ~/Projects/lipsync-corrector
rm -f examples/veo_stt.wav examples/veo_stt.json examples/veo_stt.srt
```

The extracted WAV, JSON, and SRT files are all git-ignored (via the
`examples/*.wav`, `*.json` is not in the ignore list — add it if needed,
but since it's a demo output, the simplest thing is to delete it). If
`examples/*.json` is not in `.gitignore`, either delete the file (chosen
here) or add the pattern.

- [ ] **Step 8: Write the milestone notes**

Create `docs/milestones/stt-transcription.md`:

```markdown
# STT Transcription Sub-project

**Date completed:** <YYYY-MM-DD>
**Track:** Auto-dub sub-project 1 of 3 (STT → translation → TTS)
**Status:** Done

## What was built

- `core/transcription/` subpackage with three focused modules:
  - `models.py` — frozen dataclasses `Word`, `Segment`, `Transcription`.
  - `serializers.py` — `write_json` and `write_srt` for on-disk output.
  - `transcriber.py` — `transcribe(audio_path, model_size, language)`
    wrapping `mlx_whisper.transcribe` and adapting the raw dict into
    our dataclasses.
- `examples/transcribe_demo.py` — standalone script chaining
  `extract_audio` + `transcribe` + serializers for manual E2E runs.
- 38 new tests (13 model + 16 serializer + 9 transcriber) for a total
  suite of 149 tests.
- README section documenting the demo and the first-run weight download.

## How to run

```bash
uv run python examples/transcribe_demo.py path/to/video.mp4
```

Produces `video.wav`, `video.json`, `video.srt` next to the input video.

## Measured results

End-to-end run on the Veo-generated clip:

- Input: `~/Downloads/Video_De_Mujer_Saludando_Generado.mp4` (8 s audio,
  Spanish)
- Detected language: **es**
- Segments: **<N>** (fill in)
- Duration: **<N> s** (fill in)
- Wall time (excluding first-time download): **<N> s** (fill in)
- Model: `mlx-community/whisper-medium-mlx` (~1.5 GB cached in HF hub)

### Sample output (first few segments)

```
<paste 2-3 segments from the SRT here>
```

## What was learned

- <fill in after running>

## Deferred to future sub-projects

- Translation of the transcribed text to another language.
- Voice cloning / TTS in the target language.
- Duration reconciliation between the new audio and the original video.
- Integration with `cli/main.py` under a unified `dub` subcommand.

## Next sub-project

**Sub-project 2: Translation.** Consume the `Transcription` object
produced here and emit a translated version (same structure, different
language). Brainstorm the provider choice (Claude API vs DeepL vs local
NLLB) and duration-preserving strategy before writing a plan.
```

Fill in the `<placeholders>` after running.

- [ ] **Step 9: Run the full suite one last time**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 149 passed.

- [ ] **Step 10: Commit milestone notes and README**

```bash
cd ~/Projects/lipsync-corrector
git add examples/transcribe_demo.py README.md docs/milestones/stt-transcription.md
git commit -m "feat: stt-transcription sub-project complete"
```

---

## Done criteria for this sub-project

- `uv run pytest` passes all tests (149 with weights cached and audio fixture present; 142 + 7 skipped otherwise).
- `uv run python examples/transcribe_demo.py <real-video>` produces a valid JSON file with segment- and word-level timestamps, a valid SRT file, and a .wav audio extraction, with no crashes.
- `core/transcription/` subpackage exists with 4 files following the plan's file structure.
- `from core.transcription import transcribe, write_json, write_srt, Transcription, Segment, Word` works.
- `docs/milestones/stt-transcription.md` written with actual measurements.
- README documents the demo and the first-run download.
- Everything committed on `stt-transcription` branch, ready to merge to `main`.
- No existing tests regress. No changes to `cli/main.py`, `core/lipsync_model.py`, `core/wav2lip_model.py`, `core/wav2lip/`, or any Wav2Lip pipeline code.

Translation and TTS sub-projects are out of scope. Do not start them in the same session.
