# STT Transcription Sub-project — Design Spec

**Date:** 2026-04-14
**Status:** Approved for implementation
**Author:** chocos (with Claude)
**Predecessor:** `2026-04-11-lipsync-corrector-design.md`

## 1. Purpose

Add speech-to-text capability to the lipsync-corrector project as the first step of a three-part auto-dubbing workflow. Given an audio file, produce a structured transcription with segment-level and word-level timestamps that downstream sub-projects (translation, TTS) can consume.

This sub-project supersedes the explicit non-goal "We do not translate text" from the predecessor spec. The parent project's purpose is unchanged: correct lip-sync on auto-dubbed videos. What changes is that we now build the dubbing audio ourselves rather than accept it as an external input.

## 2. Scope

**In scope for this sub-project:**

- A `core/transcription/` Python subpackage exposing `transcribe()` and two serializers.
- Three immutable dataclasses for the result shape: `Word`, `Segment`, `Transcription`.
- A standalone demo script at `examples/transcribe_demo.py` that chains the existing audio extractor with the new transcriber and writes JSON + SRT output.
- Unit tests for dataclasses and serializers (no weights required).
- Inference tests marked `skipif` when the model weights are not cached locally.

**Explicitly out of scope** (deferred to future sub-projects):

- Translation of the transcribed text.
- Text-to-speech regeneration in another language.
- Duration reconciliation between original audio and synthesized audio.
- Voice cloning.
- Integration of any of the above into `cli/main.py`.
- A production CLI entry point for transcription. The demo script is the only way to run this sub-project end-to-end.

## 3. Constraints and Decisions

All technical decisions below were made during brainstorming and are not open to re-litigation during implementation. They are recorded here for traceability.

1. **Whisper backend:** `mlx-whisper` (MLX-native, Apple Silicon only). Chosen over `faster-whisper` because the parent project's hardware constraint is "Apple Silicon M4 only" and MLX is the fastest path on that hardware.
2. **Default model size:** `medium` (~1.5 GB). Configurable via the `model_size` argument.
3. **Language handling:** optional parameter. `None` (default) auto-detects; passing an ISO code (e.g. `"es"`) forces that language and skips the detection pass.
4. **Input format:** the function accepts audio only, not video. Callers with a video extract audio first using the existing `core.video_io.extract_audio`. This preserves the module's single responsibility.
5. **Output format:** an immutable `Transcription` dataclass returned by the function. Serialization is a separate concern handled by `write_json` and `write_srt` helpers. The JSON file is the canonical data format and includes word-level timestamps; the SRT is a human-readable bonus with segment-level timing.
6. **Integration shape:** no CLI changes in this sub-project. Expose a Python API only. The demo script in `examples/` is the only executable path. A proper CLI (likely a `dub` subcommand) will be designed when all three steps of the auto-dub flow exist.
7. **Weight storage:** delegated to HuggingFace Hub's cache via `mlx-whisper` defaults. No manual download script. No `models/` directory entry. First run pulls ~1.5 GB from `mlx-community/whisper-medium-mlx`; subsequent runs are cached in `~/.cache/huggingface/hub/`.

## 4. File Structure

```
lipsync-corrector/
├── core/
│   └── transcription/
│       ├── __init__.py            # re-exports the 6 public names
│       ├── models.py              # Word, Segment, Transcription (frozen dataclasses)
│       ├── transcriber.py         # transcribe(audio_path, ...) -> Transcription
│       └── serializers.py         # write_json, write_srt
├── tests/
│   ├── test_transcription_models.py       # frozen, hashable, tuple-backed
│   ├── test_transcription_serializers.py  # JSON round-trip, SRT formatting
│   └── test_transcriber.py                # real inference, skipif no weights
├── examples/
│   └── transcribe_demo.py         # minimal end-to-end demo
└── pyproject.toml                  # + mlx-whisper dependency
```

**Responsibilities per file:**

- `models.py` — dataclasses only, zero logic. `frozen=True` everywhere; collections are `tuple`s so instances are hashable.
- `transcriber.py` — one public function. Validates inputs, resolves the HF repo id, calls `mlx_whisper.transcribe`, adapts the raw dict result into our dataclasses via a private `_adapt_result` helper. No disk I/O beyond reading the provided audio file.
- `serializers.py` — two functions (`write_json`, `write_srt`) plus a private `_format_srt_timestamp` helper. Both take a `Transcription` and a `Path` and write to disk.
- `__init__.py` — re-exports: `Word`, `Segment`, `Transcription`, `transcribe`, `write_json`, `write_srt`.
- `examples/transcribe_demo.py` — a ~25-line script that calls `extract_audio` → `transcribe` → `write_json` → `write_srt`, for manual end-to-end verification.

## 5. Data Types

Defined in `core/transcription/models.py`:

```python
@dataclass(frozen=True)
class Word:
    text: str
    start: float              # seconds from audio start
    end: float
    probability: float        # 0.0-1.0

@dataclass(frozen=True)
class Segment:
    text: str
    start: float
    end: float
    words: tuple[Word, ...]   # tuple (not list) so Segment is hashable
    avg_logprob: float
    no_speech_prob: float

@dataclass(frozen=True)
class Transcription:
    language: str             # ISO code like "es", "en", "fr"
    segments: tuple[Segment, ...]
    duration: float           # total audio duration in seconds
    model_size: str           # e.g. "medium" — recorded for traceability
```

All times are in seconds as floats. All fields are copied directly from the dict that `mlx_whisper.transcribe()` returns; we do not derive, smooth, or post-process anything. Any additional fields that `mlx-whisper` may emit in future versions are ignored by the adapter — this gives us immunity to non-breaking upstream changes.

## 6. API Surface

Public, in `core/transcription/__init__.py`:

```python
def transcribe(
    audio_path: Path | str,
    model_size: str = "medium",
    language: Optional[str] = None,
) -> Transcription: ...

def write_json(transcription: Transcription, out_path: Path | str) -> None: ...

def write_srt(transcription: Transcription, out_path: Path | str) -> None: ...
```

**`transcribe` semantics:**

1. Validates that `audio_path` exists; raises `FileNotFoundError` otherwise.
2. Validates `model_size ∈ {"tiny", "base", "small", "medium", "large-v3"}`; raises `ValueError` otherwise.
3. Resolves the HuggingFace repo id to `f"mlx-community/whisper-{model_size}-mlx"`.
4. Calls `mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=repo_id, word_timestamps=True, language=language)`.
5. Adapts the resulting dict into a `Transcription` dataclass (via `_adapt_result`) and returns it.

**`write_json` semantics:** serializes via `dataclasses.asdict` + `json.dump(indent=2, ensure_ascii=False)`. Tuples are emitted as JSON arrays. The file is UTF-8.

**`write_srt` semantics:** emits one SRT subtitle per `Segment`. Indices are 1-based. Timestamps use the standard `HH:MM:SS,mmm` format (comma, not period, before the milliseconds). Subtitles are separated by a blank line.

## 7. Error Handling

Three real error cases, handled without over-engineering:

1. **Audio missing or unreadable:** the function raises `FileNotFoundError` on its own validation. If `mlx-whisper` fails to decode an existing-but-invalid audio file (corrupt header, unknown codec, etc.), its internal ffmpeg call raises — we let that propagate without wrapping.
2. **Invalid `model_size`:** `ValueError` with the accepted set included in the message.
3. **Weight download failure (no network on first run):** the HuggingFace client's exceptions propagate directly. No retry, no timeout tuning, no local fallback.

We do **not** introduce a project-specific exception hierarchy, logging system, or progress reporting. Whatever `mlx-whisper` prints during downloads is visible on stderr as-is.

The serializers can only fail with standard `OSError` variants (`PermissionError`, disk full, etc.). We let those propagate naturally.

## 8. Testing Strategy

Three test files covering three categories:

### `tests/test_transcription_models.py` — no weights

Roughly 7 tests. Verifies that:
- `Word`, `Segment`, `Transcription` are all frozen (mutation raises `FrozenInstanceError`).
- Instances are hashable (because `frozen=True` + tuples everywhere).
- `Segment.words` is a tuple, not a list.
- Empty `segments` and empty `words` are permitted.

### `tests/test_transcription_serializers.py` — no weights

Roughly 10 tests. Constructs `Transcription` objects in memory and verifies that:
- `write_json` produces valid UTF-8 that round-trips through `json.load`.
- JSON contains every dataclass field.
- JSON preserves non-ASCII characters (`ensure_ascii=False`).
- `write_srt` emits 1-indexed subtitle blocks.
- `write_srt` uses the comma-separated timestamp format.
- `write_srt` separates subtitles with a blank line.
- The private `_format_srt_timestamp` helper produces correct strings for 0 s, 1.234 s, 65.5 s, 3600 s, and 3665.789 s.

### `tests/test_transcriber.py` — requires weights

Roughly 5 tests. Verifies:
- `transcribe("not_there.wav", ...)` raises `FileNotFoundError`.
- `transcribe(audio, model_size="notreal")` raises `ValueError` before any model is touched.
- `transcribe(real_audio)` returns a `Transcription` with `language == "es"` (the Veo clip is Spanish), at least one segment, and `duration > 0`.
- Forcing `language="es"` produces a result with the same `language` field.
- All `Word.start < Word.end`, and all words within a segment are in ascending temporal order.

**Skip markers:**

```python
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
WEIGHTS_CACHED = (HF_CACHE / "models--mlx-community--whisper-medium-mlx").exists()
requires_weights = pytest.mark.skipif(
    not WEIGHTS_CACHED,
    reason="mlx-community/whisper-medium-mlx not downloaded",
)

VEO_AUDIO = Path(__file__).parent.parent / "examples" / "veo_audio_16k_mono.wav"
requires_audio = pytest.mark.skipif(
    not VEO_AUDIO.exists(),
    reason="veo_audio_16k_mono.wav not present",
)
```

Inference tests use both markers. The fixture audio is the `examples/veo_audio_16k_mono.wav` file produced during Milestone 3b; it is git-ignored and lives only on machines that have run 3b end-to-end.

**Expected totals:**
- 7 model tests + 10 serializer tests + 5 inference tests = **~22 new tests**.
- Full suite grows from 111 to roughly **133** with both weights and audio present. On a clean machine without the weights cached, inference tests skip and the count is closer to 128.

## 9. Demo Script

`examples/transcribe_demo.py`, approximately 25 lines:

```python
"""Demo: transcribe a video file end-to-end.

Usage:
    uv run python examples/transcribe_demo.py <video_path> [<output_stem>]

Produces <output_stem>.json and <output_stem>.srt next to the video.
"""

import sys
from pathlib import Path

from core.video_io import ensure_ffmpeg, extract_audio
from core.transcription import transcribe, write_json, write_srt


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

The demo is not tested — it is an assembler of pieces that are tested individually.

## 10. Dependencies

One new Python dependency: `mlx-whisper>=0.4,<1.0` (transitive: `mlx`, `numba`, etc.). Added to `pyproject.toml` in the first task of the implementation plan.

Approximate `.venv` size growth: ~200-400 MB for the `mlx` runtime. The model weights (~1.5 GB) live in the HuggingFace cache outside the project directory and do not count toward repository or `.venv` size.

## 11. Relationship to the Parent Design Spec

The parent spec (`2026-04-11-lipsync-corrector-design.md`, section 1) lists as an explicit non-goal: *"We do not translate text."* This sub-project lifts that non-goal for the scope of the auto-dub workflow (STT → translation → TTS). The parent project's other non-goals remain in effect:

- We do not clone or generate voices (voice cloning will be addressed in a future sub-project if chosen).
- We do not replace face identity.
- We do not attempt cinema-grade VFX quality.

Sub-project milestones for the auto-dub workflow will be numbered independently of the parent project's milestone plan (which still has Milestone 4: blending and Milestone 6: LatentSync upgrade pending). The user is working non-linearly; the numbering will be reconciled when a future dub-integration sub-project lands.

## 12. What Success Looks Like

After implementation:

- `uv run python examples/transcribe_demo.py ~/Downloads/Video_De_Mujer_Saludando_Generado.mp4` produces a `.json` file containing the full Spanish transcript of the clip with segment- and word-level timestamps, plus a `.srt` file openable in any video editor.
- All ~22 new tests pass locally. On a machine without the weights cached, the inference tests skip cleanly.
- The module `core.transcription` is import-stable: `from core.transcription import transcribe, write_json, write_srt, Transcription, Segment, Word` works.
- No existing tests regress. The suite count goes from 111 to 133 (or 128 if weights are absent).

## 13. Open Questions (Deferred)

- Which translation provider to use in the next sub-project (Claude API vs DeepL vs local NLLB). Not blocking for this sub-project.
- Duration reconciliation strategy for the TTS sub-project (time-stretching vs frame-dropping vs regeneration with timing constraints). Not blocking for this sub-project.
- When to fold the three sub-projects into a unified `dub` CLI command. To be decided after the TTS sub-project lands and the real integration shape is clear.
