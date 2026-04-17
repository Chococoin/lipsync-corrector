# TTS Sub-project (Coqui XTTS-v2) — Design Spec

**Date:** 2026-04-17
**Status:** Approved for implementation
**Author:** chocos (with Claude)
**Predecessors:**
- `2026-04-14-stt-transcription-design.md` (sub-project 1: produces `Transcription`)
- `2026-04-14-translation-design.md` (sub-project 2: produces translated `Transcription`)

## 1. Purpose

Add text-to-speech capability to the lipsync-corrector project as the third and final step of the auto-dubbing workflow (STT → translation → TTS). Given a translated `Transcription` and the original video's audio (as a voice reference), produce a dubbed WAV file where each segment is spoken in the target language with the original speaker's cloned voice, positioned at the correct timestamps to match the original video's timing.

## 2. Scope

**In scope:**

- A `core/tts/` Python subpackage with three modules: `reference.py` (voice reference selection), `assembler.py` (temporal audio assembly), and `synthesizer.py` (the public `synthesize()` function wrapping Coqui XTTS-v2).
- Voice cloning from the original speaker's voice using ~6 seconds of automatically selected reference audio.
- Per-segment audio generation to maintain temporal alignment with the video.
- Assembly of individual segment audios into a single WAV track with correct timing (silences between segments matching original gaps).
- A standalone `examples/tts_demo.py` script.
- Unit tests for reference selection and audio assembly (no model needed).
- Integration test requiring the XTTS model (marked skipif).
- README section and milestone notes.

**Explicitly out of scope:**

- A unified `dub` CLI command chaining STT → translation → TTS → Wav2Lip. That is a separate integration milestone after this sub-project.
- Sample rate conversion (24kHz → 16kHz for Wav2Lip). Handled in the integration milestone.
- Multi-speaker support. One speaker per video.
- Fine-tuning or training of the voice cloning model.
- Quality evaluation metrics (MOS, PESQ). Success is auditory.
- Duration control via XTTS `speed` parameter. The translation step already produces text of the correct length via duration-aware word count targets. TTS generates at natural speed.

## 3. Constraints and Decisions

1. **Provider:** Coqui XTTS-v2 via the `TTS` pip package. Open-source, runs locally on Apple Silicon (MPS via PyTorch). Model weights (~1.8 GB) auto-downloaded from HuggingFace on first use.

2. **Voice cloning reference:** automatically selected from the source video's audio. The module picks the segments with lowest `no_speech_prob` (from the STT transcription) until ≥6 seconds of reference audio is accumulated. If the total audio is shorter than 6 seconds, all segments are used (fallback to full audio). The selected segments are returned in temporal order for natural-sounding reference.

3. **Per-segment generation:** one XTTS call per translated segment. Each segment's audio is generated independently with the same voice reference. The generated audios are then assembled into a single track using the original segment timestamps for positioning.

4. **Duration reconciliation:** handled upstream in the translation step. The translation prompt includes target word counts per segment based on the target language's speaking rate. XTTS generates at `speed=1.0` (natural pace). If a generated segment slightly exceeds its time slot, it is truncated. If shorter, silence fills the remainder.

5. **Output format:** WAV, 24 kHz mono (XTTS-v2's native sample rate). The integration milestone will handle downsampling to 16 kHz for Wav2Lip.

6. **Integration shape:** core module + demo script. No CLI changes. Same pattern as STT and translation sub-projects.

7. **Testing:** hybrid. Unit tests for reference selection and assembly logic (pure numpy, always run). Integration test for real synthesis (skipif model not downloaded).

8. **Error strategy:** exceptions propagate from the `TTS` library unchanged. Our own `FileNotFoundError` for missing source audio. Empty transcriptions produce a silent WAV without calling the model.

## 4. File Structure

```
lipsync-corrector/
├── core/
│   └── tts/
│       ├── __init__.py            # re-exports synthesize
│       ├── reference.py           # select_reference_segments + extract_reference_audio
│       ├── assembler.py           # assemble_track
│       └── synthesizer.py         # the public synthesize() function
├── tests/
│   ├── test_tts_reference.py      # reference selection logic (no model)
│   ├── test_tts_assembler.py      # audio assembly with gaps (no model)
│   └── test_tts_synthesizer.py    # unit + integration (skipif no model)
├── examples/
│   └── tts_demo.py                # E2E demo script
├── docs/milestones/
│   └── tts.md                     # milestone notes
├── pyproject.toml                  # + TTS dependency
└── README.md                       # new section
```

**Responsibilities:**

- `reference.py` — two pure functions. `select_reference_segments` sorts segments by `no_speech_prob` ascending, accumulates until ≥6s, returns them in temporal order. `extract_reference_audio` slices the source WAV array at the segment timestamps and concatenates.

- `assembler.py` — one function. `assemble_track` creates a silence array of `total_duration × sample_rate`, then pastes each segment's generated audio at its `start` position, truncating if it overflows the slot.

- `synthesizer.py` — the public `synthesize()` function. Loads source audio, selects reference, saves reference to a temp WAV (XTTS needs a file path), loads the XTTS model, generates audio per-segment, assembles the track, writes the output WAV.

- `tts_demo.py` — standalone script: loads translated JSON + source WAV, calls `synthesize`, prints duration.

## 5. Public API

```python
# core/tts/__init__.py
from core.tts.synthesizer import synthesize

__all__ = ["synthesize"]
```

```python
# core/tts/synthesizer.py
def synthesize(
    transcription: Transcription,
    source_audio_path: Path | str,
    output_path: Path | str,
    *,
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
) -> Path:
    """Generate a dubbed audio track from a translated Transcription.

    Uses Coqui XTTS-v2 with voice cloning from the source audio.
    Each segment is generated independently and assembled into a
    single WAV at the correct timestamps.

    Args:
        transcription: Translated Transcription (target language).
        source_audio_path: Path to the original video's audio (WAV,
            used for voice cloning reference).
        output_path: Where to write the output WAV (24 kHz mono).
        model_name: Coqui TTS model identifier.

    Returns:
        Path to the written output file.

    Raises:
        FileNotFoundError: if source_audio_path doesn't exist.
    """
```

### Internal helpers (in `reference.py`)

```python
def select_reference_segments(
    transcription: Transcription,
    min_duration: float = 6.0,
) -> list[Segment]:
    """Select the best segments for voice cloning reference."""

def extract_reference_audio(
    source_wav: np.ndarray,
    segments: list[Segment],
    sample_rate: int = 16000,
) -> np.ndarray:
    """Extract and concatenate audio slices for the selected segments."""
```

### Internal helpers (in `assembler.py`)

```python
def assemble_track(
    segment_audios: list[np.ndarray],
    segments: tuple[Segment, ...],
    total_duration: float,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Position generated audios at their segment timestamps in a
    silence track of total_duration length."""
```

## 6. Reference Audio Selection

**Algorithm:**

1. Sort segments by `no_speech_prob` ascending (most likely to be clean speech first).
2. Accumulate segments in this order until total duration ≥ `min_duration` (default 6.0 seconds).
3. If all segments together are shorter than `min_duration`, use all of them (fallback).
4. Re-sort the selected segments by `start` time (temporal order) before returning.

**Why 6 seconds:** XTTS-v2 works with as little as 3 seconds but quality improves significantly with 6-10 seconds. For the Veo clip (~8s total audio), this uses most of the available speech.

**`extract_reference_audio`:** reads the source WAV (1D float32 numpy array loaded by the caller via `soundfile`), slices `[int(seg.start * sr) : int(seg.end * sr)]` for each selected segment, concatenates them into one array. Pure numpy, no I/O.

## 7. Audio Assembly

**Algorithm for `assemble_track`:**

1. Create `np.zeros(int(total_duration * sample_rate), dtype=np.float32)`.
2. For each `(audio, segment)` pair:
   - `start_sample = int(segment.start * sample_rate)`
   - `slot_length = int((segment.end - segment.start) * sample_rate)`
   - If `len(audio) > slot_length`: truncate audio to `slot_length`.
   - If `len(audio) <= slot_length`: use as-is (remaining slot stays silent).
   - Copy audio into the track at `start_sample`.
3. Return the track.

**Truncation rationale:** thanks to duration-aware translation, most segments should fit naturally. Truncation is a safety net — losing the last fraction of a second is better than overlapping with the next segment.

## 8. Error Handling

| Case | Behavior |
|---|---|
| `source_audio_path` missing | `FileNotFoundError` |
| Transcription empty | Write silent WAV of `total_duration`. No model loaded. |
| Model not downloaded (first run) | `TTS` library downloads ~1.8 GB automatically. Network error propagates. |
| XTTS fails on a segment | Exception propagates with segment context. |
| Generated audio overflows slot | Truncated to slot length silently. |
| MPS op not supported | Falls back to CPU via `get_torch_device()`. Runtime MPS errors propagate. |

No retry, no custom exception hierarchy, no logging layer.

## 9. Testing Strategy

### `tests/test_tts_reference.py` (~8 tests, no model)

- Selects segments with lowest `no_speech_prob` first.
- Accumulates to ≥6 seconds.
- Falls back to all segments if total < 6s.
- Returns segments in temporal order (sorted by `start`).
- `extract_reference_audio` returns correct slices from a synthetic array.
- Single long segment → returns just that slice.
- Segments covering full audio → returns full audio.
- Empty segments list → returns empty array.

### `tests/test_tts_assembler.py` (~6 tests, no model)

- Output array has correct total length.
- Segment audio positioned at correct timestamp.
- Audio longer than slot is truncated.
- Audio shorter than slot leaves silence.
- Multiple segments with gaps → silence in gaps.
- Zero total_duration → empty array.

### `tests/test_tts_synthesizer.py` (~3 tests)

- `test_missing_source_raises_filenotfound` (unit, always runs).
- `test_empty_segments_writes_silence` (unit, always runs).
- `test_synthesize_produces_valid_wav` (integration, skipif no model).

**Skip marker:**

```python
import shutil
XTTS_CACHED = shutil.which("tts") is not None or \
    Path.home().joinpath(".local/share/tts").exists()
requires_xtts = pytest.mark.skipif(
    not XTTS_CACHED,
    reason="XTTS model not available",
)
```

Note: the exact cache detection may need adjustment based on how the `TTS` package stores its models. The integration test is a canary — its skip logic can be refined after the first real run.

### Expected test counts

- reference: ~8
- assembler: ~6
- synthesizer: ~3
- **Total new: ~17**
- Suite: 191 → ~208 with model, ~207 without.

## 10. Demo Script

`examples/tts_demo.py` (~40 lines):

```bash
uv run python examples/tts_demo.py \
  examples/veo_stt.en.json \
  examples/veo_audio_16k_mono.wav \
  examples/veo_dubbed_en.wav
```

1. Loads the translated JSON (reuses `_load_transcription` helper pattern from translate_demo).
2. Calls `synthesize(transcription, source_audio, output)`.
3. Prints total duration and output path.

The output WAV can be opened in QuickTime to listen to the cloned voice speaking in English.

## 11. Dependencies

New: `TTS>=0.22,<1.0` (the Coqui TTS package from PyPI).

Transitive deps added: `trainer`, `coqpit`, `gruut`, and others. Several overlap with existing deps (torch, numpy, soundfile). Estimated .venv growth: ~200-400 MB for new packages. Model weights (~1.8 GB) live in the TTS cache directory outside the project.

## 12. What Success Looks Like

After implementation:

- `uv run python examples/tts_demo.py examples/veo_stt.en.json examples/veo_audio_16k_mono.wav examples/veo_dubbed_en.wav` produces a WAV where a voice resembling the original Spanish speaker says the translated English text ("Hello everyone, welcome to Cartagena...") with each segment positioned at the correct timestamps.
- All unit tests pass without the model downloaded.
- The integration test passes with the model downloaded.
- Suite count grows from 191 to ~208.
- No changes to `cli/main.py` or any existing module.

## 13. Relationship to the Auto-Dub Pipeline

This sub-project completes the three-piece auto-dub workflow:

```
Video → STT (sub-project 1) → Transcription (ES)
     → Translation (sub-project 2) → Transcription (EN)
     → TTS (this sub-project) → dubbed_audio.wav (EN, cloned voice)
     → Wav2Lip (milestone 3b) → video with synced lips
```

A future integration milestone will chain these into a single `dub` command in `cli/main.py`. That milestone also handles:
- Sample rate conversion (24 kHz TTS output → 16 kHz for Wav2Lip).
- End-to-end orchestration (video in → video out with dubbed audio and synced lips).
- Error handling across the full pipeline.

## 14. Open Questions (Deferred)

- **MPS compatibility of XTTS-v2.** It should work (standard PyTorch ops) but is not guaranteed. If MPS fails at runtime, CPU fallback is the workaround. Not blocking — we discover this during the first real run.
- **Quality of voice cloning with only ~6 seconds of reference.** Sufficient for a demo; may need more reference audio for production quality. Not blocking for this milestone.
- **Whether `speed` parameter tuning is needed after the duration-aware translation.** If segments consistently overflow their slots despite correct word counts, we can add speed adjustment. Deferred — evaluate after the first E2E run.
