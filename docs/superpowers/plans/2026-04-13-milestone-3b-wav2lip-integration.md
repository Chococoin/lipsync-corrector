# Milestone 3b: Wav2Lip Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder `IdentityModel` with a real `Wav2LipModel` that produces lip-synced output on the Veo test clip, proving the full pipeline works end-to-end with a real ML model.

**Architecture:** Isolate all Wav2Lip-specific code in a `core/wav2lip/` subpackage (the neural network class, audio preprocessing, and video-frame↔mel-chunk alignment). Then write `core/wav2lip_model.py` that exposes `Wav2LipModel(LipSyncModel)` — a drop-in replacement for `IdentityModel`. The CLI gains a `--model {identity,wav2lip}` flag with `identity` as the default so that running tests without downloaded weights still works. Milestone 4 will later handle blending improvements — this milestone stays strictly on the model integration.

**Tech Stack:** Python 3.11, PyTorch (MPS backend) for model inference, librosa for audio loading + mel spectrogram, soundfile as the librosa audio backend. Adds roughly 1.5 GB to `.venv` (torch is heavy). The Wav2Lip checkpoint `wav2lip_gan.pth` (~420 MB) is downloaded manually by the engineer to `models/` — it is not fetched by the code at test time.

**Repo:** `~/Projects/lipsync-corrector` on `main`, branching to `milestone-3b`.

---

## File Structure (end state of this milestone)

```
lipsync-corrector/
├── core/
│   ├── wav2lip/                      # NEW subpackage
│   │   ├── __init__.py               # NEW: re-exports Wav2Lip class and load helper
│   │   ├── conv.py                   # NEW: Conv2d/Conv2dTranspose helpers (fetched)
│   │   ├── model.py                  # NEW: Wav2Lip nn.Module class (fetched)
│   │   ├── audio.py                  # NEW: mel spectrogram extraction + hparams
│   │   └── frame_sync.py             # NEW: video frame index → mel chunk alignment
│   ├── wav2lip_model.py              # NEW: Wav2LipModel(LipSyncModel) — the public wrapper
│   ├── blender.py                    # unchanged
│   ├── mouth_region.py               # unchanged
│   ├── lipsync_model.py              # unchanged
│   ├── face_tracker.py               # unchanged
│   ├── video_io.py                   # unchanged
│   └── device.py                     # unchanged
├── cli/
│   └── main.py                       # MODIFIED: add --model flag
├── tests/
│   ├── test_wav2lip_audio.py         # NEW
│   ├── test_wav2lip_frame_sync.py    # NEW
│   ├── test_wav2lip_model_unit.py    # NEW: tests that don't need weights
│   ├── test_wav2lip_model_inference.py # NEW: marked as requiring weights
│   ├── test_cli.py                   # MODIFIED: --model flag tests
│   └── ...                           # other test files unchanged
├── models/
│   └── wav2lip_gan.pth               # DOWNLOADED MANUALLY (git-ignored)
├── docs/milestones/
│   └── milestone-3b.md               # NEW: written at end
├── pyproject.toml                    # MODIFIED: new deps
└── README.md                         # MODIFIED: add weight download instructions
```

**Responsibility per new/changed file:**

- `core/wav2lip/model.py` + `conv.py` — the exact PyTorch `nn.Module` architecture that matches the published `wav2lip_gan.pth` checkpoint. Fetched from the Rudrabha reference repo at a pinned commit. No logic of ours; this file is effectively a versioned third-party vendor.
- `core/wav2lip/audio.py` — loads audio from any path with `soundfile`, resamples to 16 kHz mono, computes an 80-band mel spectrogram using the exact hyperparameters the pretrained Wav2Lip was trained with (window=800, hop=200, n_mels=80, fmin=55, fmax=7600). Also adapted from the reference repo — those constants must match exactly.
- `core/wav2lip/frame_sync.py` — pure math: given a video frame index `i`, a video fps, and the mel spectrogram, return the 16-column mel chunk aligned to frame `i`. Fully unit-testable without any models.
- `core/wav2lip_model.py` — the public `Wav2LipModel(LipSyncModel)` class. Constructor takes a checkpoint path and device, loads the `nn.Module`, sets eval mode. `process(face_crops, audio_path)` runs the full batched inference: builds 6-channel face tensors (masked + full), loads audio once, iterates frames pulling mel chunks via `frame_sync`, calls the model in batches, returns the modified crops.
- `cli/main.py` — new `--model {identity,wav2lip}` flag. Default is `identity` so existing tests keep passing without weights. When `--model wav2lip`, the CLI instantiates `Wav2LipModel` instead of `IdentityModel`. If the checkpoint is missing, print a clear error with the expected path and exit 1.

---

## Task 1: Add PyTorch + librosa dependencies

**Files:**
- Modify: `~/Projects/lipsync-corrector/pyproject.toml`

- [ ] **Step 1: Create the feature branch**

```bash
cd ~/Projects/lipsync-corrector
git checkout -b milestone-3b
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
]
```

- [ ] **Step 3: Sync env**

```bash
cd ~/Projects/lipsync-corrector
uv sync
```

Expected: downloads torch wheels (~750 MB), librosa + numba + llvmlite stack. Takes a few minutes the first time.

- [ ] **Step 4: Smoke test PyTorch MPS**

```bash
uv run python -c "
import torch
print('torch:', torch.__version__)
print('mps available:', torch.backends.mps.is_available())
x = torch.randn(3, 4).to('mps')
print('tensor device:', x.device)
print('sum:', x.sum().item())
"
```

Expected output includes `mps available: True`, `tensor device: mps:0`, and a finite float for `sum`.

- [ ] **Step 5: Smoke test librosa**

```bash
uv run python -c "
import numpy as np
import librosa
t = np.linspace(0, 1, 16000, endpoint=False)
y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
m = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=80)
print('mel shape:', m.shape)
print('mel dtype:', m.dtype)
"
```

Expected: mel shape `(80, N)` with `N > 20`, dtype float32.

- [ ] **Step 6: Run existing suite — verify no regressions**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 84 passed. The new deps should not affect any existing test.

- [ ] **Step 7: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add pyproject.toml uv.lock
git commit -m "deps: add torch and librosa for wav2lip integration"
```

---

## Task 2: Vendor the Wav2Lip `nn.Module` class

**Files:**
- Create: `~/Projects/lipsync-corrector/core/wav2lip/__init__.py`
- Create: `~/Projects/lipsync-corrector/core/wav2lip/conv.py`
- Create: `~/Projects/lipsync-corrector/core/wav2lip/model.py`
- Create: `~/Projects/lipsync-corrector/tests/test_wav2lip_model_unit.py`

**Background:** The Wav2Lip paper (Rudrabha et al., ACM MM 2020) ships reference code in `github.com/Rudrabha/Wav2Lip`. The `nn.Module` that matches the published `wav2lip_gan.pth` checkpoint lives in `models/wav2lip.py` and depends on helpers in `models/conv.py`. We copy those two files verbatim — they are the only way to load the pretrained weights correctly. If the upstream repo changes the class, weight loading will fail with a `state_dict` mismatch. That is why we fetch from the `master` branch via `curl` and pin a local copy.

- [ ] **Step 1: Create subpackage and fetch files**

```bash
cd ~/Projects/lipsync-corrector
mkdir -p core/wav2lip
curl -fsSL https://raw.githubusercontent.com/Rudrabha/Wav2Lip/master/models/conv.py -o core/wav2lip/conv.py
curl -fsSL https://raw.githubusercontent.com/Rudrabha/Wav2Lip/master/models/wav2lip.py -o core/wav2lip/model.py
wc -l core/wav2lip/conv.py core/wav2lip/model.py
```

Expected: both files exist, `conv.py` around 40 lines, `model.py` around 180 lines.

If `curl` fails because the upstream repo is gone, use the identical files from any maintained fork (for example `https://raw.githubusercontent.com/indianajson/Wav2Lip/master/models/`). The interface we require is stable across forks.

- [ ] **Step 2: Adapt imports in `core/wav2lip/model.py`**

The fetched `model.py` starts with:

```python
from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d
```

That relative import already matches our layout (`core/wav2lip/model.py` importing from `core/wav2lip/conv.py`), so no change is needed. Verify the first non-comment line is this import — if it says `from models.conv import ...` instead, change it to the relative import above.

- [ ] **Step 3: Create `core/wav2lip/__init__.py`**

```python
"""Wav2Lip third-party vendor subpackage.

Contains the reference nn.Module architecture and audio preprocessing
adapted from https://github.com/Rudrabha/Wav2Lip. These files are vendored
so that weight loading against the published wav2lip_gan.pth checkpoint is
reproducible and not dependent on an external repo remaining online.
"""

from core.wav2lip.model import Wav2Lip

__all__ = ["Wav2Lip"]
```

- [ ] **Step 4: Write failing unit test**

Create `tests/test_wav2lip_model_unit.py`:

```python
import pytest
import torch

from core.wav2lip import Wav2Lip


class TestWav2LipArchitecture:
    def test_can_instantiate_on_cpu(self):
        model = Wav2Lip()
        assert isinstance(model, torch.nn.Module)

    def test_expected_input_output_shapes(self):
        model = Wav2Lip().eval()
        face = torch.zeros(1, 6, 96, 96)
        mel = torch.zeros(1, 1, 80, 16)
        with torch.no_grad():
            out = model(mel, face)
        assert out.shape == (1, 3, 96, 96)

    def test_batch_inference(self):
        model = Wav2Lip().eval()
        face = torch.zeros(4, 6, 96, 96)
        mel = torch.zeros(4, 1, 80, 16)
        with torch.no_grad():
            out = model(mel, face)
        assert out.shape == (4, 3, 96, 96)
```

- [ ] **Step 5: Run tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_wav2lip_model_unit.py -v
```

Expected: 3 passed. This confirms that the fetched class matches the architecture we expect: accepts `(N, 1, 80, 16)` mel + `(N, 6, 96, 96)` face, returns `(N, 3, 96, 96)`. If any test fails with a shape mismatch, the upstream file was modified — re-fetch from a known fork and try again.

- [ ] **Step 6: Run full suite**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 87 passed (84 existing + 3 new).

- [ ] **Step 7: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/wav2lip/__init__.py core/wav2lip/conv.py core/wav2lip/model.py tests/test_wav2lip_model_unit.py
git commit -m "feat: vendor Wav2Lip nn.Module from Rudrabha reference repo"
```

---

## Task 3: Audio preprocessing (`core/wav2lip/audio.py`)

**Files:**
- Create: `~/Projects/lipsync-corrector/core/wav2lip/audio.py`
- Create: `~/Projects/lipsync-corrector/tests/test_wav2lip_audio.py`

**Background:** Wav2Lip was trained with a very specific mel spectrogram recipe. Deviating from these hyperparameters even slightly produces garbage output from the pretrained model. The constants below are copied from `hparams.py` in the Rudrabha repo. Do not change them.

- [ ] **Step 1: Write failing tests**

Create `tests/test_wav2lip_audio.py`:

```python
import numpy as np
import pytest

from core.wav2lip.audio import (
    SAMPLE_RATE,
    N_MELS,
    load_wav_mono_16k,
    melspectrogram,
)


class TestConstants:
    def test_sample_rate_is_16k(self):
        assert SAMPLE_RATE == 16000

    def test_n_mels_is_80(self):
        assert N_MELS == 80


class TestLoadWavMono16k:
    def test_returns_float32_1d(self, tmp_path):
        import soundfile as sf
        t = np.linspace(0, 1.0, 48000, endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        path = tmp_path / "sine.wav"
        sf.write(str(path), y, 48000)
        out = load_wav_mono_16k(path)
        assert out.dtype == np.float32
        assert out.ndim == 1

    def test_resamples_to_16k(self, tmp_path):
        import soundfile as sf
        t = np.linspace(0, 1.0, 48000, endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        path = tmp_path / "sine.wav"
        sf.write(str(path), y, 48000)
        out = load_wav_mono_16k(path)
        assert abs(len(out) - 16000) < 100  # allow resampler edge effects

    def test_downmixes_stereo_to_mono(self, tmp_path):
        import soundfile as sf
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = 0.5 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
        stereo = np.stack([left, right], axis=1)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, 16000)
        out = load_wav_mono_16k(path)
        assert out.ndim == 1


class TestMelSpectrogram:
    def test_shape_has_80_mel_bands(self):
        y = np.random.randn(16000).astype(np.float32) * 0.1
        m = melspectrogram(y)
        assert m.shape[0] == 80

    def test_returns_float32(self):
        y = np.random.randn(16000).astype(np.float32) * 0.1
        m = melspectrogram(y)
        assert m.dtype == np.float32

    def test_silence_is_finite(self):
        y = np.zeros(16000, dtype=np.float32)
        m = melspectrogram(y)
        assert np.all(np.isfinite(m))

    def test_longer_audio_gives_more_time_frames(self):
        short = np.random.randn(16000).astype(np.float32) * 0.1
        long = np.random.randn(32000).astype(np.float32) * 0.1
        m_short = melspectrogram(short)
        m_long = melspectrogram(long)
        assert m_long.shape[1] > m_short.shape[1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_wav2lip_audio.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.wav2lip.audio'`.

- [ ] **Step 3: Implement `core/wav2lip/audio.py`**

```python
"""Audio preprocessing for Wav2Lip.

All constants below are copied verbatim from hparams.py in the Rudrabha
Wav2Lip reference repo. The pretrained wav2lip_gan.pth checkpoint was
trained with these exact values; changing them silently corrupts the
mel input and produces garbage output.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 800
HOP_SIZE = 200
WIN_SIZE = 800
FMIN = 55
FMAX = 7600
MIN_LEVEL_DB = -100.0
REF_LEVEL_DB = 20.0


def load_wav_mono_16k(path: Path | str) -> np.ndarray:
    """Load an audio file, downmix to mono, and resample to 16 kHz.

    Returns a 1-D float32 numpy array.
    """
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    return y


def _amp_to_db(x: np.ndarray) -> np.ndarray:
    min_level = np.exp(MIN_LEVEL_DB / 20.0 * np.log(10.0))
    return 20.0 * np.log10(np.maximum(min_level, x))


def _normalize(spec_db: np.ndarray) -> np.ndarray:
    return np.clip((spec_db - MIN_LEVEL_DB) / -MIN_LEVEL_DB, 0.0, 1.0)


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """Compute the 80-band log-mel spectrogram Wav2Lip expects.

    Input: 1-D float32 waveform at 16 kHz.
    Output: float32 array of shape (80, T).
    """
    stft = librosa.stft(
        y=wav,
        n_fft=N_FFT,
        hop_length=HOP_SIZE,
        win_length=WIN_SIZE,
    )
    mag = np.abs(stft)
    mel_basis = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )
    mel = mel_basis @ mag
    mel_db = _amp_to_db(mel) - REF_LEVEL_DB
    mel_norm = _normalize(mel_db)
    return mel_norm.astype(np.float32)
```

- [ ] **Step 4: Run tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_wav2lip_audio.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Run full suite**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 96 passed (87 + 9).

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/wav2lip/audio.py tests/test_wav2lip_audio.py
git commit -m "feat: add wav2lip audio preprocessing (mel spectrogram)"
```

---

## Task 4: Frame ↔ mel chunk alignment (`core/wav2lip/frame_sync.py`)

**Files:**
- Create: `~/Projects/lipsync-corrector/core/wav2lip/frame_sync.py`
- Create: `~/Projects/lipsync-corrector/tests/test_wav2lip_frame_sync.py`

**Background:** For each video frame `i` at `fps`, Wav2Lip needs a 16-column slice of the mel spectrogram centered on that frame's position in the audio. The reference implementation uses these exact numbers: the mel hop is 200 samples at 16 kHz, so there are 80 mel columns per second. The chunk start for frame `i` is `int(80 * i / fps)` and the chunk width is `MEL_STEP_SIZE = 16` columns. This is a small but error-prone piece of math; isolate it so it's trivially testable.

- [ ] **Step 1: Write failing tests**

Create `tests/test_wav2lip_frame_sync.py`:

```python
import numpy as np
import pytest

from core.wav2lip.frame_sync import MEL_STEP_SIZE, get_mel_chunk_for_frame


class TestConstants:
    def test_mel_step_size_is_16(self):
        assert MEL_STEP_SIZE == 16


class TestGetMelChunkForFrame:
    def _mel(self, t=500):
        return np.arange(80 * t, dtype=np.float32).reshape(80, t)

    def test_returns_correct_shape(self):
        mel = self._mel()
        chunk = get_mel_chunk_for_frame(mel, frame_idx=0, fps=25.0)
        assert chunk.shape == (80, 16)

    def test_frame_zero_starts_at_column_zero(self):
        mel = self._mel()
        chunk = get_mel_chunk_for_frame(mel, frame_idx=0, fps=25.0)
        np.testing.assert_array_equal(chunk, mel[:, 0:16])

    def test_frame_one_at_25fps_starts_at_column_3(self):
        # int(80 * 1 / 25) == 3
        mel = self._mel()
        chunk = get_mel_chunk_for_frame(mel, frame_idx=1, fps=25.0)
        np.testing.assert_array_equal(chunk, mel[:, 3:19])

    def test_frame_two_at_24fps_starts_at_column_6(self):
        # int(80 * 2 / 24) == 6
        mel = self._mel()
        chunk = get_mel_chunk_for_frame(mel, frame_idx=2, fps=24.0)
        np.testing.assert_array_equal(chunk, mel[:, 6:22])

    def test_near_end_is_left_padded(self):
        mel = self._mel(t=20)
        chunk = get_mel_chunk_for_frame(mel, frame_idx=10, fps=25.0)
        assert chunk.shape == (80, 16)

    def test_beyond_end_still_returns_shape_16(self):
        mel = self._mel(t=10)
        chunk = get_mel_chunk_for_frame(mel, frame_idx=100, fps=25.0)
        assert chunk.shape == (80, 16)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_wav2lip_frame_sync.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `core/wav2lip/frame_sync.py`**

```python
"""Frame index → mel chunk alignment for Wav2Lip inference."""

from __future__ import annotations

import numpy as np

MEL_STEP_SIZE = 16
MEL_COLS_PER_SECOND = 80  # 16000 / 200 (hop_size)


def get_mel_chunk_for_frame(
    mel: np.ndarray,
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    """Return the 80x16 mel chunk aligned to video frame `frame_idx`.

    The chunk start column is int(80 * frame_idx / fps). The chunk is 16
    columns wide. If the video extends past the end of the audio, the
    last valid window is returned (clamped to the tail of `mel`). The
    result is always shape (80, 16).
    """
    n_mel, n_cols = mel.shape
    start = int(MEL_COLS_PER_SECOND * frame_idx / fps)
    end = start + MEL_STEP_SIZE
    if end > n_cols:
        # Shift the window left so the chunk stays inside the mel
        end = n_cols
        start = max(0, end - MEL_STEP_SIZE)
    chunk = mel[:, start:end]
    if chunk.shape[1] < MEL_STEP_SIZE:
        # Audio is shorter than 16 mel columns total: right-pad with edge
        pad = MEL_STEP_SIZE - chunk.shape[1]
        chunk = np.pad(chunk, ((0, 0), (0, pad)), mode="edge")
    return chunk
```

- [ ] **Step 4: Run tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_wav2lip_frame_sync.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/wav2lip/frame_sync.py tests/test_wav2lip_frame_sync.py
git commit -m "feat: add wav2lip frame↔mel chunk alignment helper"
```

---

## Task 5: `Wav2LipModel` wrapper (`core/wav2lip_model.py`)

**Files:**
- Create: `~/Projects/lipsync-corrector/core/wav2lip_model.py`
- Create: `~/Projects/lipsync-corrector/tests/test_wav2lip_model_inference.py`

**Background:** This is the only file in the milestone that implements real business logic. It takes the pieces built in tasks 2–4 and glues them into a `LipSyncModel` concrete subclass. The inference flow is:

1. Load audio once via `load_wav_mono_16k`, compute `melspectrogram` once.
2. For each face crop `i`, build a mel chunk via `get_mel_chunk_for_frame` using the video fps stored on the model.
3. Build the face tensor: each crop becomes a 6-channel image where channels 0–2 are the crop with the lower half zeroed (the "identity reference"), and channels 3–5 are the full unmodified crop. Scale to `[0, 1]` float.
4. Batch N frames at a time, move to device, run the model, convert back to uint8 HWC.
5. Return the list of modified crops.

The `fps` is passed to the constructor because the `LipSyncModel.process` signature we already committed doesn't carry it — we inject it at construction time from the CLI.

- [ ] **Step 1: Write failing tests**

Create `tests/test_wav2lip_model_inference.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from core.wav2lip_model import Wav2LipModel, DEFAULT_CHECKPOINT_PATH


WEIGHTS_EXIST = DEFAULT_CHECKPOINT_PATH.exists()
requires_weights = pytest.mark.skipif(
    not WEIGHTS_EXIST,
    reason=f"wav2lip_gan.pth not present at {DEFAULT_CHECKPOINT_PATH}",
)


class TestConstructor:
    def test_missing_checkpoint_raises_filenotfound(self, tmp_path):
        missing = tmp_path / "not_there.pth"
        with pytest.raises(FileNotFoundError) as exc:
            Wav2LipModel(checkpoint_path=missing, fps=25.0)
        assert "not_there.pth" in str(exc.value)


@requires_weights
class TestWav2LipInference:
    def test_returns_same_number_of_crops(self, tmp_path):
        model = Wav2LipModel(fps=25.0)
        crops = [np.full((96, 96, 3), 128, dtype=np.uint8) for _ in range(5)]
        audio = _make_silence_wav(tmp_path, duration_s=1.0)
        result = model.process(crops, audio)
        assert len(result) == 5

    def test_returns_same_shapes_and_dtype(self, tmp_path):
        model = Wav2LipModel(fps=25.0)
        crops = [np.full((96, 96, 3), 128, dtype=np.uint8) for _ in range(3)]
        audio = _make_silence_wav(tmp_path, duration_s=1.0)
        result = model.process(crops, audio)
        for original, processed in zip(crops, result):
            assert processed.shape == (96, 96, 3)
            assert processed.dtype == np.uint8

    def test_output_differs_from_input(self, tmp_path):
        model = Wav2LipModel(fps=25.0)
        crop = np.full((96, 96, 3), 128, dtype=np.uint8)
        audio = _make_silence_wav(tmp_path, duration_s=1.0)
        result = model.process([crop], audio)
        # Wav2Lip should *not* return the input unchanged — that's the
        # visual signature that distinguishes it from IdentityModel.
        assert not np.array_equal(result[0], crop)


def _make_silence_wav(tmp_path: Path, duration_s: float) -> Path:
    import soundfile as sf
    n = int(16000 * duration_s)
    y = np.zeros(n, dtype=np.float32)
    path = tmp_path / "silence.wav"
    sf.write(str(path), y, 16000)
    return path
```

- [ ] **Step 2: Run tests to verify they fail (or skip)**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_wav2lip_model_inference.py -v
```

Expected: `ModuleNotFoundError: No module named 'core.wav2lip_model'`.

- [ ] **Step 3: Implement `core/wav2lip_model.py`**

```python
"""Wav2Lip concrete implementation of LipSyncModel."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.device import get_torch_device
from core.lipsync_model import LipSyncModel
from core.wav2lip import Wav2Lip
from core.wav2lip.audio import load_wav_mono_16k, melspectrogram
from core.wav2lip.frame_sync import MEL_STEP_SIZE, get_mel_chunk_for_frame

DEFAULT_CHECKPOINT_PATH = Path(__file__).parent.parent / "models" / "wav2lip_gan.pth"

IMG_SIZE = 96
INFERENCE_BATCH_SIZE = 16


class Wav2LipModel(LipSyncModel):
    """Real lip-sync model using the pretrained Wav2Lip GAN checkpoint.

    The constructor loads the checkpoint and prepares the model on the
    best available device (MPS on Apple Silicon, else CUDA, else CPU).
    The `fps` argument is the source video fps, needed for frame↔mel
    alignment during inference.
    """

    def __init__(
        self,
        fps: float,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        device: Optional[str] = None,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"wav2lip checkpoint not found: {checkpoint_path}. "
                f"Download wav2lip_gan.pth and place it at that path. "
                f"See README for the download link."
            )
        self.fps = float(fps)
        self.device = device or get_torch_device()
        self.model = self._load_model(checkpoint_path, self.device)

    @staticmethod
    def _load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
        # weights_only=False is required for legacy checkpoints like
        # wav2lip_gan.pth that contain a dict rather than a bare state_dict.
        # Torch >=2.6 defaults to True and will refuse to load them.
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=device,
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Some forks save with "module." prefix from DataParallel
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.removeprefix("module.")] = v
        model = Wav2Lip()
        model.load_state_dict(cleaned)
        model = model.to(device)
        model.eval()
        return model

    def process(
        self,
        face_crops: list[np.ndarray],
        audio_path: Optional[Path],
    ) -> list[np.ndarray]:
        if audio_path is None:
            raise ValueError("Wav2LipModel requires an audio_path")
        if not face_crops:
            return []

        wav = load_wav_mono_16k(audio_path)
        mel = melspectrogram(wav)

        out_crops: list[np.ndarray] = [None] * len(face_crops)  # type: ignore[list-item]
        for batch_start in range(0, len(face_crops), INFERENCE_BATCH_SIZE):
            batch_end = min(batch_start + INFERENCE_BATCH_SIZE, len(face_crops))
            face_batch = []
            mel_batch = []
            for i in range(batch_start, batch_end):
                face_batch.append(face_crops[i])
                mel_batch.append(get_mel_chunk_for_frame(mel, i, self.fps))

            face_tensor, mel_tensor = self._build_batch(face_batch, mel_batch)
            face_tensor = face_tensor.to(self.device)
            mel_tensor = mel_tensor.to(self.device)

            with torch.no_grad():
                pred = self.model(mel_tensor, face_tensor)

            pred_np = pred.detach().cpu().numpy()
            for local_i, frame_i in enumerate(range(batch_start, batch_end)):
                out_crops[frame_i] = self._tensor_to_crop(pred_np[local_i])

        return out_crops

    @staticmethod
    def _build_batch(
        face_batch: list[np.ndarray],
        mel_batch: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        faces = np.stack(face_batch, axis=0).astype(np.float32) / 255.0
        masked = faces.copy()
        masked[:, IMG_SIZE // 2:, :, :] = 0.0
        six_channel = np.concatenate([masked, faces], axis=3)
        six_channel = np.transpose(six_channel, (0, 3, 1, 2))
        face_tensor = torch.from_numpy(six_channel).float()

        mels = np.stack(mel_batch, axis=0).astype(np.float32)
        mels = mels[:, np.newaxis, :, :]
        mel_tensor = torch.from_numpy(mels).float()
        return face_tensor, mel_tensor

    @staticmethod
    def _tensor_to_crop(pred_chw: np.ndarray) -> np.ndarray:
        img = np.transpose(pred_chw, (1, 2, 0))
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img
```

- [ ] **Step 4: Run unit test (no weights)**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_wav2lip_model_inference.py::TestConstructor -v
```

Expected: 1 passed.

- [ ] **Step 5: Download the checkpoint**

```bash
mkdir -p ~/Projects/lipsync-corrector/models
# Option A: HuggingFace mirror (most reliable)
curl -fL -o ~/Projects/lipsync-corrector/models/wav2lip_gan.pth \
  https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth
ls -lh ~/Projects/lipsync-corrector/models/wav2lip_gan.pth
```

Expected: the file exists and is around 420 MB. If the HuggingFace URL is dead, search HuggingFace for `wav2lip_gan.pth` or use another known mirror. The checksum of the file should roughly match 420 MB — if you get a 1 KB HTML error page, the URL is stale.

- [ ] **Step 6: Run inference tests (with weights)**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_wav2lip_model_inference.py -v
```

Expected: 4 passed. If the model loads but inference produces garbage, verify that the mel hparams in `core/wav2lip/audio.py` exactly match the Rudrabha reference — those constants are the most common cause of "model runs but output is wrong" failures.

- [ ] **Step 7: Run full suite**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 107 passed (103 existing + 4 new, assuming weights present). If weights are absent, the inference tests skip and you see 104 passed + 3 skipped.

- [ ] **Step 8: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add core/wav2lip_model.py tests/test_wav2lip_model_inference.py
git commit -m "feat: add Wav2LipModel implementing LipSyncModel"
```

---

## Task 6: CLI `--model` flag

**Files:**
- Modify: `~/Projects/lipsync-corrector/cli/main.py`
- Modify: `~/Projects/lipsync-corrector/tests/test_cli.py`

- [ ] **Step 1: Add failing tests**

In `tests/test_cli.py`, append two tests to `TestParseArgs`:

```python
    def test_model_flag_accepts_identity(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4", "--lipsync", "--model", "identity"])
        assert args.model == "identity"

    def test_model_flag_accepts_wav2lip(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4", "--lipsync", "--model", "wav2lip"])
        assert args.model == "wav2lip"

    def test_model_defaults_to_identity(self):
        args = parse_args(["--video", "in.mp4", "--output", "out.mp4", "--lipsync"])
        assert args.model == "identity"
```

Append a new test class to `tests/test_cli.py`:

```python
class TestMainWav2LipModel:
    def test_missing_checkpoint_errors_cleanly(self, tmp_video_with_audio, tmp_path, monkeypatch):
        import core.wav2lip_model as w
        missing = tmp_path / "not_there.pth"
        monkeypatch.setattr(w, "DEFAULT_CHECKPOINT_PATH", missing)
        result = main([
            "--video", str(tmp_video_with_audio),
            "--output", str(tmp_path / "out.mp4"),
            "--lipsync",
            "--model", "wav2lip",
        ])
        assert result == 1
```

- [ ] **Step 2: Run CLI tests to see them fail**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_cli.py -v
```

Expected: 4 new failures (3 parse_args + 1 main flow).

- [ ] **Step 3: Modify `cli/main.py` — add flag**

In `parse_args()`, immediately after the `--lipsync` argument, add:

```python
    parser.add_argument(
        "--model",
        choices=["identity", "wav2lip"],
        default="identity",
        help="Which LipSyncModel to use. 'identity' is the placeholder that passes "
             "crops through unchanged (default). 'wav2lip' runs the real pretrained "
             "Wav2Lip GAN model and requires models/wav2lip_gan.pth to be present.",
    )
```

- [ ] **Step 4: Modify `cli/main.py` — wire model selection**

In `main()`, replace the existing lipsync initialization block:

```python
    lipsync_tracker = None
    lipsync_model = None
    crop_face_region = None
    blend_back = None
    if args.lipsync:
        from core.face_tracker import FaceTracker
        from core.lipsync_model import IdentityModel
        from core.mouth_region import crop_face_region
        from core.blender import blend_back
        print("Loading lip-sync pipeline (placeholder IdentityModel)...")
        lipsync_tracker = FaceTracker()
        lipsync_model = IdentityModel()
```

with:

```python
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
            from core.wav2lip_model import Wav2LipModel
            print("Loading lip-sync pipeline (Wav2LipModel)...")
            with VideoReader(video_path) as _probe:
                _fps = _probe.fps
            try:
                lipsync_model = Wav2LipModel(fps=_fps)
            except FileNotFoundError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
```

Note: the probe `with VideoReader(video_path) as _probe` reads the fps from the same file the main loop will open a moment later. It is fine to open and close the reader twice; the cost is negligible and keeps the model constructor signature simple.

- [ ] **Step 5: Run CLI tests**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest tests/test_cli.py -v
```

Expected: all CLI tests pass (existing 15 + 4 new = 19 total).

- [ ] **Step 6: Run full suite**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: 111 passed (107 existing + 4 new) if weights present, else 108 passed + 3 skipped.

- [ ] **Step 7: Commit**

```bash
cd ~/Projects/lipsync-corrector
git add cli/main.py tests/test_cli.py
git commit -m "feat: add --model flag selecting IdentityModel or Wav2LipModel"
```

---

## Task 7: End-to-end run + milestone notes

**Files:**
- Create: `~/Projects/lipsync-corrector/docs/milestones/milestone-3b.md`
- Modify: `~/Projects/lipsync-corrector/README.md` (add weight download note)

**Background:** For the first real run we need a video and an audio file. The dubbed-audio workflow is the whole point of the project but for this milestone we can use the original audio from the Veo clip itself — the goal is to confirm the Wav2Lip pipeline runs end-to-end and produces output with visible mouth movements matching the audio. Using a truly "dubbed" audio track is deferred until the pipeline is known to work.

- [ ] **Step 1: Extract the original audio as a dubbed-audio placeholder**

```bash
cd ~/Projects/lipsync-corrector
mkdir -p examples
ffmpeg -y -loglevel error \
  -i ~/Downloads/Video_De_Mujer_Saludando_Generado.mp4 \
  -vn -ac 1 -ar 16000 -c:a pcm_s16le \
  examples/veo_audio_16k_mono.wav
ls -lh examples/veo_audio_16k_mono.wav
```

Expected: a mono 16 kHz WAV file a few hundred KB in size.

- [ ] **Step 2: Run the full Wav2Lip pipeline on the Veo clip**

```bash
cd ~/Projects/lipsync-corrector
uv run python -m cli.main \
  --video ~/Downloads/Video_De_Mujer_Saludando_Generado.mp4 \
  --audio examples/veo_audio_16k_mono.wav \
  --output examples/wav2lip_first_run.mp4 \
  --lipsync --model wav2lip
```

Expected output:
- "Loading lip-sync pipeline (Wav2LipModel)..."
- Face tracker loads (prints insightface model names)
- Processes 192 frames
- "Wrote 192 frames to intermediate."
- "Audio preserved."
- "Done. Output: examples/wav2lip_first_run.mp4"

Expected wall time on an M4: roughly a few minutes, dominated by the tracker + torch inference. If runtime is extreme (>15 min) something is wrong — probably the model fell back to CPU. Check the printed device when loading.

- [ ] **Step 3: Inspect the output visually**

```bash
open examples/wav2lip_first_run.mp4
```

Expected visual behavior:
- Mouth region clearly moves in sync with the audio — opening and closing visibly.
- Output is noticeably lower resolution inside the face region (96x96 upscaled + Wav2Lip's native blur) — this is expected and will be addressed in Milestone 4 (blending/upscaling pass).
- Face region borders may show a visible seam — also expected; blending improvements are the next milestone.
- No crashes, no artifacts outside the face region.

Record subjective quality in the milestone notes (step 5).

- [ ] **Step 4: Verify frame count and audio**

```bash
cd ~/Projects/lipsync-corrector
uv run python -c "
from pathlib import Path
from core.video_io import VideoReader, has_audio_stream
out = Path('examples/wav2lip_first_run.mp4')
with VideoReader(out) as r:
    print(f'Output: {r.frame_count} frames, {r.fps:.1f} fps, {r.width}x{r.height}')
print(f'Audio preserved: {has_audio_stream(out)}')
"
```

Expected: 192 frames, 24 fps, 1280x720, audio preserved.

- [ ] **Step 5: Write the milestone notes**

Create `docs/milestones/milestone-3b.md`:

```markdown
# Milestone 3b: Wav2Lip Integration

**Date completed:** <YYYY-MM-DD>
**Track:** B
**Status:** Done

## What was built

- `core/wav2lip/` — vendored Wav2Lip nn.Module (`model.py`, `conv.py`),
  audio preprocessing (`audio.py`), and frame↔mel alignment (`frame_sync.py`).
- `core/wav2lip_model.py` — `Wav2LipModel(LipSyncModel)` that loads the
  pretrained `wav2lip_gan.pth` checkpoint and runs batched inference on MPS.
- `cli/main.py` — `--model {identity,wav2lip}` flag wiring the real model
  into the existing `--lipsync` pipeline built in Milestone 3a.
- <N> new tests passing (audio preprocessing, frame sync math, model
  architecture, constructor errors, inference).

## How to run

1. Download `wav2lip_gan.pth` to `models/wav2lip_gan.pth` (see README).
2. Run:

```bash
uv run python -m cli.main \
  --video examples/input.mp4 \
  --audio examples/dubbed.wav \
  --output examples/out.mp4 \
  --lipsync --model wav2lip
```

## Measured results

End-to-end run on the Veo-generated clip (192 frames, 24 fps, 1280x720)
using the clip's own original audio as the "dubbed" track:

- Frame count preserved: <yes/no>
- Audio preserved: <yes/no>
- Wall time: <measured>
- Device used: <mps/cpu>
- Subjective quality: <fill in after watching>

## What was learned

- <fill in after running>

## Deferred to Milestone 4

- Blending improvements — the Wav2Lip output bbox shows a visible
  resolution seam. Milestone 4 is specifically about blending quality.
- Face alignment via landmarks (currently bbox-only).
- Silence detection to skip frames where audio is silent.

## Next milestone

Milestone 4: blending pass. Reduce visible artifacts at the face bbox
boundary and improve subjective quality.
```

Fill in the `<placeholders>` after running.

- [ ] **Step 6: Add a README note for the weight download**

Open `README.md` and append a new section at the end:

```markdown
## Downloading Wav2Lip weights

The `--model wav2lip` path requires the pretrained checkpoint
`wav2lip_gan.pth` (~420 MB) to exist at `models/wav2lip_gan.pth`. It is
not checked into git. Download it once:

```bash
mkdir -p models
curl -fL -o models/wav2lip_gan.pth \
  https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth
```

If the HuggingFace mirror disappears, search HuggingFace for
`wav2lip_gan.pth` — several community mirrors host the same file.
```

- [ ] **Step 7: Clean up the example output**

```bash
cd ~/Projects/lipsync-corrector
rm -f examples/wav2lip_first_run.mp4
```

Keep `examples/veo_audio_16k_mono.wav` — it is small and useful as a
reproducible audio fixture for future runs.

- [ ] **Step 8: Run the full test suite one last time**

```bash
cd ~/Projects/lipsync-corrector
uv run pytest
```

Expected: all tests pass. With weights present, around 111 passed. Without
weights, around 108 passed + 3 skipped.

- [ ] **Step 9: Commit milestone notes and README**

```bash
cd ~/Projects/lipsync-corrector
git add docs/milestones/milestone-3b.md README.md
git commit -m "feat: milestone-3b complete — wav2lip integration"
```

---

## Done criteria for Milestone 3b

- `uv run pytest` passes all tests (both with and without the weights file present).
- `uv run python -m cli.main --video <real-video> --audio <real-audio> --output <path> --lipsync --model wav2lip` produces a video with visible mouth movement matching the audio, preserved frame count, preserved audio, and no crashes.
- `core/wav2lip/` subpackage exists with model, audio, and frame_sync files.
- `core/wav2lip_model.py` exists and implements `LipSyncModel`.
- `cli/main.py` has a `--model` flag defaulting to `identity`.
- `docs/milestones/milestone-3b.md` written with actual measurements.
- README documents how to download the checkpoint.
- Everything committed on `milestone-3b` branch, ready to merge to `main`.
- Existing behavior (passthrough, `--debug-tracking`, `--lipsync --model identity`) untouched.

Milestone 4 (blending quality improvements) is out of scope. Do not start it in the same session.
