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
