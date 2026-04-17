"""TTS synthesis using Coqui XTTS-v2 with voice cloning."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from core.device import get_torch_device
from core.transcription.models import Transcription
from core.tts.assembler import assemble_track
from core.tts.reference import extract_reference_audio, select_reference_segments

XTTS_SAMPLE_RATE = 24000
SOURCE_SAMPLE_RATE = 16000
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def synthesize(
    transcription: Transcription,
    source_audio_path: Path | str,
    output_path: Path | str,
    *,
    model_name: str = DEFAULT_MODEL,
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
    source_audio_path = Path(source_audio_path)
    output_path = Path(output_path)

    if not source_audio_path.exists():
        raise FileNotFoundError(f"source audio not found: {source_audio_path}")

    if not transcription.segments:
        silence = np.zeros(
            int(transcription.duration * XTTS_SAMPLE_RATE), dtype=np.float32
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), silence, XTTS_SAMPLE_RATE)
        return output_path

    source_wav, source_sr = sf.read(str(source_audio_path), dtype="float32")
    if source_wav.ndim == 2:
        source_wav = source_wav.mean(axis=1)

    ref_segments = select_reference_segments(transcription)
    ref_audio = extract_reference_audio(source_wav, ref_segments, sample_rate=source_sr)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        ref_wav_path = Path(tmp.name)
        sf.write(str(ref_wav_path), ref_audio, source_sr)

    try:
        segment_audios = _generate_segments(
            transcription, ref_wav_path, model_name
        )
    finally:
        ref_wav_path.unlink(missing_ok=True)

    track = assemble_track(
        segment_audios,
        transcription.segments,
        transcription.duration,
        sample_rate=XTTS_SAMPLE_RATE,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), track, XTTS_SAMPLE_RATE)
    return output_path


def _generate_segments(
    transcription: Transcription,
    ref_wav_path: Path,
    model_name: str,
) -> list[np.ndarray]:
    """Generate audio for each segment using XTTS-v2."""
    from TTS.api import TTS

    device = get_torch_device()
    tts = TTS(model_name).to(device)

    audios: list[np.ndarray] = []
    for seg in transcription.segments:
        wav = tts.tts(
            text=seg.text,
            speaker_wav=str(ref_wav_path),
            language=transcription.language,
        )
        audios.append(np.array(wav, dtype=np.float32))

    return audios
