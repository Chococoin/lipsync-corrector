"""Speech-to-text transcription subpackage.

Exposes a Python API: transcribe an audio file with Whisper (via MLX),
receive an immutable Transcription, and optionally serialize to JSON or SRT.
"""

from core.transcription.models import Segment, Transcription, Word
from core.transcription.serializers import write_json, write_srt

__all__ = ["Segment", "Transcription", "Word", "write_json", "write_srt"]
