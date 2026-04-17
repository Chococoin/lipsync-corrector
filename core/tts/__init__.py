"""Text-to-speech subpackage.

Generates dubbed audio from a translated Transcription using Coqui XTTS-v2
with voice cloning from the original speaker.
"""

from core.tts.synthesizer import synthesize

__all__ = ["synthesize"]
