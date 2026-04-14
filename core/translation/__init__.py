"""Translation subpackage.

Exposes a Python API: translate a Transcription into another language
via the Claude API, preserving segment-level timestamps.
"""

from core.translation.translator import translate

__all__ = ["translate"]
