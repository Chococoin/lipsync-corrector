"""Wav2Lip third-party vendor subpackage.

Contains the reference nn.Module architecture and audio preprocessing
adapted from https://github.com/Rudrabha/Wav2Lip. These files are vendored
so that weight loading against the published wav2lip_gan.pth checkpoint is
reproducible and not dependent on an external repo remaining online.
"""

from core.wav2lip.model import Wav2Lip

__all__ = ["Wav2Lip"]
