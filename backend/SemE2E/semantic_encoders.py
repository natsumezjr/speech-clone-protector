"""Legacy import compatibility for the canonical :mod:`core.encoders` module."""

from core.encoders import (
    DifferentiableOpenAIWhisperMel,
    DifferentiableWhisperMel,
    SemanticEncoderEnsemble,
)

__all__ = [
    "DifferentiableOpenAIWhisperMel",
    "DifferentiableWhisperMel",
    "SemanticEncoderEnsemble",
]
