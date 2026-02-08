"""
FreeSliders Pipelines for different modalities.
"""

from .stable_diffusion import FreeSliderStableDiffusionPipeline
from .video import FreeSliderVideoPipeline
from .audio import FreeSliderAudioPipeline

__all__ = [
    "FreeSliderStableDiffusionPipeline",
    "FreeSliderVideoPipeline",
    "FreeSliderAudioPipeline",
]
