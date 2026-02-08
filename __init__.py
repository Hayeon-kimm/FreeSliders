"""
FreeSliders: Training-Free, Modality-Agnostic Concept Sliders
for Fine-Grained Diffusion Control in Images, Audio, and Video

Based on: https://arxiv.org/abs/2511.00103

Authors: Rotem Ezra, Hedi Zisling, Nimrod Berman, Ilan Naiman,
         Alexey Gorkor, Liran Nochumsohn, Eliya Nachmani, Omri Azencot
"""

__version__ = "0.1.0"
__author__ = "FreeSliders Team"

from .freesliders import FreeSliders, ConceptPrompts, TextEmbeddingSlider
from .metrics import (
    SliderMetrics,
    SliderEvaluator,
    CLIPAlignment,
    LPIPSDistance,
    compute_delta_clip,
)
from .astd import ASTD, ASTDResult, apply_astd, visualize_astd_result

__all__ = [
    # Core
    "FreeSliders",
    "ConceptPrompts",
    "TextEmbeddingSlider",
    # Metrics
    "SliderMetrics",
    "SliderEvaluator",
    "CLIPAlignment",
    "LPIPSDistance",
    "compute_delta_clip",
    # ASTD
    "ASTD",
    "ASTDResult",
    "apply_astd",
    "visualize_astd_result",
]
