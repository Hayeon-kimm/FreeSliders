"""
Evaluation Metrics for Concept Sliders

Three key properties of an effective slider:
1. Range (CR) - extent of concept variation
2. Smoothness (CSM) - consistency of intermediate transitions
3. Preservation (SP) - maintenance of non-target content
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Callable, Union
from dataclasses import dataclass
import warnings


@dataclass
class SliderMetrics:
    """Container for slider evaluation metrics."""
    conceptual_range: float
    conceptual_smoothness: float
    semantic_preservation: float
    overall_score: float

    def __repr__(self):
        return (
            f"SliderMetrics(\n"
            f"  CR={self.conceptual_range:.4f},\n"
            f"  CSM={self.conceptual_smoothness:.4f},\n"
            f"  SP={self.semantic_preservation:.4f},\n"
            f"  OS={self.overall_score:.4f}\n"
            f")"
        )


class AlignmentModel:
    """Base class for alignment models (CLIP, ViCLIP, CLAP)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.model = None
        self.processor = None

    def load(self):
        """Load the model."""
        raise NotImplementedError

    def compute_alignment(self, sample, text: str) -> float:
        """Compute alignment score between sample and text."""
        raise NotImplementedError


class CLIPAlignment(AlignmentModel):
    """CLIP-based alignment for images."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__(model_name)
        self._loaded = False

    def load(self):
        """Load CLIP model."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            self._loaded = True
        except ImportError:
            warnings.warn("transformers not installed. CLIP alignment unavailable.")

    @torch.no_grad()
    def compute_alignment(
        self,
        image,
        text: str,
        device: str = "cuda"
    ) -> float:
        """
        Compute CLIP alignment score.

        Args:
            image: PIL Image or tensor
            text: Text prompt
            device: Device to use

        Returns:
            Cosine similarity score
        """
        if not self._loaded:
            self.load()

        self.model = self.model.to(device)

        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        outputs = self.model(**inputs)

        # Normalize and compute cosine similarity
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

        similarity = (image_embeds @ text_embeds.T).item()

        return similarity


class PerceptualDistance:
    """Base class for perceptual distance metrics."""

    def compute_distance(self, sample1, sample2) -> float:
        """Compute perceptual distance between two samples."""
        raise NotImplementedError


class LPIPSDistance(PerceptualDistance):
    """LPIPS perceptual distance for images."""

    def __init__(self, net: str = "alex"):
        self.net = net
        self.model = None
        self._loaded = False

    def load(self):
        """Load LPIPS model."""
        try:
            import lpips
            self.model = lpips.LPIPS(net=self.net)
            self.model.eval()
            self._loaded = True
        except ImportError:
            warnings.warn("lpips not installed. LPIPS distance unavailable.")

    @torch.no_grad()
    def compute_distance(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        device: str = "cuda"
    ) -> float:
        """
        Compute LPIPS distance.

        Args:
            image1: First image tensor [-1, 1]
            image2: Second image tensor [-1, 1]
            device: Device to use

        Returns:
            LPIPS distance (lower = more similar)
        """
        if not self._loaded:
            self.load()

        self.model = self.model.to(device)
        image1 = image1.to(device)
        image2 = image2.to(device)

        distance = self.model(image1, image2)

        return distance.item()


class SliderEvaluator:
    """
    Evaluator for concept sliders.

    Computes three main metrics:
    - Conceptual Range (CR): Measures the extent of semantic change
    - Conceptual Smoothness (CSM): Measures uniformity of transitions
    - Semantic Preservation (SP): Measures content preservation
    """

    def __init__(
        self,
        alignment_model: Optional[AlignmentModel] = None,
        perceptual_distance: Optional[PerceptualDistance] = None,
        device: str = "cuda",
    ):
        """
        Initialize evaluator.

        Args:
            alignment_model: Model for computing text-sample alignment
            perceptual_distance: Model for computing perceptual distance
            device: Device to use
        """
        self.alignment_model = alignment_model or CLIPAlignment()
        self.perceptual_distance = perceptual_distance or LPIPSDistance()
        self.device = device

    def compute_conceptual_range(
        self,
        samples: Dict[float, torch.Tensor],
        positive_prompt: str,
        negative_prompt: str,
    ) -> float:
        """
        Compute Conceptual Range (CR).

        CR = 0.5 * (CR_pos + CR_neg)
        where:
            CR_pos = a(x_η_max, c+) - a(x_η_min, c+)
            CR_neg = a(x_η_min, c-) - a(x_η_max, c-)

        Args:
            samples: Dictionary mapping scale -> sample
            positive_prompt: Positive concept prompt
            negative_prompt: Negative concept prompt

        Returns:
            Conceptual range score (higher = better)
        """
        scales = sorted(samples.keys())
        eta_min, eta_max = min(scales), max(scales)

        sample_min = samples[eta_min]
        sample_max = samples[eta_max]

        # Compute alignment scores
        a_max_pos = self.alignment_model.compute_alignment(
            sample_max, positive_prompt, self.device
        )
        a_min_pos = self.alignment_model.compute_alignment(
            sample_min, positive_prompt, self.device
        )
        a_max_neg = self.alignment_model.compute_alignment(
            sample_max, negative_prompt, self.device
        )
        a_min_neg = self.alignment_model.compute_alignment(
            sample_min, negative_prompt, self.device
        )

        cr_pos = a_max_pos - a_min_pos
        cr_neg = a_min_neg - a_max_neg

        cr = 0.5 * (cr_pos + cr_neg)

        return cr

    def compute_conceptual_smoothness(
        self,
        samples: Dict[float, torch.Tensor],
        positive_prompt: str,
        negative_prompt: str,
    ) -> float:
        """
        Compute Conceptual Smoothness (CSM).

        CSM = std({g_i}) where g_i = A_{i+1} - A_i are consecutive gaps
        in the normalized alignment scores.

        Args:
            samples: Dictionary mapping scale -> sample
            positive_prompt: Positive concept prompt
            negative_prompt: Negative concept prompt

        Returns:
            Conceptual smoothness score (lower = smoother)
        """
        scales = sorted(samples.keys())

        # Separate positive and negative scales
        pos_scales = [s for s in scales if s >= 0]
        neg_scales = [s for s in scales if s < 0]

        all_gaps = []

        # Process positive scales
        if len(pos_scales) > 1:
            pos_alignments = []
            for s in pos_scales:
                a = self.alignment_model.compute_alignment(
                    samples[s], positive_prompt, self.device
                )
                pos_alignments.append(a)

            # Normalize to [0, 1]
            if max(pos_alignments) > min(pos_alignments):
                pos_alignments = [
                    (a - min(pos_alignments)) / (max(pos_alignments) - min(pos_alignments))
                    for a in pos_alignments
                ]

                # Compute gaps
                for i in range(len(pos_alignments) - 1):
                    all_gaps.append(pos_alignments[i + 1] - pos_alignments[i])

        # Process negative scales
        if len(neg_scales) > 1:
            neg_scales = sorted(neg_scales, reverse=True)  # Order from 0 to most negative
            neg_alignments = []
            for s in neg_scales:
                a = self.alignment_model.compute_alignment(
                    samples[s], negative_prompt, self.device
                )
                neg_alignments.append(a)

            # Normalize to [0, 1]
            if max(neg_alignments) > min(neg_alignments):
                neg_alignments = [
                    (a - min(neg_alignments)) / (max(neg_alignments) - min(neg_alignments))
                    for a in neg_alignments
                ]

                # Compute gaps
                for i in range(len(neg_alignments) - 1):
                    all_gaps.append(neg_alignments[i + 1] - neg_alignments[i])

        if not all_gaps:
            return 0.0

        csm = np.std(all_gaps)

        return csm

    def compute_semantic_preservation(
        self,
        samples: Dict[float, torch.Tensor],
    ) -> float:
        """
        Compute Semantic Preservation (SP).

        SP = (1/|G|) * Σ d(x_η, x_0) for η in G (excluding 0)

        Args:
            samples: Dictionary mapping scale -> sample

        Returns:
            Preservation score (lower = better preservation)
        """
        if 0 not in samples:
            # Use closest to zero as reference
            scales = sorted(samples.keys(), key=abs)
            reference_scale = scales[0]
        else:
            reference_scale = 0

        reference_sample = samples[reference_scale]

        distances = []
        for scale, sample in samples.items():
            if scale == reference_scale:
                continue
            d = self.perceptual_distance.compute_distance(
                sample, reference_sample, self.device
            )
            distances.append(d)

        if not distances:
            return 0.0

        sp = np.mean(distances)

        return sp

    def compute_overall_score(
        self,
        cr: float,
        csm: float,
        sp: float,
        epsilon: float = 1.0,
    ) -> float:
        """
        Compute Overall Score (OS).

        OS = CR / (ε + SP) + (1 - CSM)

        Args:
            cr: Conceptual Range
            csm: Conceptual Smoothness
            sp: Semantic Preservation
            epsilon: Stabilization constant

        Returns:
            Overall score (higher = better)
        """
        os = cr / (epsilon + sp) + (1 - csm)
        return os

    def evaluate(
        self,
        samples: Dict[float, torch.Tensor],
        positive_prompt: str,
        negative_prompt: str,
    ) -> SliderMetrics:
        """
        Evaluate a slider across all metrics.

        Args:
            samples: Dictionary mapping scale -> sample
            positive_prompt: Positive concept prompt
            negative_prompt: Negative concept prompt

        Returns:
            SliderMetrics object with all scores
        """
        cr = self.compute_conceptual_range(samples, positive_prompt, negative_prompt)
        csm = self.compute_conceptual_smoothness(samples, positive_prompt, negative_prompt)
        sp = self.compute_semantic_preservation(samples)
        os = self.compute_overall_score(cr, csm, sp)

        return SliderMetrics(
            conceptual_range=cr,
            conceptual_smoothness=csm,
            semantic_preservation=sp,
            overall_score=os,
        )


def compute_delta_clip(
    samples: Dict[float, torch.Tensor],
    positive_prompt: str,
    alignment_model: AlignmentModel,
    device: str = "cuda",
) -> float:
    """
    Compute ∆CLIP metric (for comparison with prior work).

    ∆CLIP(μ) = |a(x_μ, c+) - a(x_0, c+)|
    ∆CLIP_avg = (1/|S|) * Σ ∆CLIP(μ)

    Note: This metric has limitations as described in the paper.

    Args:
        samples: Dictionary mapping scale -> sample
        positive_prompt: Positive concept prompt
        alignment_model: Alignment model to use
        device: Device to use

    Returns:
        Average ∆CLIP score
    """
    if 0 not in samples:
        scales = sorted(samples.keys(), key=abs)
        reference_scale = scales[0]
    else:
        reference_scale = 0

    a_ref = alignment_model.compute_alignment(
        samples[reference_scale], positive_prompt, device
    )

    delta_clips = []
    for scale, sample in samples.items():
        if scale == reference_scale:
            continue
        a_scale = alignment_model.compute_alignment(sample, positive_prompt, device)
        delta_clips.append(abs(a_scale - a_ref))

    if not delta_clips:
        return 0.0

    return np.mean(delta_clips)
