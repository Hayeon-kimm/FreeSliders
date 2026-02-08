"""
ASTD: Automatic Saturation and Traversal Detection

A two-stage procedure for automatically improving slider quality:
1. Saturation Detection - Estimate where a concept saturates
2. Traversal Adjustment - Reparameterize for perceptually uniform changes
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings


@dataclass
class ASTDResult:
    """Container for ASTD results."""
    saturation_point_pos: float
    saturation_point_neg: float
    optimized_scales: List[float]
    alignment_scores: Dict[float, float]
    perceptual_scores: Dict[float, float]
    reparameterization_func: Optional[Callable]


class ASTD:
    """
    Automatic Saturation and Traversal Detection.

    Solves two key problems:
    1. Determining where a concept saturates (e.g., maximum age)
    2. Reparameterizing traversal for perceptually uniform changes
    """

    def __init__(
        self,
        alignment_func: Callable,
        perceptual_func: Callable,
        trade_off_ratio: float = 1.0,
        candidate_scales: Optional[List[float]] = None,
    ):
        """
        Initialize ASTD.

        Args:
            alignment_func: Function to compute alignment score a(x, c)
            perceptual_func: Function to compute perceptual distance d(x0, x)
            trade_off_ratio: r value for balancing preservation vs intensity
            candidate_scales: Scales to evaluate for saturation detection
        """
        self.alignment_func = alignment_func
        self.perceptual_func = perceptual_func
        self.trade_off_ratio = trade_off_ratio
        self.candidate_scales = candidate_scales or [0, 0.5, 1, 2, 4, 8, 16]

    def detect_saturation(
        self,
        samples: Dict[float, torch.Tensor],
        reference_sample: torch.Tensor,
        concept_prompt: str,
    ) -> float:
        """
        Step 1: Saturation Detection.

        Find the largest scale η where r(x, η) = a(x_η, c) / d(x_0, x_η) >= r_threshold

        Args:
            samples: Dictionary mapping scale -> sample
            reference_sample: Reference sample (x_0)
            concept_prompt: Concept prompt to measure alignment against

        Returns:
            Detected saturation point
        """
        scales = sorted(samples.keys())
        valid_scales = []

        for scale in scales:
            if scale == 0:
                continue

            sample = samples[scale]

            # Compute alignment score
            alignment = self.alignment_func(sample, concept_prompt)

            # Compute perceptual distance
            perceptual = self.perceptual_func(reference_sample, sample)

            # Avoid division by zero
            if perceptual < 1e-6:
                perceptual = 1e-6

            # Compute ratio
            ratio = alignment / perceptual

            if ratio >= self.trade_off_ratio:
                valid_scales.append(scale)

        if not valid_scales:
            # Return the largest scale if no valid ones found
            return max(scales)

        return max(valid_scales)

    def fit_reparameterization(
        self,
        scales: List[float],
        alignment_scores: List[float],
    ) -> Callable:
        """
        Step 2: Traversal Adjustment.

        Fit a monotone reparameterization that maps alignment axis to scale axis
        for perceptually uniform changes.

        Args:
            scales: Scale values
            alignment_scores: Corresponding alignment scores

        Returns:
            Reparameterization function
        """
        # Normalize alignment scores to [0, 1]
        a_min, a_max = min(alignment_scores), max(alignment_scores)
        if a_max - a_min < 1e-6:
            # No variation - return identity mapping
            return lambda x: x

        normalized = [(a - a_min) / (a_max - a_min) for a in alignment_scores]

        # Create interpolation function (alignment -> scale)
        try:
            # Sort by normalized alignment for monotone interpolation
            sorted_pairs = sorted(zip(normalized, scales))
            sorted_norm, sorted_scales = zip(*sorted_pairs)

            interp_func = interp1d(
                sorted_norm,
                sorted_scales,
                kind='linear',
                bounds_error=False,
                fill_value=(sorted_scales[0], sorted_scales[-1])
            )

            return interp_func

        except Exception as e:
            warnings.warn(f"Failed to fit reparameterization: {e}")
            return lambda x: x * max(scales)

    def get_uniform_scales(
        self,
        reparameterization_func: Callable,
        num_steps: int = 7,
    ) -> List[float]:
        """
        Get uniformly distributed scales using the reparameterization.

        Args:
            reparameterization_func: Mapping from [0,1] to scales
            num_steps: Number of scale steps to generate

        Returns:
            List of optimized scale values
        """
        uniform_t = np.linspace(0, 1, num_steps)
        optimized_scales = []
        for t in uniform_t:
            result = reparameterization_func(t)
            # Handle numpy arrays and scalars
            if isinstance(result, np.ndarray):
                result = result.item() if result.ndim == 0 else float(result.flat[0])
            optimized_scales.append(float(result))

        return optimized_scales

    def run(
        self,
        generator_func: Callable,
        reference_sample: torch.Tensor,
        positive_prompt: str,
        negative_prompt: str,
        num_output_scales: int = 7,
    ) -> ASTDResult:
        """
        Run the full ASTD procedure.

        Args:
            generator_func: Function that takes scales and returns samples dict
            reference_sample: Reference sample (x_0)
            positive_prompt: Positive concept prompt
            negative_prompt: Negative concept prompt
            num_output_scales: Number of optimized scales to output

        Returns:
            ASTDResult with all detection results
        """
        # Generate samples at candidate scales
        pos_scales = [s for s in self.candidate_scales if s >= 0]
        neg_scales = [-s for s in self.candidate_scales if s > 0]

        all_scales = sorted(set(neg_scales + pos_scales))
        samples = generator_func(all_scales)

        # Detect saturation for positive direction
        pos_samples = {s: samples[s] for s in pos_scales if s in samples}
        saturation_pos = self.detect_saturation(
            pos_samples, reference_sample, positive_prompt
        )

        # Detect saturation for negative direction
        neg_samples = {s: samples[s] for s in neg_scales if s in samples}
        saturation_neg = self.detect_saturation(
            neg_samples, reference_sample, negative_prompt
        )

        # Compute alignment and perceptual scores
        alignment_scores = {}
        perceptual_scores = {}

        for scale, sample in samples.items():
            if scale >= 0:
                prompt = positive_prompt
            else:
                prompt = negative_prompt

            alignment_scores[scale] = self.alignment_func(sample, prompt)
            perceptual_scores[scale] = self.perceptual_func(reference_sample, sample)

        # Fit reparameterization for positive direction
        pos_scale_list = sorted([s for s in pos_scales if s <= saturation_pos])
        pos_align_list = [alignment_scores[s] for s in pos_scale_list]

        if len(pos_scale_list) > 1:
            reparam_pos = self.fit_reparameterization(pos_scale_list, pos_align_list)
        else:
            reparam_pos = lambda x: x * saturation_pos

        # Fit reparameterization for negative direction
        neg_scale_list = sorted([s for s in neg_scales if s >= -abs(saturation_neg)], reverse=True)
        neg_align_list = [alignment_scores[s] for s in neg_scale_list]

        if len(neg_scale_list) > 1:
            reparam_neg = self.fit_reparameterization(neg_scale_list, neg_align_list)
        else:
            reparam_neg = lambda x: -x * abs(saturation_neg)

        # Generate optimized scales
        num_pos = (num_output_scales + 1) // 2
        num_neg = num_output_scales - num_pos

        pos_optimized = self.get_uniform_scales(reparam_pos, num_pos)
        neg_optimized = self.get_uniform_scales(reparam_neg, num_neg)
        neg_optimized = [-s for s in neg_optimized if s != 0]

        optimized_scales = sorted(set(neg_optimized + pos_optimized))

        # Combined reparameterization function
        def combined_reparam(t):
            if t >= 0:
                return reparam_pos(t)
            else:
                return reparam_neg(-t)

        return ASTDResult(
            saturation_point_pos=saturation_pos,
            saturation_point_neg=saturation_neg,
            optimized_scales=optimized_scales,
            alignment_scores=alignment_scores,
            perceptual_scores=perceptual_scores,
            reparameterization_func=combined_reparam,
        )


def apply_astd(
    freeslider,
    concept,
    alignment_func: Callable,
    perceptual_func: Callable,
    num_samples: int = 30,
    num_scales: int = 13,
    trade_off_ratio: float = 1.0,
    **generation_kwargs,
) -> Tuple[ASTDResult, Dict[float, torch.Tensor]]:
    """
    Convenience function to apply ASTD to a FreeSlider.

    Args:
        freeslider: FreeSlider instance
        concept: ConceptPrompts for the concept
        alignment_func: Alignment scoring function
        perceptual_func: Perceptual distance function
        num_samples: Number of samples to generate per scale
        num_scales: Number of candidate scales to test
        trade_off_ratio: r value for saturation detection
        **generation_kwargs: Additional arguments for generation

    Returns:
        Tuple of (ASTDResult, optimized_samples)
    """
    # Define candidate scales
    candidate_scales = [0, 0.5, 1, 2, 4, 8, 16]

    # Create ASTD instance
    astd = ASTD(
        alignment_func=alignment_func,
        perceptual_func=perceptual_func,
        trade_off_ratio=trade_off_ratio,
        candidate_scales=candidate_scales,
    )

    # Generator function for ASTD
    def generator_func(scales):
        return freeslider.generate(
            concept=concept,
            scales=scales,
            **generation_kwargs,
        )

    # Generate reference sample (scale=0)
    ref_samples = freeslider.generate(
        concept=concept,
        scales=[0],
        **generation_kwargs,
    )
    reference_sample = ref_samples[0]

    # Run ASTD
    result = astd.run(
        generator_func=generator_func,
        reference_sample=reference_sample,
        positive_prompt=concept.positive,
        negative_prompt=concept.negative,
        num_output_scales=7,
    )

    # Generate final samples with optimized scales
    optimized_samples = freeslider.generate(
        concept=concept,
        scales=result.optimized_scales,
        **generation_kwargs,
    )

    return result, optimized_samples


def visualize_astd_result(result: ASTDResult, save_path: Optional[str] = None):
    """
    Visualize ASTD results.

    Args:
        result: ASTDResult from ASTD.run()
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed. Cannot visualize.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Alignment vs Perceptual (Saturation Detection)
    ax1 = axes[0]
    scales = sorted(result.alignment_scores.keys())

    perceptual_vals = [result.perceptual_scores[s] for s in scales]
    alignment_vals = [result.alignment_scores[s] for s in scales]

    ax1.scatter(perceptual_vals, alignment_vals, c=scales, cmap='coolwarm')
    for i, s in enumerate(scales):
        ax1.annotate(f'{s}', (perceptual_vals[i], alignment_vals[i]))

    # Plot reference line (r=1)
    max_p = max(perceptual_vals)
    ax1.plot([0, max_p], [0, max_p], 'k--', label='r=1')

    ax1.set_xlabel('Perceptual Score')
    ax1.set_ylabel('Alignment Score')
    ax1.set_title('Step 1: Saturation Detection')
    ax1.legend()

    # Plot 2: Alignment vs Scale (Traversal Adjustment)
    ax2 = axes[1]

    ax2.scatter(scales, alignment_vals)
    ax2.plot(scales, alignment_vals, 'b-', alpha=0.5)

    # Mark saturation points
    ax2.axvline(x=result.saturation_point_pos, color='g', linestyle='--',
                label=f'Sat+ = {result.saturation_point_pos}')
    ax2.axvline(x=-result.saturation_point_neg, color='r', linestyle='--',
                label=f'Sat- = {-result.saturation_point_neg}')

    # Mark optimized scales
    for s in result.optimized_scales:
        ax2.axvline(x=s, color='gray', alpha=0.3, linestyle=':')

    ax2.set_xlabel('Scale (η)')
    ax2.set_ylabel('Alignment Score')
    ax2.set_title('Step 2: Traversal Adjustment')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()
