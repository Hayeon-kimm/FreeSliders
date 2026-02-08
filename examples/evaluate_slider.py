"""
Example: Evaluating Concept Sliders with FreeSliders Metrics

This example demonstrates how to evaluate slider quality using
the proposed metrics: Conceptual Range (CR), Conceptual Smoothness (CSM),
and Semantic Preservation (SP).
"""

import torch
import sys
sys.path.append('..')

from freesliders import ConceptPrompts
from pipelines.stable_diffusion import FreeSliderStableDiffusionPipeline
from metrics import SliderEvaluator, CLIPAlignment, LPIPSDistance, compute_delta_clip
from astd import ASTD, visualize_astd_result


def main():
    # Initialize pipeline
    print("Loading model...")
    pipeline = FreeSliderStableDiffusionPipeline(
        model_id="CompVis/stable-diffusion-v1-4",
        device="cuda",
        dtype=torch.float16,
    )

    # Define concept
    concept = ConceptPrompts(
        base="A realistic image of a person.",
        positive="A realistic image of a person, smiling widely, very happy.",
        negative="A realistic image of a person, frowning, very sad.",
    )

    # Standard scales
    scales = [-3, -2, -1, 0, 1, 2, 3]

    # Generate samples
    print("\nGenerating samples...")
    results = pipeline.generate(
        concept=concept,
        scales=scales,
        num_inference_steps=50,
        intervention_step=15,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        output_type="pil",  # Need PIL for metrics
    )

    # Initialize evaluator
    print("\nInitializing evaluator...")
    alignment_model = CLIPAlignment()
    perceptual_distance = LPIPSDistance()

    evaluator = SliderEvaluator(
        alignment_model=alignment_model,
        perceptual_distance=perceptual_distance,
        device="cuda",
    )

    # Convert PIL images to tensors for perceptual distance
    # (CLIP can work with PIL directly)
    import torchvision.transforms as transforms
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # [0,1] -> [-1,1]
    ])

    results_tensor = {}
    for scale, img in results.items():
        results_tensor[scale] = to_tensor(img).unsqueeze(0).cuda()

    # Evaluate slider
    print("\nEvaluating slider quality...")

    # Compute each metric
    cr = evaluator.compute_conceptual_range(
        results,  # PIL images for CLIP
        concept.positive,
        concept.negative,
    )
    print(f"Conceptual Range (CR): {cr:.4f}")

    csm = evaluator.compute_conceptual_smoothness(
        results,
        concept.positive,
        concept.negative,
    )
    print(f"Conceptual Smoothness (CSM): {csm:.4f}")

    sp = evaluator.compute_semantic_preservation(results_tensor)
    print(f"Semantic Preservation (SP): {sp:.4f}")

    os = evaluator.compute_overall_score(cr, csm, sp)
    print(f"Overall Score (OS): {os:.4f}")

    # Compare with ∆CLIP
    delta_clip = compute_delta_clip(
        results,
        concept.positive,
        alignment_model,
        device="cuda",
    )
    print(f"\nFor comparison - ∆CLIP: {delta_clip:.4f}")

    # Full evaluation
    print("\n" + "="*50)
    print("Full Evaluation Results:")
    print("="*50)
    metrics = evaluator.evaluate(
        results,
        concept.positive,
        concept.negative,
    )
    print(metrics)

    print("\nDone!")


def run_astd_example():
    """Example of using ASTD for automatic scale optimization."""
    print("\n" + "="*50)
    print("ASTD Example: Automatic Saturation and Traversal Detection")
    print("="*50)

    # Initialize pipeline
    pipeline = FreeSliderStableDiffusionPipeline(
        model_id="CompVis/stable-diffusion-v1-4",
        device="cuda",
        dtype=torch.float16,
    )

    concept = ConceptPrompts(
        base="A realistic image of a person.",
        positive="A realistic image of a person, very old, aged, wrinkly.",
        negative="A realistic image of a person, detailed facial features, clear skin.",
    )

    # Initialize CLIP for alignment
    alignment_model = CLIPAlignment()
    alignment_model.load()

    # Initialize LPIPS for perceptual distance
    perceptual_model = LPIPSDistance()
    perceptual_model.load()

    def alignment_func(sample, prompt):
        return alignment_model.compute_alignment(sample, prompt, device="cuda")

    def perceptual_func(sample1, sample2):
        return perceptual_model.compute_distance(sample1, sample2, device="cuda")

    # Initialize ASTD
    astd = ASTD(
        alignment_func=alignment_func,
        perceptual_func=perceptual_func,
        trade_off_ratio=1.0,
        candidate_scales=[0, 0.5, 1, 2, 4, 8, 16],
    )

    # Generator function
    def generator_func(scales):
        return pipeline.generate(
            concept=concept,
            scales=scales,
            num_inference_steps=50,
            intervention_step=15,
            guidance_scale=7.5,
            seed=42,
            output_type="pil",
        )

    # Generate reference
    ref_results = generator_func([0])
    reference_sample = ref_results[0]

    # Run ASTD
    print("\nRunning ASTD...")
    result = astd.run(
        generator_func=generator_func,
        reference_sample=reference_sample,
        positive_prompt=concept.positive,
        negative_prompt=concept.negative,
        num_output_scales=7,
    )

    print(f"\nSaturation Point (positive): {result.saturation_point_pos}")
    print(f"Saturation Point (negative): {result.saturation_point_neg}")
    print(f"Optimized Scales: {result.optimized_scales}")

    # Visualize
    visualize_astd_result(result, save_path="astd_visualization.png")

    # Generate with optimized scales
    print("\nGenerating with optimized scales...")
    optimized_results = generator_func(result.optimized_scales)

    from pipelines.stable_diffusion import create_image_slider_grid
    grid = create_image_slider_grid(optimized_results)
    grid.save("slider_astd_optimized.png")
    print("Saved optimized slider to slider_astd_optimized.png")


if __name__ == "__main__":
    main()

    # Uncomment to run ASTD example
    # run_astd_example()
