"""
Example: Image Concept Slider with FreeSliders

This example demonstrates how to use FreeSliders for fine-grained
control over image generation using Stable Diffusion.
"""

import torch
import sys
sys.path.append('..')

from freesliders import ConceptPrompts
from pipelines.stable_diffusion import FreeSliderStableDiffusionPipeline, create_image_slider_grid


def main():
    # Initialize pipeline
    print("Loading Stable Diffusion model...")
    pipeline = FreeSliderStableDiffusionPipeline(
        model_id="CompVis/stable-diffusion-v1-4",
        device="cuda",
        dtype=torch.float16,
    )

    # Define concept: Smiling
    concept_smile = ConceptPrompts(
        base="A realistic image of a person.",
        positive="A realistic image of a person, smiling widely, very happy.",
        negative="A realistic image of a person, frowning, very sad.",
    )

    # Define concept: Age
    concept_age = ConceptPrompts(
        base="A realistic image of a person.",
        positive="A realistic image of a person, very old, aged, wrinkly.",
        negative="A realistic image of a person, detailed facial features, clear skin.",
    )

    # Define slider scales
    scales = [-3, -2, -1, 0, 1, 2, 3]

    # Generate with smiling concept
    print("\nGenerating images with 'smiling' concept slider...")
    results_smile = pipeline.generate(
        concept=concept_smile,
        scales=scales,
        num_inference_steps=50,
        intervention_step=15,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        output_type="pil",
    )

    # Create grid visualization
    grid_smile = create_image_slider_grid(results_smile)
    grid_smile.save("slider_smile.png")
    print("Saved smiling slider to slider_smile.png")

    # Generate with age concept
    print("\nGenerating images with 'age' concept slider...")
    results_age = pipeline.generate(
        concept=concept_age,
        scales=scales,
        num_inference_steps=50,
        intervention_step=15,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        output_type="pil",
    )

    grid_age = create_image_slider_grid(results_age)
    grid_age.save("slider_age.png")
    print("Saved age slider to slider_age.png")

    # Composition example: Age + Smile
    print("\nGenerating composed slider (age + smile)...")
    composed_image = pipeline.generate_composition(
        concepts=[
            (concept_age, 2.0),   # Increase age
            (concept_smile, 1.5), # Add smile
        ],
        num_inference_steps=50,
        intervention_step=15,
        guidance_scale=7.5,
        height=512,
        width=512,
        seed=42,
        output_type="pil",
    )
    composed_image.save("slider_composed.png")
    print("Saved composed slider to slider_composed.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
