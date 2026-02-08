"""
FreeSliders Pipeline for Stable Diffusion (Images)
"""

import torch
from typing import List, Optional, Union, Dict, Callable
from PIL import Image
import numpy as np

try:
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

# Import ConceptPrompts - handle both package and direct execution
try:
    from ..freesliders import ConceptPrompts
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from freesliders import ConceptPrompts


class FreeSliderStableDiffusionPipeline:
    """
    FreeSliders implementation for Stable Diffusion.

    This pipeline enables training-free concept control for image generation.
    """

    def __init__(
        self,
        model_id: str = "CompVis/stable-diffusion-v1-4",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the pipeline.

        Args:
            model_id: HuggingFace model ID
            device: Device to run on
            dtype: Data type for computations
        """
        if not HAS_DIFFUSERS:
            raise ImportError("diffusers is required. Install with: pip install diffusers")

        self.device = device
        self.dtype = dtype
        self.model_id = model_id

        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)

        # Use DDIM scheduler for consistent results
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # Cache components
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a text prompt."""
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]

        return embeddings.to(self.dtype)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images."""
        latents = 1 / self.vae.config.scaling_factor * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def latents_to_pil(self, latents: torch.Tensor) -> List[Image.Image]:
        """Convert latents to PIL images."""
        images = self.decode_latents(latents)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype(np.uint8)
        pil_images = [Image.fromarray(img) for img in images]
        return pil_images

    @torch.no_grad()
    def generate(
        self,
        concept: ConceptPrompts,
        scales: List[float],
        num_inference_steps: int = 50,
        intervention_step: int = 15,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
    ) -> Dict[float, Union[torch.Tensor, Image.Image]]:
        """
        Generate images with concept slider control.

        Args:
            concept: ConceptPrompts with base, positive, and negative prompts
            scales: List of slider scales (η values)
            num_inference_steps: Number of denoising steps
            intervention_step: Step k after which to apply concept modification
            guidance_scale: CFG scale for neutral prompt
            height: Output height
            width: Output width
            seed: Random seed for reproducibility
            latents: Initial latents (optional)
            output_type: "pil" or "latent"

        Returns:
            Dictionary mapping scale -> generated image/latent
        """
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Encode prompts
        prompt_embeds_base = self.encode_prompt(concept.base)
        prompt_embeds_pos = self.encode_prompt(concept.positive)
        prompt_embeds_neg = self.encode_prompt(concept.negative)
        uncond_embeds = self.encode_prompt("")

        # CFG embeddings
        prompt_embeds_cfg = torch.cat([uncond_embeds, prompt_embeds_base])

        # Initialize scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Initialize latents
        if latents is None:
            latent_channels = self.unet.config.in_channels
            latents = torch.randn(
                (1, latent_channels, height // 8, width // 8),
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )

        latents = latents * self.scheduler.init_noise_sigma

        # Step 1: Diffuse using neutral prompt until step k
        for i, t in enumerate(timesteps[:intervention_step]):
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_cfg,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Store latents at intervention point
        latents_at_k = latents.clone()

        # Step 2: Generate for each scale
        results = {}

        for scale in scales:
            latents_scale = latents_at_k.clone()

            for i, t in enumerate(timesteps[intervention_step:]):
                # Predict noise for base with CFG
                latent_model_input = torch.cat([latents_scale] * 2)

                noise_pred_base = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_cfg,
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred_base.chunk(2)
                noise_pred_neutral = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Predict noise for positive and negative (without CFG)
                noise_pred_pos = self.unet(
                    latents_scale,
                    t,
                    encoder_hidden_states=prompt_embeds_pos,
                ).sample

                noise_pred_neg = self.unet(
                    latents_scale,
                    t,
                    encoder_hidden_states=prompt_embeds_neg,
                ).sample

                # Apply concept modification: ε_mod = ε_neutral + η * (ε_+ - ε_-)
                noise_pred_mod = noise_pred_neutral + scale * (noise_pred_pos - noise_pred_neg)

                # Step
                latents_scale = self.scheduler.step(noise_pred_mod, t, latents_scale).prev_sample

            if output_type == "pil":
                results[scale] = self.latents_to_pil(latents_scale)[0]
            else:
                results[scale] = latents_scale

        return results

    @torch.no_grad()
    def generate_composition(
        self,
        concepts: List[tuple],  # List of (ConceptPrompts, scale) tuples
        num_inference_steps: int = 50,
        intervention_step: int = 15,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        output_type: str = "pil",
    ) -> Union[torch.Tensor, Image.Image]:
        """
        Generate with multiple concepts composed together.

        Args:
            concepts: List of (ConceptPrompts, scale) tuples
            num_inference_steps: Number of denoising steps
            intervention_step: Step k
            guidance_scale: CFG scale
            height: Output height
            width: Output width
            seed: Random seed
            output_type: "pil" or "latent"

        Returns:
            Generated image/latent
        """
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Use first concept's base prompt
        base_prompt = concepts[0][0].base
        prompt_embeds_base = self.encode_prompt(base_prompt)
        uncond_embeds = self.encode_prompt("")
        prompt_embeds_cfg = torch.cat([uncond_embeds, prompt_embeds_base])

        # Encode all concept prompts
        concept_embeds = []
        for concept, scale in concepts:
            pos_embeds = self.encode_prompt(concept.positive)
            neg_embeds = self.encode_prompt(concept.negative)
            concept_embeds.append((pos_embeds, neg_embeds, scale))

        # Initialize
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        latents = torch.randn(
            (1, self.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents = latents * self.scheduler.init_noise_sigma

        # Step 1: Base diffusion
        for i, t in enumerate(timesteps[:intervention_step]):
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_cfg,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Step 2: Composed concept diffusion
        for i, t in enumerate(timesteps[intervention_step:]):
            # Base prediction with CFG
            latent_model_input = torch.cat([latents] * 2)

            noise_pred_base = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_cfg,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_base.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Add each concept direction
            for pos_embeds, neg_embeds, scale in concept_embeds:
                noise_pred_pos = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=pos_embeds,
                ).sample

                noise_pred_neg = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=neg_embeds,
                ).sample

                noise_pred = noise_pred + scale * (noise_pred_pos - noise_pred_neg)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        if output_type == "pil":
            return self.latents_to_pil(latents)[0]
        else:
            return latents


def create_image_slider_grid(
    results: Dict[float, Image.Image],
    rows: int = 1,
) -> Image.Image:
    """
    Create a grid visualization of slider results.

    Args:
        results: Dictionary mapping scale -> PIL Image
        rows: Number of rows in grid

    Returns:
        Combined grid image
    """
    scales = sorted(results.keys())
    images = [results[s] for s in scales]

    n = len(images)
    cols = (n + rows - 1) // rows

    w, h = images[0].size
    grid = Image.new('RGB', (cols * w, rows * h))

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, (col * w, row * h))

    return grid
