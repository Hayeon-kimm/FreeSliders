"""
FreeSliders: Training-Free, Modality-Agnostic Concept Sliders
for Fine-Grained Diffusion Control in Images, Audio, and Video

Based on the paper: https://arxiv.org/abs/2511.00103
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ConceptPrompts:
    """Defines a concept with base, positive, and negative prompts."""
    base: str
    positive: str
    negative: str

    def __repr__(self):
        return f"Concept(base='{self.base[:30]}...', pos='{self.positive[:30]}...', neg='{self.negative[:30]}...')"


class FreeSliders:
    """
    FreeSliders: Training-free concept control for diffusion models.

    The core idea is to directly estimate the Concept Slider update during inference
    by computing:
        ε_mod = ε_θ(x_t, c_base, t) + η * [ε_θ(x_t, c_+, t) - ε_θ(x_t, c_-, t)]

    This enables plug-and-play, modality-agnostic concept control without
    per-concept training or architectural modifications.
    """

    def __init__(
        self,
        model,
        scheduler,
        text_encoder: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize FreeSliders.

        Args:
            model: The diffusion model (UNet, DiT, etc.)
            scheduler: The noise scheduler
            text_encoder: Text encoder for prompts
            tokenizer: Tokenizer for text encoding
            device: Device to run on
            dtype: Data type for computations
        """
        self.model = model
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a text prompt into embeddings."""
        if self.tokenizer is None or self.text_encoder is None:
            raise ValueError("Text encoder and tokenizer must be provided")

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

    def predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        guidance_scale: Optional[float] = None,
        use_cfg: bool = False,
    ) -> torch.Tensor:
        """
        Predict noise using the model.

        Args:
            latents: Noisy latents
            timestep: Current timestep
            encoder_hidden_states: Text embeddings
            guidance_scale: CFG scale (if use_cfg=True)
            use_cfg: Whether to apply classifier-free guidance

        Returns:
            Predicted noise
        """
        if use_cfg and guidance_scale is not None:
            # Duplicate latents for CFG
            latents_input = torch.cat([latents] * 2)
            timestep_input = torch.cat([timestep] * 2) if timestep.dim() > 0 else timestep

            # Predict noise
            noise_pred = self.model(
                latents_input,
                timestep_input,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            # Apply CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = self.model(
                latents,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

        return noise_pred

    @torch.no_grad()
    def generate(
        self,
        concept: ConceptPrompts,
        scales: List[float],
        num_inference_steps: int = 50,
        intervention_step: int = 15,
        guidance_scale: float = 7.5,
        latents: Optional[torch.Tensor] = None,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None,
        output_type: str = "latent",
    ) -> dict:
        """
        Generate samples with concept slider control.

        Args:
            concept: ConceptPrompts with base, positive, and negative prompts
            scales: List of slider scales (η values)
            num_inference_steps: Number of denoising steps
            intervention_step: Step k after which to apply concept modification
            guidance_scale: CFG scale for neutral prompt
            latents: Initial latents (if None, randomly sampled)
            height: Output height
            width: Output width
            generator: Random generator for reproducibility
            output_type: "latent" or "pil"

        Returns:
            Dictionary with generated samples for each scale
        """
        # Encode prompts
        prompt_embeds_base = self.encode_prompt(concept.base)
        prompt_embeds_pos = self.encode_prompt(concept.positive)
        prompt_embeds_neg = self.encode_prompt(concept.negative)

        # For CFG, we need unconditional embeddings
        uncond_embeds = self.encode_prompt("")
        prompt_embeds_cfg = torch.cat([uncond_embeds, prompt_embeds_base])

        # Initialize scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Initialize latents
        if latents is None:
            latent_channels = self.model.config.in_channels
            latents = torch.randn(
                (1, latent_channels, height // 8, width // 8),
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )

        # Scale latents
        latents = latents * self.scheduler.init_noise_sigma

        # Step 1: Diffuse using neutral prompt until step k
        for i, t in enumerate(timesteps[:intervention_step]):
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.model(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_cfg,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Store the latents at intervention point
        latents_at_k = latents.clone()

        # Step 2: Generate samples for each scale
        results = {}

        for scale in scales:
            latents_scale = latents_at_k.clone()

            for i, t in enumerate(timesteps[intervention_step:]):
                # Predict noise for base prompt with CFG
                latent_model_input = torch.cat([latents_scale] * 2)

                noise_pred_base = self.model(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_cfg,
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred_base.chunk(2)
                noise_pred_neutral = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Predict noise for positive and negative prompts (without CFG)
                noise_pred_pos = self.model(
                    latents_scale,
                    t,
                    encoder_hidden_states=prompt_embeds_pos,
                ).sample

                noise_pred_neg = self.model(
                    latents_scale,
                    t,
                    encoder_hidden_states=prompt_embeds_neg,
                ).sample

                # Apply concept modification
                # ε_mod = ε_neutral + η * (ε_+ - ε_-)
                noise_pred_mod = noise_pred_neutral + scale * (noise_pred_pos - noise_pred_neg)

                # Step
                latents_scale = self.scheduler.step(noise_pred_mod, t, latents_scale).prev_sample

            results[scale] = latents_scale

        return results

    def compose_concepts(
        self,
        concepts: List[Tuple[ConceptPrompts, float]],
        latents: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """
        Compose multiple concept sliders together.

        Args:
            concepts: List of (ConceptPrompts, scale) tuples
            latents: Current latents
            timestep: Current timestep
            guidance_scale: CFG scale

        Returns:
            Modified noise prediction
        """
        # Start with base neutral prediction
        base_prompt = concepts[0][0].base
        prompt_embeds_base = self.encode_prompt(base_prompt)
        uncond_embeds = self.encode_prompt("")
        prompt_embeds_cfg = torch.cat([uncond_embeds, prompt_embeds_base])

        latent_model_input = torch.cat([latents] * 2)

        noise_pred_base = self.model(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds_cfg,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred_base.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Add each concept's direction
        for concept, scale in concepts:
            prompt_embeds_pos = self.encode_prompt(concept.positive)
            prompt_embeds_neg = self.encode_prompt(concept.negative)

            noise_pred_pos = self.model(
                latents,
                timestep,
                encoder_hidden_states=prompt_embeds_pos,
            ).sample

            noise_pred_neg = self.model(
                latents,
                timestep,
                encoder_hidden_states=prompt_embeds_neg,
            ).sample

            noise_pred = noise_pred + scale * (noise_pred_pos - noise_pred_neg)

        return noise_pred


class TextEmbeddingSlider:
    """
    Alternative approach: Text Embedding (TE) variant.

    Manipulates text embeddings directly instead of noise predictions.
    Less expressive but computationally simpler.
    """

    def __init__(
        self,
        model,
        scheduler,
        text_encoder,
        tokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

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

    def get_modified_embedding(
        self,
        concept: ConceptPrompts,
        scale: float,
    ) -> torch.Tensor:
        """
        Compute modified text embedding.

        e_mod = e_neutral + μ * (e_positive - e_negative)
        """
        e_neutral = self.encode_prompt(concept.base)
        e_positive = self.encode_prompt(concept.positive)
        e_negative = self.encode_prompt(concept.negative)

        e_mod = e_neutral + scale * (e_positive - e_negative)

        return e_mod

    @torch.no_grad()
    def generate(
        self,
        concept: ConceptPrompts,
        scales: List[float],
        num_inference_steps: int = 50,
        intervention_step: int = 15,
        guidance_scale: float = 7.5,
        latents: Optional[torch.Tensor] = None,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        """Generate with text embedding manipulation."""
        # Encode prompts
        e_neutral = self.encode_prompt(concept.base)
        uncond_embeds = self.encode_prompt("")

        # Initialize scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Initialize latents
        if latents is None:
            latent_channels = self.model.config.in_channels
            latents = torch.randn(
                (1, latent_channels, height // 8, width // 8),
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )

        latents = latents * self.scheduler.init_noise_sigma

        # Step 1: Diffuse using neutral embedding until step k
        prompt_embeds_cfg = torch.cat([uncond_embeds, e_neutral])

        for i, t in enumerate(timesteps[:intervention_step]):
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.model(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_cfg,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents_at_k = latents.clone()

        # Step 2: Generate for each scale using modified embeddings
        results = {}

        for scale in scales:
            latents_scale = latents_at_k.clone()

            # Get modified embedding
            e_mod = self.get_modified_embedding(concept, scale)
            prompt_embeds_mod = torch.cat([uncond_embeds, e_mod])

            for i, t in enumerate(timesteps[intervention_step:]):
                latent_model_input = torch.cat([latents_scale] * 2)

                noise_pred = self.model(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_mod,
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents_scale = self.scheduler.step(noise_pred, t, latents_scale).prev_sample

            results[scale] = latents_scale

        return results
