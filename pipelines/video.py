"""
FreeSliders Pipeline for Video Generation (CogVideoX, LTX-Video)
"""

import torch
from typing import List, Optional, Union, Dict
import numpy as np

try:
    from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler
    HAS_COGVIDEO = True
except ImportError:
    HAS_COGVIDEO = False

# Import ConceptPrompts - handle both package and direct execution
try:
    from ..freesliders import ConceptPrompts
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from freesliders import ConceptPrompts


class FreeSliderVideoPipeline:
    """
    FreeSliders implementation for video generation models.

    Supports CogVideoX and similar video diffusion models.
    """

    def __init__(
        self,
        model_id: str = "THUDM/CogVideoX-2b",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the video pipeline.

        Args:
            model_id: HuggingFace model ID
            device: Device to run on
            dtype: Data type for computations
        """
        if not HAS_COGVIDEO:
            raise ImportError("diffusers with CogVideoX support required")

        self.device = device
        self.dtype = dtype
        self.model_id = model_id

        # Load pipeline
        self.pipe = CogVideoXPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)

        # Enable memory optimizations
        self.pipe.enable_model_cpu_offload()

        # Cache components
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a text prompt for video model."""
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

    @torch.no_grad()
    def generate(
        self,
        concept: ConceptPrompts,
        scales: List[float],
        num_inference_steps: int = 50,
        intervention_step: int = 12,
        guidance_scale: float = 7.5,
        num_frames: int = 49,
        height: int = 480,
        width: int = 720,
        seed: Optional[int] = None,
        output_type: str = "tensor",
    ) -> Dict[float, torch.Tensor]:
        """
        Generate videos with concept slider control.

        Args:
            concept: ConceptPrompts
            scales: List of slider scales
            num_inference_steps: Number of denoising steps
            intervention_step: Step k
            guidance_scale: CFG scale
            num_frames: Number of video frames
            height: Output height
            width: Output width
            seed: Random seed
            output_type: "tensor" or "frames"

        Returns:
            Dictionary mapping scale -> video tensor
        """
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Encode prompts
        prompt_embeds_base = self.encode_prompt(concept.base)
        prompt_embeds_pos = self.encode_prompt(concept.positive)
        prompt_embeds_neg = self.encode_prompt(concept.negative)
        uncond_embeds = self.encode_prompt("")

        prompt_embeds_cfg = torch.cat([uncond_embeds, prompt_embeds_base])

        # Initialize scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Initialize latents for video
        latent_channels = self.transformer.config.in_channels
        latent_height = height // 8
        latent_width = width // 8
        latent_frames = num_frames // 4  # Temporal compression

        latents = torch.randn(
            (1, latent_frames, latent_channels, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents = latents * self.scheduler.init_noise_sigma

        # Step 1: Base diffusion until step k
        for i, t in enumerate(timesteps[:intervention_step]):
            latent_model_input = torch.cat([latents] * 2)
            # Timestep needs to be 1D tensor matching batch size
            timestep_batch = t.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_batch,
                encoder_hidden_states=prompt_embeds_cfg,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents_at_k = latents.clone()

        # Step 2: Generate for each scale
        results = {}

        for scale in scales:
            latents_scale = latents_at_k.clone()

            for i, t in enumerate(timesteps[intervention_step:]):
                # Base prediction with CFG
                latent_model_input = torch.cat([latents_scale] * 2)
                timestep_batch_cfg = t.expand(latent_model_input.shape[0])
                timestep_batch_single = t.expand(latents_scale.shape[0])

                noise_pred_base = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_batch_cfg,
                    encoder_hidden_states=prompt_embeds_cfg,
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred_base.chunk(2)
                noise_pred_neutral = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Positive and negative predictions
                noise_pred_pos = self.transformer(
                    hidden_states=latents_scale,
                    timestep=timestep_batch_single,
                    encoder_hidden_states=prompt_embeds_pos,
                ).sample

                noise_pred_neg = self.transformer(
                    hidden_states=latents_scale,
                    timestep=timestep_batch_single,
                    encoder_hidden_states=prompt_embeds_neg,
                ).sample

                # Apply concept modification
                noise_pred_mod = noise_pred_neutral + scale * (noise_pred_pos - noise_pred_neg)

                latents_scale = self.scheduler.step(noise_pred_mod, t, latents_scale).prev_sample

            # Decode latents to video frames
            if output_type == "tensor":
                results[scale] = latents_scale
            else:
                # Decode with VAE
                video_frames = self.decode_latents(latents_scale)
                results[scale] = video_frames

        return results

    def decode_latents(self, latents: torch.Tensor) -> List:
        """
        Decode latents to video frames using VAE.

        Args:
            latents: Latent tensor [B, T, C, H, W]

        Returns:
            List of PIL Images
        """
        from PIL import Image

        # CogVideoX VAE expects [B, C, T, H, W]
        latents = latents.permute(0, 2, 1, 3, 4)

        # Scale latents
        latents = latents / self.vae.config.scaling_factor

        with torch.no_grad():
            video = self.vae.decode(latents).sample

        # video shape: [B, C, T, H, W]
        video = video.squeeze(0)  # [C, T, H, W]
        video = video.permute(1, 2, 3, 0)  # [T, H, W, C]

        # Normalize to [0, 255]
        video = (video + 1) / 2
        video = video.clamp(0, 1)
        video = (video * 255).to(torch.uint8).cpu().numpy()

        # Convert to list of PIL Images
        frames = []
        for i in range(video.shape[0]):
            frame = Image.fromarray(video[i])
            frames.append(frame)

        return frames


def save_video_frames(
    video_tensor: torch.Tensor,
    output_path: str,
    fps: int = 8,
):
    """
    Save video tensor as frames or video file.

    Args:
        video_tensor: Video tensor [B, T, C, H, W]
        output_path: Output path
        fps: Frames per second
    """
    try:
        import imageio
    except ImportError:
        print("imageio required for video saving. Install with: pip install imageio[ffmpeg]")
        return

    # Convert tensor to numpy
    video = video_tensor.squeeze(0).cpu().float().numpy()
    video = (video + 1) / 2  # [-1, 1] -> [0, 1]
    video = (video * 255).astype(np.uint8)
    video = video.transpose(0, 2, 3, 1)  # [T, H, W, C]

    imageio.mimwrite(output_path, video, fps=fps)
    print(f"Saved video to {output_path}")


def save_video(
    frames: Union[List, torch.Tensor],
    output_path: str,
    fps: int = 8,
):
    """
    Save video frames (PIL Images or tensor) to video file.

    Args:
        frames: List of PIL Images or video tensor [B, T, C, H, W]
        output_path: Output path for video file
        fps: Frames per second
    """
    from PIL import Image

    # If tensor, use save_video_frames
    if isinstance(frames, torch.Tensor):
        save_video_frames(frames, output_path, fps)
        return

    # frames is a list of PIL Images
    if not frames:
        print("No frames to save")
        return

    try:
        import imageio
    except ImportError:
        print("imageio required for video saving. Install with: pip install imageio[ffmpeg]")
        # Fallback: save as individual frames
        from pathlib import Path
        output_dir = Path(output_path).parent / (Path(output_path).stem + "_frames")
        output_dir.mkdir(exist_ok=True, parents=True)
        for i, frame in enumerate(frames):
            frame.save(output_dir / f"frame_{i:04d}.png")
        print(f"Saved {len(frames)} frames to {output_dir}")
        return

    # Convert PIL Images to numpy arrays
    frame_arrays = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            frame_arrays.append(np.array(frame))
        else:
            frame_arrays.append(frame)

    imageio.mimwrite(output_path, frame_arrays, fps=fps)
    print(f"Saved video to {output_path}")
