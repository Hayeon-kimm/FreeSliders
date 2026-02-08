"""
FreeSliders Pipeline for Audio Generation (Stable Audio Open)
"""

import torch
from typing import List, Optional, Dict
import numpy as np

try:
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    HAS_STABLE_AUDIO = True
except ImportError:
    HAS_STABLE_AUDIO = False

# Import ConceptPrompts - handle both package and direct execution
try:
    from ..freesliders import ConceptPrompts
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from freesliders import ConceptPrompts


class FreeSliderAudioPipeline:
    """
    FreeSliders implementation for audio generation.

    Supports Stable Audio Open and similar audio diffusion models.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-audio-open-1.0",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the audio pipeline.

        Args:
            model_id: Model identifier
            device: Device to run on
            dtype: Data type for computations
        """
        self.device = device
        self.dtype = dtype
        self.model_id = model_id

        if HAS_STABLE_AUDIO:
            self._load_model()
        else:
            print("stable-audio-tools not installed. Using placeholder implementation.")
            self.model = None

    def _load_model(self):
        """Load the audio model."""
        try:
            self.model, self.model_config = get_pretrained_model(self.model_id)
            self.model = self.model.to(self.device)
            self.sample_rate = self.model_config.get("sample_rate", 44100)
            self.sample_size = self.model_config.get("sample_size", 441000)
        except Exception as e:
            print(f"Failed to load audio model: {e}")
            print("Using placeholder implementation.")
            self.model = None
            self.sample_rate = 44100
            self.sample_size = 441000

    def encode_prompt(self, prompt: str) -> Dict:
        """Encode a text prompt for audio model."""
        # For Stable Audio, conditioning is handled differently
        return {"prompt": prompt}

    @torch.no_grad()
    def generate(
        self,
        concept: ConceptPrompts,
        scales: List[float],
        num_inference_steps: int = 36,
        intervention_step: int = 4,
        guidance_scale: float = 7.0,
        duration: float = 10.0,
        seed: Optional[int] = None,
        output_type: str = "waveform",
    ) -> Dict[float, torch.Tensor]:
        """
        Generate audio with concept slider control.

        Args:
            concept: ConceptPrompts
            scales: List of slider scales
            num_inference_steps: Number of denoising steps
            intervention_step: Step k
            guidance_scale: CFG scale
            duration: Audio duration in seconds
            seed: Random seed
            output_type: "waveform" or "spectrogram"

        Returns:
            Dictionary mapping scale -> audio tensor
        """
        if self.model is None:
            # Return placeholder data
            print("Model not loaded. Returning placeholder audio.")
            sample_length = int(duration * 44100)
            return {
                scale: torch.randn(1, sample_length)
                for scale in scales
            }

        if seed is not None:
            torch.manual_seed(seed)

        results = {}

        # Generate base audio first to establish structure
        conditioning_base = [{
            "prompt": concept.base,
            "seconds_start": 0,
            "seconds_total": duration,
        }]

        # For each scale, we need to modify the denoising process
        for scale in scales:
            scale = float(scale)
            # Create modified conditioning
            conditioning = [{
                "prompt": concept.base,
                "seconds_start": 0,
                "seconds_total": duration,
            }]

            # Compute sample_size from duration
            sample_size = int(duration * self.sample_rate)

            # Custom generation with FreeSliders modification
            audio = self._generate_with_slider(
                concept=concept,
                scale=scale,
                conditioning=conditioning,
                sample_size=sample_size,
                num_steps=num_inference_steps,
                intervention_step=intervention_step,
                guidance_scale=guidance_scale,
            )

            results[scale] = audio

        return results

    def _generate_with_slider(
        self,
        concept: ConceptPrompts,
        scale: float,
        conditioning: List[Dict],
        sample_size: int,
        num_steps: int,
        intervention_step: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Internal generation with slider modification.

        This implements the FreeSliders algorithm for audio.
        """
        # Get model components
        diffusion_model = self.model.model  # DiTWrapper
        conditioner = self.model.conditioner
        pretransform = self.model.pretransform
        diff_objective = self.model.diffusion_objective

        # Encode conditioning for base, positive, and negative prompts
        cond_base_tensors = conditioner([{"prompt": concept.base, "seconds_start": 0, "seconds_total": conditioning[0]["seconds_total"]}], self.device)
        cond_pos_tensors = conditioner([{"prompt": concept.positive, "seconds_start": 0, "seconds_total": conditioning[0]["seconds_total"]}], self.device)
        cond_neg_tensors = conditioner([{"prompt": concept.negative, "seconds_start": 0, "seconds_total": conditioning[0]["seconds_total"]}], self.device)

        # Get conditioning inputs in the format expected by DiTWrapper
        cond_base_inputs = self.model.get_conditioning_inputs(cond_base_tensors)
        cond_pos_inputs = self.model.get_conditioning_inputs(cond_pos_tensors)
        cond_neg_inputs = self.model.get_conditioning_inputs(cond_neg_tensors)

        # Initialize latents based on requested sample_size
        latent_size = sample_size // pretransform.downsampling_ratio
        latents = torch.randn(1, self.model.io_channels, latent_size, device=self.device)

        # Cast to model dtype
        model_dtype = next(diffusion_model.parameters()).dtype
        latents = latents.type(model_dtype)
        cond_base_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in cond_base_inputs.items()}
        cond_pos_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in cond_pos_inputs.items()}
        cond_neg_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in cond_neg_inputs.items()}

        # Build time schedule and sampling based on diffusion objective
        if diff_objective == "v":
            import k_diffusion as K

            # Wrap DiTWrapper in VDenoiser for proper sigma<->t conversion and scaling
            denoiser_base = K.external.VDenoiser(diffusion_model)

            # k-diffusion sigma schedule
            sigma_min, sigma_max, rho = 0.01, 100, 1.0
            sigmas = K.sampling.get_sigmas_polyexponential(num_steps, sigma_min, sigma_max, rho, device=self.device)
            latents = latents * sigmas[0]

            # Euler sampling with FreeSliders modification
            for i in range(num_steps):
                # sigma must be a 1D tensor with batch dim for DiT compatibility
                sigma = sigmas[i].unsqueeze(0).to(self.device)
                sigma_next = sigmas[i + 1]

                if i < intervention_step:
                    denoised = denoiser_base(latents, sigma, cfg_scale=guidance_scale, batch_cfg=True, **cond_base_inputs)
                else:
                    denoised_base = denoiser_base(latents, sigma, cfg_scale=guidance_scale, batch_cfg=True, **cond_base_inputs)
                    denoised_pos = denoiser_base(latents, sigma, cfg_scale=1.0, batch_cfg=True, **cond_pos_inputs)
                    denoised_neg = denoiser_base(latents, sigma, cfg_scale=1.0, batch_cfg=True, **cond_neg_inputs)
                    denoised = denoised_base + scale * (denoised_pos - denoised_neg)

                # Euler step: d = (x - denoised) / sigma
                d = (latents - denoised) / sigma
                latents = latents + (sigma_next - sigma) * d

        else:
            # Rectified flow time schedule
            import math
            rf_sigma_max = 1
            logsnr_max = -6
            logsnr = torch.linspace(logsnr_max, 2, num_steps + 1)
            t_schedule = torch.sigmoid(-logsnr)
            t_schedule[0] = rf_sigma_max
            t_schedule[-1] = 0

            ts = latents.new_ones([latents.shape[0]])

            for i in range(num_steps):
                t_curr = t_schedule[i]
                t_next = t_schedule[i + 1]
                dt = t_next - t_curr
                t_curr_tensor = t_curr * ts

                if i < intervention_step:
                    v = diffusion_model(latents, t_curr_tensor, cfg_scale=guidance_scale, batch_cfg=True, **cond_base_inputs)
                else:
                    v_base = diffusion_model(latents, t_curr_tensor, cfg_scale=guidance_scale, batch_cfg=True, **cond_base_inputs)
                    v_pos = diffusion_model(latents, t_curr_tensor, cfg_scale=1.0, batch_cfg=True, **cond_pos_inputs)
                    v_neg = diffusion_model(latents, t_curr_tensor, cfg_scale=1.0, batch_cfg=True, **cond_neg_inputs)
                    v = v_base + scale * (v_pos - v_neg)

                latents = latents + dt * v

        # Decode latents to audio
        with torch.no_grad():
            latents = latents.to(next(pretransform.parameters()).dtype)
            audio = pretransform.decode(latents)

        return audio.squeeze(0)


def save_audio(
    audio_tensor: torch.Tensor,
    output_path: str,
    sample_rate: int = 44100,
):
    """
    Save audio tensor to file.

    Args:
        audio_tensor: Audio tensor [T] or [C, T]
        output_path: Output path (.wav)
        sample_rate: Sample rate
    """
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile required for audio saving. Install with: pip install soundfile")
        return

    # Handle both tensor and numpy input
    if isinstance(audio_tensor, torch.Tensor):
        audio = audio_tensor.detach().cpu().numpy()
    else:
        audio = audio_tensor

    # Squeeze extra dimensions
    audio = np.squeeze(audio)

    # Handle different shapes
    if audio.ndim == 2:
        # If [C, T] format, transpose to [T, C]
        if audio.shape[0] < audio.shape[1]:
            audio = audio.T

    # Normalize to [-1, 1] range if needed
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = audio / max(abs(audio.max()), abs(audio.min()))

    try:
        sf.write(output_path, audio, sample_rate)
        print(f"Saved audio to {output_path}")
    except Exception as e:
        print(f"Failed to save audio: {e}")


def plot_spectrogram(
    audio_tensor: torch.Tensor,
    sample_rate: int = 44100,
    save_path: Optional[str] = None,
):
    """
    Plot spectrogram of audio.

    Args:
        audio_tensor: Audio tensor
        sample_rate: Sample rate
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display
    except ImportError:
        print("matplotlib and librosa required for spectrogram plotting")
        return

    audio = audio_tensor.squeeze().cpu().numpy()

    # Compute spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sample_rate)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectrogram to {save_path}")

    plt.show()
