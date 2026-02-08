"""
FreeSliders Condition Pipeline Test Script (SDEdit Approach)
Tests image, video, and audio pipelines for real input editing.

Instead of generating from random noise, this script:
1. Encodes a real input (image/audio/video) into latent space
2. Adds noise at a specified strength
3. Denoises with concept slider control

This is equivalent to the SDEdit approach:
  Real Input → Encode(VAE) → Add Noise(at t_start) → Phase1(base denoise) → Phase2(concept-guided) → Decode → Output
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from freesliders import ConceptPrompts


def test_image_pipeline(input_path: str, noise_strength: float, base: str, positive: str, negative: str, scales: list, output_dir: Path):
    """Test Stable Diffusion image pipeline with real image input (SDEdit)."""
    print("\n" + "="*60)
    print("Testing Image Condition Pipeline (Stable Diffusion + SDEdit)")
    print("="*60)

    try:
        from PIL import Image
        from torchvision import transforms
        from pipelines.stable_diffusion import (
            FreeSliderStableDiffusionPipeline,
            create_image_slider_grid,
        )

        # Initialize pipeline
        print("Loading Stable Diffusion model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        pipeline = FreeSliderStableDiffusionPipeline(
            model_id="CompVis/stable-diffusion-v1-4",
            device=device,
            dtype=dtype,
        )

        height, width = 256, 256

        # 1) Load & preprocess image
        print(f"Loading input image: {input_path}")
        image = Image.open(input_path).convert("RGB").resize((width, height))
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device, dtype=dtype)
        image_tensor = image_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]

        # 2) VAE encode -> clean latents
        print("Encoding image to latent space...")
        with torch.no_grad():
            clean_latents = pipeline.vae.encode(image_tensor).latent_dist.sample()
            clean_latents = clean_latents * pipeline.vae.config.scaling_factor

        # 3) Compute start step from noise_strength
        num_inference_steps = 20
        intervention_step = 5
        guidance_scale = 7.5
        seed = 42

        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps_full = pipeline.scheduler.timesteps

        start_step = int(num_inference_steps * (1 - noise_strength))
        start_step = max(0, min(start_step, num_inference_steps - 1))
        t_start = timesteps_full[start_step]

        print(f"Noise strength: {noise_strength}, Start step: {start_step}/{num_inference_steps}, t_start: {t_start}")

        # 4) Add noise at t_start
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        noise = torch.randn(clean_latents.shape, generator=generator, device=device, dtype=dtype)
        noisy_latents = pipeline.scheduler.add_noise(clean_latents, noise, t_start)

        # 5) Encode prompts
        concept = ConceptPrompts(base=base, positive=positive, negative=negative)

        prompt_embeds_base = pipeline.encode_prompt(concept.base)
        prompt_embeds_pos = pipeline.encode_prompt(concept.positive)
        prompt_embeds_neg = pipeline.encode_prompt(concept.negative)
        uncond_embeds = pipeline.encode_prompt("")
        prompt_embeds_cfg = torch.cat([uncond_embeds, prompt_embeds_base])

        # 6) Denoising from start_step
        timesteps = timesteps_full[start_step:]

        # Adjust intervention_step relative to the truncated schedule
        relative_intervention = min(intervention_step, len(timesteps) // 2)

        print(f"Denoising {len(timesteps)} steps (intervention at relative step {relative_intervention})...")

        # Phase 1: Base denoise until relative intervention step
        latents = noisy_latents.clone()

        for i, t in enumerate(timesteps[:relative_intervention]):
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = pipeline.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_cfg,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample

        latents_at_k = latents.clone()

        # Phase 2: Concept-guided denoise for each scale
        results = {}

        for scale in scales:
            latents_scale = latents_at_k.clone()

            for i, t in enumerate(timesteps[relative_intervention:]):
                # Base prediction with CFG
                latent_model_input = torch.cat([latents_scale] * 2)

                noise_pred_base = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_cfg,
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred_base.chunk(2)
                noise_pred_neutral = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Positive and negative predictions (without CFG)
                noise_pred_pos = pipeline.unet(
                    latents_scale, t, encoder_hidden_states=prompt_embeds_pos,
                ).sample

                noise_pred_neg = pipeline.unet(
                    latents_scale, t, encoder_hidden_states=prompt_embeds_neg,
                ).sample

                # Apply concept modification: ε_mod = ε_neutral + η * (ε_+ - ε_-)
                noise_pred_mod = noise_pred_neutral + scale * (noise_pred_pos - noise_pred_neg)

                latents_scale = pipeline.scheduler.step(noise_pred_mod, t, latents_scale).prev_sample

            results[scale] = pipeline.latents_to_pil(latents_scale)[0]

        # Save results
        output_dir.mkdir(exist_ok=True, parents=True)

        grid = create_image_slider_grid(results)
        grid_path = output_dir / "test_condition_image_slider.png"
        grid.save(grid_path)

        print(f"SUCCESS: Image condition slider saved to {grid_path}")
        return True

    except Exception as e:
        print(f"FAILED: Image condition pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_pipeline(input_path: str, noise_strength: float, base: str, positive: str, negative: str, scales: list, output_dir: Path):
    """Test audio pipeline with real audio input (SDEdit)."""
    print("\n" + "="*60)
    print("Testing Audio Condition Pipeline (Stable Audio + SDEdit)")
    print("="*60)

    try:
        from pipelines.audio import FreeSliderAudioPipeline, save_audio

        # Check if stable-audio-tools is available
        try:
            from stable_audio_tools import get_pretrained_model
            HAS_STABLE_AUDIO = True
        except ImportError:
            HAS_STABLE_AUDIO = False

        if not HAS_STABLE_AUDIO:
            print("stable-audio-tools not installed. Testing mock implementation...")

        pipeline = FreeSliderAudioPipeline(
            model_id="stabilityai/stable-audio-open-1.0",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        if pipeline.model is None:
            print("SKIPPED: Audio model not loaded (stable-audio-tools not installed)")
            print("  Install with: pip install stable-audio-tools")
            return None

        device = pipeline.device
        concept = ConceptPrompts(base=base, positive=positive, negative=negative)

        # 1) Load audio
        print(f"Loading input audio: {input_path}")
        try:
            import soundfile as sf
        except ImportError:
            print("FAILED: soundfile is required. Install with: pip install soundfile")
            return False

        audio_data, sr = sf.read(input_path)
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).to(device)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # [C, T]
        elif audio_tensor.ndim == 2 and audio_tensor.shape[1] < audio_tensor.shape[0]:
            audio_tensor = audio_tensor.T  # [T, C] -> [C, T]
        audio_tensor = audio_tensor.unsqueeze(0)  # [B, C, T]

        # 2) Get model components
        diffusion_model = pipeline.model.model
        conditioner = pipeline.model.conditioner
        pretransform = pipeline.model.pretransform
        diff_objective = pipeline.model.diffusion_objective

        model_dtype = next(diffusion_model.parameters()).dtype
        audio_tensor = audio_tensor.to(model_dtype)

        # 3) Encode audio -> clean latents
        print("Encoding audio to latent space...")
        with torch.no_grad():
            clean_latents = pretransform.encode(audio_tensor)

        # 4) Encode conditioning
        duration = audio_data.shape[0] / sr if audio_data.ndim == 1 else audio_data.shape[0] / sr

        cond_base_tensors = conditioner([{"prompt": concept.base, "seconds_start": 0, "seconds_total": duration}], device)
        cond_pos_tensors = conditioner([{"prompt": concept.positive, "seconds_start": 0, "seconds_total": duration}], device)
        cond_neg_tensors = conditioner([{"prompt": concept.negative, "seconds_start": 0, "seconds_total": duration}], device)

        cond_base_inputs = pipeline.model.get_conditioning_inputs(cond_base_tensors)
        cond_pos_inputs = pipeline.model.get_conditioning_inputs(cond_pos_tensors)
        cond_neg_inputs = pipeline.model.get_conditioning_inputs(cond_neg_tensors)

        cond_base_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in cond_base_inputs.items()}
        cond_pos_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in cond_pos_inputs.items()}
        cond_neg_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in cond_neg_inputs.items()}

        # 5) Add noise based on diffusion objective
        num_steps = 36
        intervention_step = 4
        guidance_scale = 7.0

        start_step = int(num_steps * (1 - noise_strength))
        start_step = max(0, min(start_step, num_steps - 1))

        print(f"Noise strength: {noise_strength}, Start step: {start_step}/{num_steps}")

        noise = torch.randn_like(clean_latents)

        results = {}

        for scale in scales:
            scale = float(scale)

            if diff_objective == "v":
                import k_diffusion as K

                denoiser_base = K.external.VDenoiser(diffusion_model)
                sigma_min, sigma_max, rho = 0.01, 100, 1.0
                sigmas = K.sampling.get_sigmas_polyexponential(num_steps, sigma_min, sigma_max, rho, device=device)

                # Add noise at start_step sigma level
                sigma_start = sigmas[start_step]
                noisy_latents = clean_latents + sigma_start * noise

                # Truncated schedule from start_step
                relative_intervention = min(intervention_step, (num_steps - start_step) // 2)
                latents = noisy_latents.clone()

                for i in range(start_step, num_steps):
                    sigma = sigmas[i].unsqueeze(0).to(device)
                    sigma_next = sigmas[i + 1]
                    relative_i = i - start_step

                    if relative_i < relative_intervention:
                        denoised = denoiser_base(latents, sigma, cfg_scale=guidance_scale, batch_cfg=True, **cond_base_inputs)
                    else:
                        denoised_base = denoiser_base(latents, sigma, cfg_scale=guidance_scale, batch_cfg=True, **cond_base_inputs)
                        denoised_pos = denoiser_base(latents, sigma, cfg_scale=1.0, batch_cfg=True, **cond_pos_inputs)
                        denoised_neg = denoiser_base(latents, sigma, cfg_scale=1.0, batch_cfg=True, **cond_neg_inputs)
                        denoised = denoised_base + scale * (denoised_pos - denoised_neg)

                    d = (latents - denoised) / sigma
                    latents = latents + (sigma_next - sigma) * d

            else:
                # Rectified flow
                import math
                rf_sigma_max = 1
                logsnr_max = -6
                logsnr = torch.linspace(logsnr_max, 2, num_steps + 1)
                t_schedule = torch.sigmoid(-logsnr)
                t_schedule[0] = rf_sigma_max
                t_schedule[-1] = 0

                # Add noise at start_step level: noisy = (1 - t) * clean + t * noise
                t_start = t_schedule[start_step]
                noisy_latents = (1 - t_start) * clean_latents + t_start * noise

                ts = clean_latents.new_ones([clean_latents.shape[0]])

                relative_intervention = min(intervention_step, (num_steps - start_step) // 2)
                latents = noisy_latents.clone()

                for i in range(start_step, num_steps):
                    t_curr = t_schedule[i]
                    t_next = t_schedule[i + 1]
                    dt = t_next - t_curr
                    t_curr_tensor = t_curr * ts
                    relative_i = i - start_step

                    if relative_i < relative_intervention:
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
                audio_out = pretransform.decode(latents)

            results[scale] = audio_out.squeeze(0)

        # Save results
        output_dir.mkdir(exist_ok=True, parents=True)

        for scale_val, audio in results.items():
            output_path = output_dir / f"condition_{base}_{scale_val}.wav"
            save_audio(audio, str(output_path), sample_rate=pipeline.sample_rate)

        print(f"SUCCESS: Audio condition results saved to {output_dir}")
        return True

    except Exception as e:
        print(f"FAILED: Audio condition pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_pipeline(input_path: str, noise_strength: float, base: str, positive: str, negative: str, scales: list, output_dir: Path):
    """Test video pipeline with real video input (SDEdit)."""
    print("\n" + "="*60)
    print("Testing Video Condition Pipeline (CogVideoX + SDEdit)")
    print("="*60)

    try:
        try:
            from diffusers import CogVideoXPipeline
            HAS_COGVIDEO = True
        except ImportError:
            HAS_COGVIDEO = False

        from pipelines.video import FreeSliderVideoPipeline, save_video

        if not HAS_COGVIDEO:
            print("SKIPPED: CogVideoX not installed. Install with:")
            print("  pip install diffusers[cogvideox]")
            return None

        # Check GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Available GPU memory: {gpu_mem:.1f} GB")
            if gpu_mem < 10:
                print("SKIPPED: Insufficient GPU memory for CogVideoX")
                return None
        else:
            print("SKIPPED: CUDA not available for CogVideoX")
            return None

        # Initialize pipeline
        print("Loading CogVideoX model (requires ~10GB VRAM)...")
        device = "cuda"
        dtype = torch.float16

        video_pipeline = FreeSliderVideoPipeline(
            model_id="THUDM/CogVideoX-2b",
            device=device,
            dtype=dtype,
        )

        num_frames = 49
        height, width = 480, 720

        # 1) Load video frames
        print(f"Loading input video: {input_path}")
        from PIL import Image

        try:
            import imageio
            reader = imageio.get_reader(input_path)
            frames = []
            for frame in reader:
                img = Image.fromarray(frame).resize((width, height))
                frames.append(img)
                if len(frames) >= num_frames:
                    break
            reader.close()
        except ImportError:
            print("FAILED: imageio is required. Install with: pip install imageio[ffmpeg]")
            return False

        # Pad or truncate frames to num_frames
        while len(frames) < num_frames:
            frames.append(frames[-1])
        frames = frames[:num_frames]

        # Convert frames to tensor [B, C, T, H, W] in [-1, 1]
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        frame_tensors = [to_tensor(f) for f in frames]
        video_tensor = torch.stack(frame_tensors, dim=1).unsqueeze(0).to(device, dtype=dtype)  # [B, C, T, H, W]
        video_tensor = video_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]

        # 2) VAE encode -> clean latents
        print("Encoding video to latent space...")
        with torch.no_grad():
            clean_latents = video_pipeline.vae.encode(video_tensor).latent_dist.sample()
            clean_latents = clean_latents * video_pipeline.vae.config.scaling_factor

        # CogVideoX transformer expects [B, T, C, H, W]
        clean_latents = clean_latents.permute(0, 2, 1, 3, 4)

        # 3) Compute start step from noise_strength
        num_inference_steps = 50
        intervention_step = 12
        guidance_scale = 7.5
        seed = 42

        video_pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps_full = video_pipeline.scheduler.timesteps

        start_step = int(num_inference_steps * (1 - noise_strength))
        start_step = max(0, min(start_step, num_inference_steps - 1))
        t_start = timesteps_full[start_step]

        print(f"Noise strength: {noise_strength}, Start step: {start_step}/{num_inference_steps}, t_start: {t_start}")

        # 4) Add noise at t_start
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        noise = torch.randn(clean_latents.shape, generator=generator, device=device, dtype=dtype)
        noisy_latents = video_pipeline.scheduler.add_noise(clean_latents, noise, t_start)

        # 5) Encode prompts
        concept = ConceptPrompts(base=base, positive=positive, negative=negative)

        prompt_embeds_base = video_pipeline.encode_prompt(concept.base)
        prompt_embeds_pos = video_pipeline.encode_prompt(concept.positive)
        prompt_embeds_neg = video_pipeline.encode_prompt(concept.negative)
        uncond_embeds = video_pipeline.encode_prompt("")
        prompt_embeds_cfg = torch.cat([uncond_embeds, prompt_embeds_base])

        # 6) Denoising from start_step
        timesteps = timesteps_full[start_step:]
        relative_intervention = min(intervention_step, len(timesteps) // 2)

        print(f"Denoising {len(timesteps)} steps (intervention at relative step {relative_intervention})...")

        # Phase 1: Base denoise
        latents = noisy_latents.clone()

        for i, t in enumerate(timesteps[:relative_intervention]):
            latent_model_input = torch.cat([latents] * 2)
            timestep_batch = t.expand(latent_model_input.shape[0])

            noise_pred = video_pipeline.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_batch,
                encoder_hidden_states=prompt_embeds_cfg,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = video_pipeline.scheduler.step(noise_pred, t, latents).prev_sample

        latents_at_k = latents.clone()

        # Phase 2: Concept-guided denoise for each scale
        results = {}

        for scale in scales:
            latents_scale = latents_at_k.clone()

            for i, t in enumerate(timesteps[relative_intervention:]):
                latent_model_input = torch.cat([latents_scale] * 2)
                timestep_batch_cfg = t.expand(latent_model_input.shape[0])
                timestep_batch_single = t.expand(latents_scale.shape[0])

                noise_pred_base = video_pipeline.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_batch_cfg,
                    encoder_hidden_states=prompt_embeds_cfg,
                ).sample

                noise_pred_uncond, noise_pred_text = noise_pred_base.chunk(2)
                noise_pred_neutral = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                noise_pred_pos = video_pipeline.transformer(
                    hidden_states=latents_scale,
                    timestep=timestep_batch_single,
                    encoder_hidden_states=prompt_embeds_pos,
                ).sample

                noise_pred_neg = video_pipeline.transformer(
                    hidden_states=latents_scale,
                    timestep=timestep_batch_single,
                    encoder_hidden_states=prompt_embeds_neg,
                ).sample

                noise_pred_mod = noise_pred_neutral + scale * (noise_pred_pos - noise_pred_neg)

                latents_scale = video_pipeline.scheduler.step(noise_pred_mod, t, latents_scale).prev_sample

            # Decode latents to video frames
            video_frames = video_pipeline.decode_latents(latents_scale)
            results[scale] = video_frames

        # Save results
        output_dir.mkdir(exist_ok=True, parents=True)

        for scale_val, frames_out in results.items():
            video_path = output_dir / f"condition_{base}_{scale_val}.mp4"
            save_video(frames_out, str(video_path))

        print(f"SUCCESS: Video condition results saved to {output_dir}")
        return True

    except Exception as e:
        print(f"FAILED: Video condition pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test evaluation metrics."""
    print("\n" + "="*60)
    print("Testing Evaluation Metrics")
    print("="*60)

    try:
        from metrics import (
            SliderEvaluator,
            CLIPAlignment,
            LPIPSDistance,
            SliderMetrics,
        )

        print("Initializing CLIP alignment model...")
        alignment = CLIPAlignment()

        print("Initializing LPIPS distance model...")
        lpips_dist = LPIPSDistance()

        print("Creating evaluator...")
        evaluator = SliderEvaluator(
            alignment_model=alignment,
            perceptual_distance=lpips_dist,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        print("SUCCESS: Metrics modules loaded correctly")
        return True

    except Exception as e:
        print(f"FAILED: Metrics test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_astd():
    """Test ASTD module."""
    print("\n" + "="*60)
    print("Testing ASTD Module")
    print("="*60)

    try:
        from astd import ASTD, ASTDResult

        # Mock functions for testing
        def mock_alignment(sample, prompt):
            return 0.5 + torch.rand(1).item() * 0.3

        def mock_perceptual(s1, s2):
            return torch.rand(1).item() * 0.5

        astd = ASTD(
            alignment_func=mock_alignment,
            perceptual_func=mock_perceptual,
            trade_off_ratio=1.0,
            candidate_scales=[0, 0.5, 1, 2, 4],
        )

        print("SUCCESS: ASTD module loaded correctly")
        return True

    except Exception as e:
        print(f"FAILED: ASTD test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test FreeSliders condition pipelines (SDEdit approach)")
    parser.add_argument("--image", action="store_true", help="Test image condition pipeline")
    parser.add_argument("--video", action="store_true", help="Test video condition pipeline")
    parser.add_argument("--audio", action="store_true", help="Test audio condition pipeline")
    parser.add_argument("--metrics", action="store_true", help="Test metrics")
    parser.add_argument("--astd", action="store_true", help="Test ASTD")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    # Input and noise arguments
    parser.add_argument("--input", type=str, required=True, help="Input file path (image/audio/video)")
    parser.add_argument("--noise_strength", type=float, default=0.5,
                        help="Noise strength 0.0~1.0 (1.0=full noise like generate, 0.0=no noise)")

    # Concept prompts arguments
    parser.add_argument("--base", type=str, default="A realistic image of a person.",
                        help="Base prompt for the concept")
    parser.add_argument("--positive", type=str, default="A realistic image of a person, smiling widely, very happy.",
                        help="Positive prompt for the concept")
    parser.add_argument("--negative", type=str, default="A realistic image of a person, frowning, very sad.",
                        help="Negative prompt for the concept")

    # scales list
    parser.add_argument("--scales", type=float, nargs="+", default=[-1, 0, 1],
                        help="Scales for edited intensity (e.g. --scales -3 0 3)")

    # Output directory argument
    parser.add_argument("--output_dir", type=str, default="test_condition_outputs",
                        help="Output directory for saving results")

    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Validate noise_strength range
    if not 0.0 <= args.noise_strength <= 1.0:
        print(f"Error: noise_strength must be between 0.0 and 1.0, got {args.noise_strength}")
        sys.exit(1)

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "test_condition_outputs"

    # If no specific test selected, run all
    if not any([args.image, args.video, args.audio, args.metrics, args.astd, args.all]):
        args.all = True

    print("="*60)
    print("FreeSliders Condition Pipeline Tests (SDEdit)")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Input file: {args.input}")
    print(f"Noise strength: {args.noise_strength}")

    results = {}

    if args.all or args.metrics:
        results["metrics"] = test_metrics()

    if args.all or args.astd:
        results["astd"] = test_astd()

    if args.all or args.image:
        results["image"] = test_image_pipeline(args.input, args.noise_strength, args.base, args.positive, args.negative, args.scales, output_dir)

    if args.all or args.audio:
        results["audio"] = test_audio_pipeline(args.input, args.noise_strength, args.base, args.positive, args.negative, args.scales, output_dir)

    if args.all or args.video:
        results["video"] = test_video_pipeline(args.input, args.noise_strength, args.base, args.positive, args.negative, args.scales, output_dir)

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, result in results.items():
        if result is True:
            status = "PASSED"
        elif result is False:
            status = "FAILED"
        else:
            status = "SKIPPED"
        print(f"  {test_name}: {status}")

    # Return exit code
    failed = sum(1 for r in results.values() if r is False)
    return failed


if __name__ == "__main__":
    sys.exit(main())
