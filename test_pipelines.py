"""
FreeSliders Pipeline Test Script
Tests image, video, and audio pipelines for basic functionality.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from freesliders import ConceptPrompts


def test_image_pipeline(base: str, positive: str, negative: str, scales:list, output_dir: Path):
    """Test Stable Diffusion image pipeline."""
    print("\n" + "="*60)
    print("Testing Image Pipeline (Stable Diffusion)")
    print("="*60)

    try:
        from pipelines.stable_diffusion import (
            FreeSliderStableDiffusionPipeline,
            create_image_slider_grid
        )

        # Initialize pipeline
        print("Loading Stable Diffusion model...")
        pipeline = FreeSliderStableDiffusionPipeline(
            model_id="CompVis/stable-diffusion-v1-4",
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Define test concept
        concept = ConceptPrompts(
            base=base,
            positive=positive,
            negative=negative,
        )

        # # Test with minimal scales
        # scales = [-2, 0, 2]

        print("Generating images...")
        results = pipeline.generate(
            concept=concept,
            scales=scales,
            num_inference_steps=20,  # Fewer steps for testing
            intervention_step=5,
            guidance_scale=7.5,
            height=256,  # Smaller for faster testing
            width=256,
            seed=42,
            output_type="pil",
        )

        # Save results
        output_dir.mkdir(exist_ok=True, parents=True)

        grid = create_image_slider_grid(results)
        grid.save(output_dir / "test_image_slider.png")

        print(f"SUCCESS: Image slider saved to {output_dir / 'test_image_slider.png'}")
        return True

    except Exception as e:
        print(f"FAILED: Image pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_pipeline(base: str, positive: str, negative: str, scales: list, output_dir: Path):
    """Test video generation pipeline."""
    print("\n" + "="*60)
    print("Testing Video Pipeline")
    print("="*60)

    try:
        # Check if CogVideoX is available
        try:
            from diffusers import CogVideoXPipeline
            HAS_COGVIDEO = True
        except ImportError:
            HAS_COGVIDEO = False
            print("CogVideoX not available. Testing with mock implementation.")

        from pipelines.video import FreeSliderVideoPipeline

        if not HAS_COGVIDEO:
            print("SKIPPED: CogVideoX not installed. Install with:")
            print("  pip install diffusers[cogvideox]")
            return None

        # Initialize pipeline (this requires significant GPU memory)
        print("Loading CogVideoX model (requires ~10GB VRAM)...")

        # Check GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Available GPU memory: {gpu_mem:.1f} GB")
            if gpu_mem < 10:
                print("SKIPPED: Insufficient GPU memory for CogVideoX")
                return None

        pipeline = FreeSliderVideoPipeline(
            model_id="THUDM/CogVideoX-2b",
            device="cuda",
            dtype=torch.float16,
        )

        concept = ConceptPrompts(
            base=base,
            positive=positive,
            negative=negative,
        )

        # scales = [0, 2]

        print("Generating video frames...")
        results = pipeline.generate(
            concept=concept,
            scales=scales,
            num_inference_steps=50,
            intervention_step=12,
            num_frames=49,
            height=480,
            width=720,
            seed=42,
            output_type="frames",  # Get decoded frames instead of latents
        )

        # Save results
        output_dir.mkdir(exist_ok=True, parents=True)

        from pipelines.video import save_video

        for scale, frames in results.items():
            video_path = output_dir / f"{base}_{scale}.mp4"
            save_video(frames, str(video_path))

        print(f"SUCCESS: Video generation completed. Generated {len(results)} scales.")
        print(f"Videos saved to {output_dir}")
        return True

    except Exception as e:
        print(f"FAILED: Video pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_pipeline(base: str, positive: str, negative: str, scales: list, output_dir: Path):
    """Test audio generation pipeline."""
    print("\n" + "="*60)
    print("Testing Audio Pipeline")
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

        concept = ConceptPrompts(
            base=base,
            positive=positive,
            negative=negative,
        )

        # scales = [-1, 0, 1]

        print("Generating audio...")
        results = pipeline.generate(
            concept=concept,
            scales=scales,
            num_inference_steps=10,
            intervention_step=2,
            duration=2.0,  # Short for testing
            seed=42,
        )

        # Save results
        output_dir.mkdir(exist_ok=True, parents=True)

        for scale, audio in results.items():
            output_path = output_dir / f"{base}_{scale}.wav"
            save_audio(audio, str(output_path), sample_rate=44100)

        if HAS_STABLE_AUDIO:
            print(f"SUCCESS: Audio files saved to {output_dir}")
        else:
            print("SUCCESS: Mock audio generation completed (stable-audio-tools not installed)")

        return True

    except Exception as e:
        print(f"FAILED: Audio pipeline test failed with error: {e}")
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
    parser = argparse.ArgumentParser(description="Test FreeSliders pipelines")
    parser.add_argument("--image", action="store_true", help="Test image pipeline")
    parser.add_argument("--video", action="store_true", help="Test video pipeline")
    parser.add_argument("--audio", action="store_true", help="Test audio pipeline")
    parser.add_argument("--metrics", action="store_true", help="Test metrics")
    parser.add_argument("--astd", action="store_true", help="Test ASTD")
    parser.add_argument("--all", action="store_true", help="Run all tests")

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
    parser.add_argument("--output_dir", type=str, default="test_outputs",
                        help="Output directory for saving results")

    args = parser.parse_args()

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "test_outputs"

    # If no specific test selected, run all
    if not any([args.image, args.video, args.audio, args.metrics, args.astd, args.all]):
        args.all = True

    print("="*60)
    print("FreeSliders Pipeline Tests")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = {}

    if args.all or args.metrics:
        results["metrics"] = test_metrics()

    if args.all or args.astd:
        results["astd"] = test_astd()

    if args.all or args.image:
        results["image"] = test_image_pipeline(args.base, args.positive, args.negative, args.scales, output_dir)

    if args.all or args.audio:
        results["audio"] = test_audio_pipeline(args.base, args.positive, args.negative, args.scales, output_dir)

    if args.all or args.video:
        results["video"] = test_video_pipeline(args.base, args.positive, args.negative, args.scales, output_dir)

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
