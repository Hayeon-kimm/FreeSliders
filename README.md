# FreeSliders(Non-Official)

**Training-Free, Modality-Agnostic Concept Sliders for Fine-Grained Diffusion Control in Images, Audio, and Video**

Implementation based on [FreeSliders (arXiv:2511.00103)](https://arxiv.org/abs/2511.00103)

## Overview

FreeSliders enables fine-grained control over diffusion model outputs without requiring per-concept training or architecture-specific modifications. By directly estimating concept directions at inference time, it works across images, video, and audio.

### Key Features

- **Training-Free**: No LoRA or fine-tuning required
- **Plug-and-Play**: Works with any diffusion model
- **Multi-Modal**: Supports images, video, and audio
- **Composable**: Combine multiple concepts simultaneously

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/FreeSliders.git
cd FreeSliders

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from freesliders import ConceptPrompts
from pipelines.stable_diffusion import FreeSliderStableDiffusionPipeline

# Initialize pipeline
pipeline = FreeSliderStableDiffusionPipeline(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
)

# Define concept
concept = ConceptPrompts(
    base="A realistic image of a person.",
    positive="A realistic image of a person, smiling widely.",
    negative="A realistic image of a person, frowning.",
)

# Generate with different slider scales
scales = [-3, -2, -1, 0, 1, 2, 3]
results = pipeline.generate(
    concept=concept,
    scales=scales,
    seed=42,
)
```

## Core Algorithm

The FreeSliders formula:

```
ε_mod = ε_θ(x_t, c_base, t) + η × [ε_θ(x_t, c_+, t) - ε_θ(x_t, c_-, t)]
```

Where:
- `ε_θ`: Pre-trained diffusion model
- `c_base`: Base prompt
- `c_+` / `c_-`: Positive/negative concept prompts
- `η`: User-controlled scale

## Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **CR** (Conceptual Range) | Extent of concept variation | Higher ↑ |
| **CSM** (Conceptual Smoothness) | Uniformity of transitions | Lower ↓ |
| **SP** (Semantic Preservation) | Content preservation | Lower ↓ |
| **OS** (Overall Score) | Combined metric | Higher ↑ |

## Project Structure

```
FreeSliders/
├── freesliders.py       # Core FreeSliders implementation
├── metrics.py           # Evaluation metrics (CR, CSM, SP)
├── astd.py              # Auto Saturation & Traversal Detection
├── pipelines/
│   ├── stable_diffusion.py  # Image pipeline
│   ├── video.py             # Video pipeline
│   └── audio.py             # Audio pipeline
└── examples/
    ├── image_slider.py
    └── evaluate_slider.py
```

## Supported Models

### Images
- Stable Diffusion v1.4, v1.5
- Stable Diffusion 3, SDXL

### Video
- CogVideoX-2B
- LTX-Video

### Audio
- Stable Audio Open 1.0

## Running Tests

```bash

# Example
python test_pipelines.py --audio --base "rain falling" --positive "light rain falling" --negative "heavy rain falling" --scales -3 0 3 
```

## Citation

```bibtex
@article{ezra2025freesliders,
  title={FreeSliders: Training-Free, Modality-Agnostic Concept Sliders
         for Fine-Grained Diffusion Control in Images, Audio, and Video},
  author={Ezra, Rotem and Zisling, Hedi and Berman, Nimrod and
          Naiman, Ilan and Gorkor, Alexey and Nochumsohn, Liran and
          Nachmani, Eliya and Azencot, Omri},
  journal={arXiv preprint arXiv:2511.00103},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

Based on the research paper by Rotem Ezra et al. from Ben-Gurion University of the Negev.
