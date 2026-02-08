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
├── freesliders.py              # Core FreeSliders implementation
├── metrics.py                  # Evaluation metrics (CR, CSM, SP)
├── astd.py                     # Auto Saturation & Traversal Detection
├── test_pipelines.py           # Generation test script (from paper)
├── test_condition_pipelines.py # Condition editing test script (custom)
├── pipelines/
│   ├── stable_diffusion.py     # Image pipeline
│   ├── video.py                # Video pipeline
│   └── audio.py                # Audio pipeline
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

### Generation (from paper)

Random noise에서 시작하여 이미지/오디오/비디오를 생성합니다.

```bash
# Image generation
python test_pipelines.py --image --base "A photo of a person" \
  --positive "A photo of a person smiling" --negative "A photo of a person frowning" \
  --scales -3 0 3

# Audio generation
python test_pipelines.py --audio --base "rain falling" \
  --positive "light rain falling" --negative "heavy rain falling" \
  --scales -3 0 3
```

### Condition Editing (custom - not in original paper)

> `test_condition_pipelines.py`는 원래 논문에는 없는 기능으로, SDEdit 방식을 활용하여 별도로 추가한 스크립트입니다.
> 기존 generate 플로우가 random noise에서 시작하는 것과 달리, **실제 입력(이미지/오디오/비디오)을 VAE로 인코딩한 뒤 noise를 추가**하여 concept slider로 편집합니다.

```
[기존 Generate]  Random Noise → Denoise → Output
[Condition Edit] Real Input → Encode → Add Noise → Denoise → Output
```

`--noise_strength`로 편집 강도를 조절합니다:
- `1.0` : 완전한 noise (기존 generate와 동일)
- `0.5` : 원본 구조를 보존하면서 concept 적용
- `0.3` : 원본을 거의 유지하며 약하게 편집

```bash
# Image editing - 웃는 정도 조절
python test_condition_pipelines.py --image --input photo.jpg \
  --base "A photo of a person" --positive "A photo of a person smiling" \
  --negative "A photo of a person frowning" --scales -2 0 2 --noise_strength 0.5

# Audio editing - 비 소리 강도 조절
python test_condition_pipelines.py --audio --input rain.wav \
  --base "rain falling" --positive "heavy rain" --negative "light drizzle" \
  --scales -1 0 1 --noise_strength 0.4

# Video editing
python test_condition_pipelines.py --video --input clip.mp4 \
  --base "a person walking" --positive "a person running" \
  --negative "a person standing" --scales -1 0 1 --noise_strength 0.5
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
