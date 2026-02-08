#!/bin/bash
# FreeSliders Conda Environment Setup

ENV_NAME="freesliders"

echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing core dependencies..."
pip install diffusers>=0.25.0
pip install transformers>=4.30.0
pip install accelerate>=0.20.0
pip install scipy>=1.7.0
pip install numpy>=1.21.0

echo "Installing evaluation dependencies..."
pip install lpips>=0.1.4
pip install Pillow>=9.0.0

echo "Installing audio dependencies..."
pip install soundfile>=0.12.0
pip install librosa>=0.10.0

echo "Installing video dependencies..."
pip install imageio>=2.25.0
pip install imageio-ffmpeg>=0.4.8

echo "Installing visualization..."
pip install matplotlib>=3.5.0

echo "Environment setup complete!"
echo "Activate with: conda activate $ENV_NAME"
