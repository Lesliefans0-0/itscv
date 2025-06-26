# Inference-Time-Scaling Computer Vision (ITSCV)

This repository contains research code for exploring inference-time scaling techniques in computer vision using Vision Transformers (ViT). The project investigates iterative and recursive approaches to improve model performance during inference.

## Overview

The project implements three main model variants:
- **Standard ViT**: Baseline Vision Transformer implementation
- **Iterative ViT**: Adds iterative tokens that can be processed multiple times during inference
- **Recursive ViT**: Implements recursive processing of transformer layers

## Features

- PyTorch Lightning-based training framework
- Support for multiple ViT architectures
- TensorBoard logging and model checkpointing
- Configurable experiment management
- ImageNet-1k dataset support

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd itscv
```

2. Create and activate the conda environment:
```bash
conda env create -f environment_itscv.yaml
conda activate itscv
```

## Usage

### Training

1. Set up your ImageNet-1k dataset path in the configuration files
2. Run training with a specific experiment configuration:

```bash
python main.py --experiment <experiment_name>
```

### Configuration

Experiment configurations are stored in the `configs/` directory. Each YAML file defines:
- Model architecture and hyperparameters
- Training settings (batch size, learning rate, etc.)
- Data paths and processing settings

### Available Experiments

- `vit_base_*`: Standard Vision Transformer experiments
- `vit_iterative_*`: Iterative ViT experiments
- `vit_recursive_*`: Recursive ViT experiments

## Project Structure

```
itscv/
├── configs/              # Experiment configurations
├── data/                 # Data loading modules
├── lightning_models/     # PyTorch Lightning model wrappers
├── models/              # Model architectures
├── experiments/         # Training outputs (gitignored)
├── main.py             # Training entry point
├── config.py           # Configuration management
└── callbacks.py        # Custom training callbacks
```

## Research Notes

This project was developed to investigate whether inference-time scaling techniques could improve computer vision model performance. The research explored various approaches including iterative token processing and recursive layer applications.

## License

[Add your chosen license here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information here]
```
