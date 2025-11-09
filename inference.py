"""Inference utilities for generating photorealistic images from underpaintings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import Config
from model import UnderpaintingToPhotoModel
from utils import analyze_underpainting, load_image, stack_feature_maps, tensor_to_image


def load_model(checkpoint_path: Path, cfg: Config) -> UnderpaintingToPhotoModel:
    """Load a trained model from ``checkpoint_path``."""

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = UnderpaintingToPhotoModel(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        feature_channels=cfg.feature_channels,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_underpainting(image_path: Path, cfg: Config) -> torch.Tensor:
    """Load and analyse an underpainting returning a tensor ready for inference."""

    image = load_image(image_path, mode="RGB")
    size = (cfg.image_size[1], cfg.image_size[0])
    image = image.resize(size)
    array = np.array(image)
    features = analyze_underpainting(array)
    stacked = stack_feature_maps(features)
    tensor = torch.from_numpy(stacked).float()
    tensor = tensor * 2.0 - 1.0
    tensor = tensor.unsqueeze(0)  # add batch dimension
    return tensor


def generate_photo_from_underpainting(model, input_image_path, output_image_path, config):
    """Generate a photorealistic image from an underpainting and save it."""

    device = next(model.parameters()).device
    conditioning = preprocess_underpainting(Path(input_image_path), config).to(device)
    with torch.no_grad():
        prediction = model(conditioning)
    image = tensor_to_image(prediction[0])
    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_image_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference using a trained model")
    parser.add_argument("--input", type=Path, required=True, help="Path to the rough underpainting")
    parser.add_argument("--output", type=Path, required=True, help="Where to save the photorealistic image")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint to load")
    args = parser.parse_args()

    cfg = Config()
    model = load_model(args.checkpoint, cfg)
    generate_photo_from_underpainting(model, args.input, args.output, cfg)


if __name__ == "__main__":
    main()

