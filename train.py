"""Training script for the underpainting-to-photorealistic translation model."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from config import Config, ensure_directories
from dataset import create_dataloaders
from losses import CompositeLoss
from model import UnderpaintingToPhotoModel
from torch_utils import ensure_torch_available, nn, torch
from utils import ImageTriplet, save_image_triplet, tensor_to_image


ensure_torch_available()


def _prepare_perceptual_input(tensor: torch.Tensor) -> torch.Tensor:
    """Map tensors from ``[-1, 1]`` to VGG-friendly ``[0, 1]`` normalisation."""

    return (tensor + 1.0) / 2.0


def train(cfg: Config) -> None:
    ensure_directories(cfg)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    loaders = create_dataloaders(cfg)

    model = UnderpaintingToPhotoModel(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        feature_channels=cfg.feature_channels,
    ).to(device)

    criterion = CompositeLoss(cfg.loss_weights, cfg.perceptual_layers, use_gan=False)
    optimiser = Adam(model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    scheduler = StepLR(optimiser, step_size=cfg.lr_scheduler_step, gamma=cfg.lr_scheduler_gamma)

    global_step = 0
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        progress = tqdm(loaders["train"], desc=f"Epoch {epoch}/{cfg.num_epochs}", unit="batch")
        for conditioning, target in progress:
            conditioning = conditioning.to(device)
            target = target.to(device)

            optimiser.zero_grad(set_to_none=True)
            prediction = model(conditioning)

            loss = criterion(
                _prepare_perceptual_input(prediction),
                _prepare_perceptual_input(target),
            )
            loss.backward()
            optimiser.step()

            progress.set_postfix({"loss": loss.item()})
            global_step += 1

        scheduler.step()

        if epoch % 5 == 0:
            checkpoint_path = cfg.checkpoint_dir / f"epoch_{epoch:04d}.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "config": cfg.__dict__,
            }, checkpoint_path)

        if "val" in loaders:
            validate(model, loaders["val"], cfg, epoch, device)


def validate(model: nn.Module, loader, cfg: Config, epoch: int, device: torch.device) -> None:
    model.eval()
    example_dir = cfg.results_dir / f"epoch_{epoch:04d}"
    example_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, (conditioning, target) in enumerate(loader):
            conditioning = conditioning.to(device)
            target = target.to(device)
            prediction = model(conditioning)

            if idx < 3:
                underpaint_vis = tensor_to_image((conditioning[0, :1]).clamp(-1, 1))
                prediction_vis = tensor_to_image(prediction[0])
                target_vis = tensor_to_image(target[0])
                triplet = ImageTriplet(underpaint_vis, prediction_vis, target_vis)
                save_image_triplet(triplet, example_dir / f"sample_{idx:02d}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the underpainting-to-photo model")
    parser.add_argument("--config", type=Path, help="Optional path to a custom config module", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config is not None:
        namespace = runpy.run_path(args.config)
        cfg = namespace.get("cfg") or namespace.get("config") or namespace.get("CONFIG")
        if cfg is None:
            raise RuntimeError("Config file must define a variable named cfg/config/CONFIG containing a Config instance")
    else:
        cfg = Config()
    train(cfg)


if __name__ == "__main__":
    main()

