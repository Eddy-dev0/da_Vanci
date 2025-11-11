"""Dataset utilities for the underpainting to photorealistic translation task."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import Config
from utils import analyze_underpainting, load_image, stack_feature_maps
from torch_utils import ensure_torch_available, torch


ensure_torch_available()


class UnderpaintingPhotoDataset(Dataset):
    """PyTorch dataset that returns (conditioning_tensor, target_tensor) pairs.

    The dataset reads aligned underpainting/target photographs from disk,
    computes the structural feature maps required by the model and returns them
    as a multi-channel tensor.  All tensors are normalised to the ``[-1, 1]``
    range which is convenient for generator-style neural networks.
    """

    def __init__(
        self,
        underpaint_dir: Path | str,
        target_dir: Path | str,
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.underpaint_dir = Path(underpaint_dir)
        self.target_dir = Path(target_dir)
        self.transform = transform
        self.config = config or Config()

        self.pairs = self._scan_pairs()
        if not self.pairs:
            raise RuntimeError(self._format_missing_pairs_error())

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _scan_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching filenames between the underpainting and target folders."""

        if not self.underpaint_dir.exists():
            raise FileNotFoundError(f"Underpainting directory not found: {self.underpaint_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")

        underpaint_files = sorted(
            [p for p in self.underpaint_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        target_files = sorted(
            [p for p in self.target_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        pairs: List[Tuple[Path, Path]] = []
        for underpaint_path in underpaint_files:
            target_path = self.target_dir / underpaint_path.name
            if target_path.exists():
                pairs.append((underpaint_path, target_path))
        return pairs

    def _format_missing_pairs_error(self) -> str:
        """Produce a helpful error message when the dataset is empty or misconfigured."""

        def _list_examples(paths: List[Path]) -> str:
            preview = ", ".join(p.name for p in paths[:3])
            if not preview:
                return "(no images found)"
            remaining = max(len(paths) - 3, 0)
            return preview + (" ..." if remaining > 0 else "")

        underpaint_files = sorted(
            [p for p in self.underpaint_dir.glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        target_files = sorted(
            [p for p in self.target_dir.glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )

        return (
            "No paired images were detected.\n"
            f"  Underpaint directory: {self.underpaint_dir} ({len(underpaint_files)} files)\n"
            f"    Examples: {_list_examples(underpaint_files)}\n"
            f"  Target directory: {self.target_dir} ({len(target_files)} files)\n"
            f"    Examples: {_list_examples(target_files)}\n"
            "Ensure both folders contain images with matching filenames (e.g. `0001.png`)."
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.pairs)

    def _apply_joint_transform(self, underpaint: Image.Image, target: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Apply random but *aligned* augmentations to both images."""

        if self.transform:
            return self.transform(underpaint, target)

        # Default set of light augmentations that preserve alignment.
        width, height = underpaint.size
        if self.config.augment_horizontal_flip and random.random() < 0.5:
            underpaint = underpaint.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
        if self.config.augment_vertical_flip and random.random() < 0.5:
            underpaint = underpaint.transpose(Image.FLIP_TOP_BOTTOM)
            target = target.transpose(Image.FLIP_TOP_BOTTOM)

        if self.config.random_rotation_degrees > 0:
            angle = random.uniform(-self.config.random_rotation_degrees, self.config.random_rotation_degrees)
            underpaint = underpaint.rotate(angle, resample=Image.BICUBIC)
            target = target.rotate(angle, resample=Image.BICUBIC)

        size = (self.config.image_size[1], self.config.image_size[0])
        underpaint = underpaint.resize(size, Image.BICUBIC)
        target = target.resize(size, Image.BICUBIC)
        return underpaint, target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        underpaint_path, target_path = self.pairs[idx]

        underpaint_img = load_image(underpaint_path, mode="RGB")
        target_img = load_image(target_path, mode="RGB")

        underpaint_img, target_img = self._apply_joint_transform(underpaint_img, target_img)

        # Convert PIL image to numpy for analysis.
        underpaint_array = np.array(underpaint_img)
        features = analyze_underpainting(underpaint_array)
        feature_stack = stack_feature_maps(features)

        conditioning_tensor = torch.from_numpy(feature_stack).float()
        conditioning_tensor = conditioning_tensor * 2.0 - 1.0
        target_tensor = self.to_tensor(target_img)
        return conditioning_tensor, target_tensor


def create_dataloaders(cfg: Config) -> Dict[str, DataLoader]:
    """Create training and validation data loaders based on :class:`Config`."""

    dataset_kwargs = {
        "transform": None,
        "config": cfg,
    }

    loaders: Dict[str, DataLoader] = {}

    if cfg.train_underpaint_dir.exists() and cfg.train_target_dir.exists():
        train_dataset = UnderpaintingPhotoDataset(
            cfg.train_underpaint_dir,
            cfg.train_target_dir,
            **dataset_kwargs,
        )
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    if cfg.val_underpaint_dir.exists() and cfg.val_target_dir.exists():
        val_dataset = UnderpaintingPhotoDataset(
            cfg.val_underpaint_dir,
            cfg.val_target_dir,
            **dataset_kwargs,
        )
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    if not loaders:
        raise RuntimeError("No datasets available. Ensure train/val directories exist and contain images.")

    return loaders


__all__ = ["UnderpaintingPhotoDataset", "create_dataloaders"]

