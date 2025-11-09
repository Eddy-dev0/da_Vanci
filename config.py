"""Application configuration module.

This file defines a :class:`Config` object that centralises every tunable
hyperparameter and all file-system paths that the training and inference
pipelines rely on.  The painting robot that will execute this project is
expected to have very limited options for editing the code directly, so the
configuration is written in plain Python with extensive comments explaining the
purpose of each entry.

The configuration values are intentionally conservative defaults that can be
adapted to a wide range of hardware setups.  When integrating the project with
new datasets or more powerful GPUs, the operator simply updates the relevant
fields.  Every module in the codebase accepts a :class:`Config` instance so the
entire system can be orchestrated from a single location.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class Config:
    """Container for all hyperparameters and directory locations.

    The defaults have been chosen to work out-of-the-box on a typical developer
    workstation while remaining easy to scale up.  Any script can import this
    class and instantiate it without arguments to obtain the canonical project
    configuration.  When running experiments it is common to create a modified
    copy (``cfg = Config(); cfg.batch_size = 8``) rather than editing the file.
    """

    # ------------------------------------------------------------------
    # File-system layout.
    # ------------------------------------------------------------------
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_root: Path = field(default_factory=lambda: Path("data"))
    train_underpaint_dir: Path = field(init=False)
    train_target_dir: Path = field(init=False)
    val_underpaint_dir: Path = field(init=False)
    val_target_dir: Path = field(init=False)
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    results_dir: Path = field(default_factory=lambda: Path("results"))

    # ------------------------------------------------------------------
    # Data processing parameters.
    # ------------------------------------------------------------------
    image_size: Tuple[int, int] = (512, 512)  # (height, width)
    num_workers: int = 4
    augment_horizontal_flip: bool = True
    augment_vertical_flip: bool = False
    random_rotation_degrees: float = 2.0

    # ------------------------------------------------------------------
    # Optimisation parameters.
    # ------------------------------------------------------------------
    batch_size: int = 2
    learning_rate: float = 2e-4
    beta1: float = 0.5  # Adam beta1
    beta2: float = 0.999  # Adam beta2
    num_epochs: int = 100
    lr_scheduler_step: int = 50
    lr_scheduler_gamma: float = 0.5

    # ------------------------------------------------------------------
    # Model configuration.
    # ------------------------------------------------------------------
    in_channels: int = 5  # base gray, edges, blur, gradient, segmentation
    out_channels: int = 3
    feature_channels: Tuple[int, ...] = (64, 128, 256, 512, 512)

    # ------------------------------------------------------------------
    # Loss weighting.
    # ------------------------------------------------------------------
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "reconstruction": 1.0,
            "perceptual": 0.1,
            "edge": 0.05,
            "adversarial": 0.0,
        }
    )

    perceptual_layers: Tuple[str, ...] = (
        "relu1_2",
        "relu2_2",
        "relu3_3",
    )

    device: str = "cuda"

    def __post_init__(self) -> None:
        """Derive directory paths that depend on :attr:`data_root`.

        Keeping the logic here makes it impossible to forget to update the
        validation paths when moving the dataset to a new location.
        """

        self.train_underpaint_dir = self.data_root / "train" / "underpaint"
        self.train_target_dir = self.data_root / "train" / "target"
        self.val_underpaint_dir = self.data_root / "val" / "underpaint"
        self.val_target_dir = self.data_root / "val" / "target"


def ensure_directories(cfg: Config) -> None:
    """Create (if necessary) all directories used for checkpoints/results.

    The physical robot system that will run inference might not have permission
    to create directories at runtime.  By exposing this helper we can call it
    during deployment so that the directory structure is guaranteed to exist.
    """

    for path in [
        cfg.checkpoint_dir,
        cfg.results_dir,
        cfg.train_underpaint_dir,
        cfg.train_target_dir,
        cfg.val_underpaint_dir,
        cfg.val_target_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


__all__ = ["Config", "ensure_directories"]

