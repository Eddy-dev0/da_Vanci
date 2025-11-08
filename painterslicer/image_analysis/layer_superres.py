"""Utilities for composing super-resolved painting layers."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def _ensure_rgba(layer: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(layer, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("Expected layer with shape (H, W, C)")
    if arr.shape[2] == 4:
        rgb = np.clip(arr[..., :3], 0.0, 1.0)
        alpha = np.clip(arr[..., 3:4], 0.0, 1.0)
    elif arr.shape[2] == 3:
        rgb = np.clip(arr, 0.0, 1.0)
        alpha = np.ones_like(rgb[..., :1], dtype=np.float32)
    else:
        raise ValueError("Layer must have 3 (RGB) or 4 (RGBA) channels")
    return rgb.astype(np.float32), alpha.astype(np.float32)


def compose_layers(
    background: np.ndarray,
    midground: np.ndarray,
    foreground: np.ndarray,
    *,
    feather_radius: int = 3,
) -> np.ndarray:
    """Alpha-composite the provided layers from back to front.

    ``feather_radius`` controls an optional Gaussian blur that is applied to each
    layer's alpha channel prior to blending.  This helps hiding small mask seams
    after super-resolution.
    """

    bg_rgb, bg_alpha = _ensure_rgba(background)
    mid_rgb, mid_alpha = _ensure_rgba(midground)
    fg_rgb, fg_alpha = _ensure_rgba(foreground)

    shapes = {bg_rgb.shape[:2], mid_rgb.shape[:2], fg_rgb.shape[:2]}
    if len(shapes) != 1:
        raise ValueError("All layers must share the same spatial resolution")

    def maybe_feather(alpha: np.ndarray) -> np.ndarray:
        if feather_radius and feather_radius > 0 and alpha.any():
            return cv2.GaussianBlur(
                alpha,
                ksize=(0, 0),
                sigmaX=float(feather_radius),
                sigmaY=float(feather_radius),
                borderType=cv2.BORDER_REFLECT101,
            )
        return alpha

    bg_alpha = np.clip(maybe_feather(bg_alpha), 0.0, 1.0)
    mid_alpha = np.clip(maybe_feather(mid_alpha), 0.0, 1.0)
    fg_alpha = np.clip(maybe_feather(fg_alpha), 0.0, 1.0)

    comp_rgb = bg_rgb * bg_alpha
    comp_alpha = bg_alpha

    def blend(top_rgb: np.ndarray, top_alpha: np.ndarray) -> None:
        nonlocal comp_rgb, comp_alpha
        comp_rgb = top_rgb * top_alpha + comp_rgb * (1.0 - top_alpha)
        comp_alpha = top_alpha + comp_alpha * (1.0 - top_alpha)

    blend(mid_rgb, mid_alpha)
    blend(fg_rgb, fg_alpha)

    return np.concatenate([comp_rgb, comp_alpha], axis=-1).astype(np.float32)
