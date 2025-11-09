"""Utility functions shared across the training and inference pipelines.

The functions defined here focus on pre-processing the input underpaintings and
handling generic tensor/image conversions.  The painting robot will supply
layered grayscale images that contain a surprising amount of structural
information.  By extracting explicit feature maps we give the learning model a
rich, physically meaningful conditioning signal that helps it respect the
artist's intent when synthesising a photorealistic output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from PIL import Image


def _ensure_three_channel(image: np.ndarray) -> np.ndarray:
    """Ensure that an image array has three channels.

    Dataset curators sometimes provide single-channel PNG files even when they
    conceptually represent RGB information.  Normalising the channel count keeps
    the downstream processing code straightforward and eliminates surprises at
    inference time.
    """

    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        # Drop the alpha channel if present; transparency is not useful here.
        return image[..., :3]
    return image


def _normalise_gray(image: np.ndarray) -> np.ndarray:
    """Normalise a grayscale image to the ``[0, 1]`` range as ``float32``."""

    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def analyze_underpainting(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Analyse a layered grayscale underpainting into feature maps.

    Parameters
    ----------
    image:
        Input image as a ``numpy.ndarray`` in ``H x W`` or ``H x W x C`` format.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing the following feature maps, each normalised to
        ``[0, 1]``:

        ``base_gray``
            The base grayscale layer.  Regardless of the original channel count
            the image is converted to a single-channel representation that keeps
            the tonal layering painted by the artist.

        ``edge_map``
            A Canny edge detector response that highlights sharp contours.

        ``blurred``
            A Gaussian-blurred version capturing large scale shapes.  The robot
            uses this to plan broad brush strokes before adding detail.

        ``gradient_magnitude``
            Magnitude of the image gradient computed via Sobel filters, useful
            for emphasising local contrast transitions.

        ``region_map``
            A simple segmentation obtained by clustering the grayscale values
            with k-means.  This approximates paint layers or material regions.
    """

    # Ensure we operate on an unsigned 8-bit image for OpenCV algorithms.
    original = _ensure_three_channel(image)
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    base_gray = _normalise_gray(gray)

    # Edge detection emphasises contours that the robot must preserve.
    edge_map = cv2.Canny((base_gray * 255).astype(np.uint8), threshold1=50, threshold2=150)
    edge_map = _normalise_gray(edge_map)

    # Gaussian blur to capture coarse structure.
    blurred = cv2.GaussianBlur(base_gray, ksize=(9, 9), sigmaX=2.0)

    # Gradient magnitude via Sobel filters.
    grad_x = cv2.Sobel(base_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(base_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_magnitude = np.clip(gradient_magnitude / (gradient_magnitude.max() + 1e-6), 0.0, 1.0)

    # Segment the grayscale tones into a few representative regions using k-means.
    # This provides a coarse notion of distinct paint layers.
    pixels = base_gray.reshape(-1, 1)
    pixels_8u = (pixels * 255).astype(np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    K = 4
    _compactness, labels, centers = cv2.kmeans(
        pixels_8u.astype(np.float32),
        K,
        None,
        criteria,
        attempts=4,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.squeeze() / 255.0
    region_map = centers[labels.flatten()].reshape(base_gray.shape)
    region_map = np.clip(region_map, 0.0, 1.0)

    return {
        "base_gray": base_gray.astype(np.float32),
        "edge_map": edge_map.astype(np.float32),
        "blurred": blurred.astype(np.float32),
        "gradient_magnitude": gradient_magnitude.astype(np.float32),
        "region_map": region_map.astype(np.float32),
    }


def stack_feature_maps(features: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack the feature dictionary returned by :func:`analyze_underpainting`.

    Returns an ``H x W x C`` numpy array suitable for conversion to a PyTorch
    tensor.  Channels are stacked in a deterministic order to ensure the model
    receives consistent conditioning.
    """

    ordered_keys = [
        "base_gray",
        "edge_map",
        "blurred",
        "gradient_magnitude",
        "region_map",
    ]
    channels = [features[key] for key in ordered_keys if key in features]
    return np.stack(channels, axis=0)


def load_image(path: Path | str, mode: str = "RGB") -> Image.Image:
    """Load an image from disk while enforcing a specific colour mode."""

    with Image.open(path) as img:
        return img.convert(mode)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor in ``[-1, 1]`` range to a :class:`PIL.Image`."""

    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0  # map to [0, 1]
    array = (tensor.numpy() * 255.0).astype(np.uint8)
    if array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))
    return Image.fromarray(array)


@dataclass
class ImageTriplet:
    """Simple container for (input, prediction, target) images.

    Used when saving visualisations during training.
    """

    underpaint: Image.Image
    prediction: Image.Image
    target: Image.Image


def save_image_triplet(triplet: ImageTriplet, destination: Path) -> None:
    """Save a side-by-side comparison image.

    This is a lightweight utility that helps the operators quickly assess
    whether the robot respects the desired structure after each training epoch.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    width, height = triplet.underpaint.size
    canvas = Image.new("RGB", (width * 3, height))
    canvas.paste(triplet.underpaint, (0, 0))
    canvas.paste(triplet.prediction, (width, 0))
    canvas.paste(triplet.target, (width * 2, 0))
    canvas.save(destination)


__all__ = [
    "analyze_underpainting",
    "stack_feature_maps",
    "load_image",
    "tensor_to_image",
    "ImageTriplet",
    "save_image_triplet",
]

