"""Brush rendering utilities used by the PainterSlicer preview and simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

ColorTuple = Tuple[int, int, int]
Point = Tuple[float, float]


@dataclass
class BrushTool:
    """Paint strokes by stamping a soft textured kernel onto an RGBA canvas.

    The brush works on numpy float32 canvases with shape ``(H, W, 4)`` where the
    last channel represents the alpha component. RGB channels are expected to be
    stored in non-premultiplied space in the range ``[0, 1]``. Every stamp blends
    the configured colour into the canvas using standard alpha compositing.

    Parameters
    ----------
    width_px:
        Diameter of the brush footprint in pixels.
    opacity:
        Maximum opacity applied per stamp (``0..1``). This works in conjunction
        with the generated kernel and the flow factor.
    edge_softness:
        Controls how soft the kernel edge is (``0 = hard edge``). Higher values
        blend the paint softly towards the border.
    flow:
        Controls how dense the paint is laid down along a stroke (``0..1``).
        Higher values place stamps closer together and increase opacity per
        stamp.
    spacing_px:
        Optional manual spacing override. When not provided the spacing is
        derived from ``width_px`` and ``flow``.
    """

    width_px: float
    opacity: float = 0.85
    edge_softness: float = 0.5
    flow: float = 0.7
    spacing_px: float | None = None

    def __post_init__(self) -> None:  # type: ignore[override]
        self.width_px = max(1.0, float(self.width_px))
        self.opacity = float(np.clip(self.opacity, 0.0, 1.0))
        self.edge_softness = float(np.clip(self.edge_softness, 0.0, 1.0))
        self.flow = float(np.clip(self.flow, 0.0, 1.0))
        self.spacing_px = (
            max(1.0, float(self.spacing_px))
            if self.spacing_px is not None
            else self._derive_spacing()
        )
        self._kernel = self._build_kernel()

    # ------------------------------------------------------------------ kernel
    def _derive_spacing(self) -> float:
        # Dense flow -> tight spacing, sparse flow -> larger spacing
        max_spacing = max(1.0, self.width_px * 0.6)
        min_spacing = max(1.0, self.width_px * 0.15)
        return max_spacing - (max_spacing - min_spacing) * self.flow

    def _build_kernel(self) -> np.ndarray:
        radius = self.width_px / 2.0
        size = int(math.ceil(self.width_px)) + 2
        yy, xx = np.mgrid[0:size, 0:size]
        center = (size - 1) / 2.0
        dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        norm = dist / max(radius, 1e-5)
        norm = np.clip(norm, 0.0, 1.0)

        if self.edge_softness <= 0.0:
            mask = (norm <= 1.0).astype(np.float32)
        else:
            softness = self.edge_softness
            falloff = 1.0 - norm
            falloff = np.clip(falloff, 0.0, 1.0)
            mask = falloff ** (1.0 + softness * 3.0)

        mask = mask.astype(np.float32)
        if mask.max() > 0:
            mask /= mask.max()
        return mask

    # ------------------------------------------------------------------- public
    @property
    def kernel(self) -> np.ndarray:
        return self._kernel

    def _stamp_precomputed(
        self, canvas: np.ndarray, position: Point, color: np.ndarray
    ) -> None:
        """Internal helper that assumes ``color`` is a normalised RGB array."""

        if canvas.ndim != 3 or canvas.shape[2] != 4:
            raise ValueError("Canvas must have shape (H, W, 4).")

        kernel = self._kernel
        k_h, k_w = kernel.shape
        radius_y = k_h / 2.0
        radius_x = k_w / 2.0

        cx, cy = position
        # Convert to pixel grid coordinates
        cx = float(cx)
        cy = float(cy)

        x0 = int(round(cx - radius_x))
        y0 = int(round(cy - radius_y))
        x1 = x0 + k_w
        y1 = y0 + k_h

        h, w, _ = canvas.shape
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            return  # fully outside

        clip_x0 = max(0, x0)
        clip_y0 = max(0, y0)
        clip_x1 = min(w, x1)
        clip_y1 = min(h, y1)

        kernel_x0 = clip_x0 - x0
        kernel_y0 = clip_y0 - y0
        kernel_x1 = kernel_x0 + (clip_x1 - clip_x0)
        kernel_y1 = kernel_y0 + (clip_y1 - clip_y0)

        kernel_slice = kernel[kernel_y0:kernel_y1, kernel_x0:kernel_x1]
        if kernel_slice.size == 0:
            return

        src_alpha = kernel_slice * self.opacity * max(self.flow, 1e-4)
        if src_alpha.max() <= 0.0:
            return

        dst_region = canvas[clip_y0:clip_y1, clip_x0:clip_x1]

        # Expand alpha to RGB channels
        src_alpha_exp = src_alpha[..., None]
        inv_alpha = 1.0 - src_alpha_exp

        dst_rgb = dst_region[..., :3]
        dst_alpha = dst_region[..., 3]

        blended_rgb = dst_rgb * inv_alpha + color * src_alpha_exp
        blended_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)

        dst_region[..., :3] = blended_rgb
        dst_region[..., 3] = blended_alpha

    def stamp(self, canvas: np.ndarray, position: Point, color_rgb: ColorTuple) -> None:
        """Stamp the brush kernel centred around ``position`` on the canvas."""

        color = np.asarray(color_rgb, dtype=np.float32) / 255.0
        self._stamp_precomputed(canvas, position, color)

    def render_path(
        self,
        canvas: np.ndarray,
        points: Sequence[Point],
        color_rgb: ColorTuple,
    ) -> None:
        """Render a full stroke across ``points`` on ``canvas``."""

        if not points:
            return

        points = list(points)
        color = np.asarray(color_rgb, dtype=np.float32) / 255.0

        self._stamp_precomputed(canvas, points[0], color)
        if len(points) == 1:
            return

        spacing = max(self.spacing_px, 1.0)

        canvas_h, canvas_w = canvas.shape[:2]
        radius = max(self.width_px / 2.0, 1.0)
        min_x = -radius
        max_x = canvas_w + radius
        min_y = -radius
        max_y = canvas_h + radius

        prev = np.asarray(points[0], dtype=np.float32)
        for current in points[1:]:
            current_arr = np.asarray(current, dtype=np.float32)
            seg_vec = current_arr - prev
            seg_len = float(np.hypot(seg_vec[0], seg_vec[1]))
            if seg_len <= 1e-6:
                prev = current_arr
                continue

            seg_min_x = float(min(prev[0], current_arr[0]))
            seg_max_x = float(max(prev[0], current_arr[0]))
            seg_min_y = float(min(prev[1], current_arr[1]))
            seg_max_y = float(max(prev[1], current_arr[1]))
            if (
                seg_max_x < min_x
                or seg_min_x > max_x
                or seg_max_y < min_y
                or seg_min_y > max_y
            ):
                prev = current_arr
                continue

            steps = max(int(math.ceil(seg_len / spacing)), 1)
            for step in range(1, steps + 1):
                t = step / steps
                pos = prev + seg_vec * t
                self._stamp_precomputed(
                    canvas, (float(pos[0]), float(pos[1])), color
                )
            prev = current_arr


def render_strokes(
    canvas: np.ndarray,
    strokes: Iterable[Tuple[BrushTool, Sequence[Point], ColorTuple]],
) -> None:
    """Utility to render multiple strokes onto ``canvas``."""

    for tool, points, color in strokes:
        tool.render_path(canvas, points, color)
