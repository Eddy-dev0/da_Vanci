"""Simple in-memory painting simulator using the brush engine."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from painterslicer.config.config_loader import load_brush_presets, load_machine_config
from painterslicer.utils.brush_tool import BrushTool

Point = Tuple[float, float]
ColorTuple = Tuple[int, int, int]


class Simulator:
    """Incrementally render paint strokes onto a virtual canvas."""

    def __init__(
        self,
        canvas_size: Tuple[int, int] | None = None,
    ) -> None:
        machine_cfg = load_machine_config()
        work = machine_cfg.get("work_area_mm", {"x": 300.0, "y": 200.0})
        self.work_area_mm = (float(work.get("x", 300.0)), float(work.get("y", 200.0)))

        if canvas_size is None:
            canvas_size = (1024, 768)
        self.canvas_size = canvas_size

        self._brush_presets = load_brush_presets()
        self._brush_overrides: Dict[str, Dict[str, float]] = {}
        self._brush_cache: Dict[str, Tuple[Dict[str, float], BrushTool]] = {}

        self._canvas = self._create_blank_canvas()

    # ------------------------------------------------------------------ helpers
    def _create_blank_canvas(self) -> np.ndarray:
        w, h = self.canvas_size
        return np.zeros((h, w, 4), dtype=np.float32)

    def reset(self, canvas_size: Optional[Tuple[int, int]] = None) -> None:
        if canvas_size is not None:
            self.canvas_size = canvas_size
        self._canvas = self._create_blank_canvas()
        self._brush_cache.clear()

    def apply_brush_overrides(self, overrides: Optional[Dict[str, Dict[str, float]]]) -> None:
        self._brush_overrides = {k: dict(v) for k, v in (overrides or {}).items()}
        self._brush_cache.clear()

    def _mm_to_px(self, point: Point) -> Point:
        w, h = self.canvas_size
        work_w, work_h = self.work_area_mm
        x_mm, y_mm = point
        x_px = (x_mm / work_w) * w
        y_px = (y_mm / work_h) * h
        return (x_px, y_px)

    def _get_brush_params(self, tool_name: str) -> Dict[str, float]:
        base = dict(self._brush_presets.get(tool_name, {}))
        override = self._brush_overrides.get(tool_name, {})
        base.update({k: v for k, v in override.items() if v is not None})
        return base

    def _get_brush_tool(self, tool_name: str) -> BrushTool:
        params = self._get_brush_params(tool_name)
        cache_entry = self._brush_cache.get(tool_name)
        if cache_entry and cache_entry[0] == params:
            return cache_entry[1]

        brush = BrushTool(
            width_px=params.get("width_px", 24.0),
            opacity=params.get("opacity", 0.85),
            edge_softness=params.get("edge_softness", 0.5),
            flow=params.get("flow", 0.7),
            spacing_px=params.get("spacing_px"),
        )
        self._brush_cache[tool_name] = (params, brush)
        return brush

    # ---------------------------------------------------------------- rendering
    @property
    def canvas(self) -> np.ndarray:
        return self._canvas

    def render_step(self, paint_step: Dict[str, object]) -> np.ndarray:
        """Render a single step of the paint plan onto the canvas."""

        points_mm: Sequence[Point] = paint_step.get("points", [])  # type: ignore[assignment]
        if not points_mm:
            return self._canvas

        color: ColorTuple = tuple(paint_step.get("color_rgb", (255, 255, 255)))  # type: ignore[arg-type]
        tool_name = str(paint_step.get("tool", "medium_brush"))

        tool = self._get_brush_tool(tool_name)
        points_px = [self._mm_to_px(pt) for pt in points_mm]
        tool.render_path(self._canvas, points_px, color)
        return self._canvas

    def render_sequence(self, steps: Iterable[Dict[str, object]]) -> np.ndarray:
        for step in steps:
            self.render_step(step)
        return self._canvas

    def as_uint8(self) -> np.ndarray:
        """Return the current canvas as 8-bit RGBA image."""

        arr = np.clip(self._canvas, 0.0, 1.0)
        return (arr * 255).astype(np.uint8)
