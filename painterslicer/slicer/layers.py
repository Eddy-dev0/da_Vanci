"""Shared layer models used by the slicer and preview pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


Point = Tuple[float, float]


@dataclass
class PaintLayer:
    """Container for a fully prepared paint layer.

    The dataclass mirrors the information the analyser generates and augments it
    with slicer specific metadata such as physical paths, execution parameters
    and brush configuration.  Instances travel all the way from colour analysis
    to preview rendering which keeps the required data structured instead of
    shipping loosely coupled dictionaries around.
    """

    color_rgb: Tuple[int, int, int]
    """Display colour of the layer in RGB 0..255."""

    mm_paths: List[List[Point]]
    """Tool paths converted to millimetres."""

    stage: Optional[str] = None
    technique: Optional[str] = None
    shading: Optional[str] = None
    coverage: float = 0.0
    order: int = 0
    label: Optional[int] = None

    # Enriched execution metadata -------------------------------------------------
    depth: float = 0.0
    opacity: float = 1.0
    style_key: Optional[str] = None
    mask: Optional[np.ndarray] = None
    brush: Dict[str, Any] = field(default_factory=dict)

    tool: Optional[str] = None
    pressure: float = 0.0
    z_down: float = 0.0
    z_up: float = 0.0
    passes: int = 1
    clean_interval: int = 0
    needs_cleaning: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_execution(self, **kwargs: Any) -> "PaintLayer":
        """Update execution attributes and return ``self`` for chaining."""

        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def sort_depth_key(self) -> Tuple[float, float, float]:
        """Return a stable sort key based on depth, order and coverage."""

        return (float(self.depth), float(self.order), -float(self.coverage))

