"""Image analysis helpers used by PainterSlicer."""

from .analyzer import ImageAnalyzer, segment_image_into_layers
from .pipeline import (
    CalibrationProfile,
    PaletteResult,
    PaintingPipeline,
    PipelineResult,
    StrokeInstruction,
    enhance_layer,
)
from .layer_superres import compose_layers

__all__ = [
    "ImageAnalyzer",
    "segment_image_into_layers",
    "CalibrationProfile",
    "PaletteResult",
    "PaintingPipeline",
    "PipelineResult",
    "StrokeInstruction",
    "enhance_layer",
    "compose_layers",
]
