from typing import TypedDict, List


class PaintStroke(TypedDict):
    x: float
    y: float
    z: float
    pressure: float
    speed: float


class PaintStep(TypedDict):
    tool: str
    action: str
    layer: str
    path: List[PaintStroke]
