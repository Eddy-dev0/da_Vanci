"""Utilities for safely importing PyTorch across different environments."""

from __future__ import annotations

import types
from textwrap import dedent

TORCH_IMPORT_ERROR: Exception | None = None
_TORCH_IMPORT_HELP = dedent(
    """
    PyTorch konnte nicht initialisiert werden. Auf Windows deutet dies meist auf
    fehlende Microsoft Visual C++ Redistributable Packages oder einen Konflikt mit
    GPU-Treibern hin. Installiere zunächst die aktuelle "Microsoft Visual C++
    2015-2022 Redistributable" (x64) von der Microsoft-Webseite. Anschliessend
    kannst du die CPU-Version von PyTorch direkt installieren:

        pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision

    Nachdem die Abhängigkeiten installiert wurden, starte dein Python-Environment
    neu und führe das Skript erneut aus.
    """
)


class _MissingTorchModule(types.ModuleType):
    """Proxy module that raises a descriptive error when accessed."""

    def __init__(self, name: str, import_error: Exception) -> None:
        super().__init__(name)
        self._import_error = import_error

    def __getattr__(self, name: str):  # pragma: no cover - defensive
        raise RuntimeError(_TORCH_IMPORT_HELP) from self._import_error


try:  # pragma: no cover - import side effects
    import torch as _torch  # type: ignore[import]
    import torch.nn as _nn  # type: ignore[import]
except OSError as exc:  # pragma: no cover - platform dependent
    TORCH_IMPORT_ERROR = exc
    torch = _MissingTorchModule("torch", exc)  # type: ignore[assignment]
    nn = _MissingTorchModule("torch.nn", exc)  # type: ignore[assignment]
else:
    torch = _torch  # type: ignore[assignment]
    nn = _nn  # type: ignore[assignment]


def ensure_torch_available() -> None:
    """Raise a descriptive error if PyTorch failed to load."""

    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(_TORCH_IMPORT_HELP) from TORCH_IMPORT_ERROR


__all__ = ["torch", "nn", "ensure_torch_available", "TORCH_IMPORT_ERROR"]
