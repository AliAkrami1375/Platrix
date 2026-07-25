"""Pluggable plate-detection backends.

Imports are lazy so importing a leaf module (e.g. ``platrix.detection.yolo``)
never triggers the ``detection -> base -> core -> pipeline -> detection`` cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from platrix.config import Settings
    from platrix.detection.base import PlateDetector


def build_detector(settings: "Settings") -> "PlateDetector":
    """Factory selecting a detector backend from configuration.

    ``auto`` (the default) uses the YOLO detector when a trained model is
    present (``.onnx`` or ``.pt``), otherwise the weights-free contour detector.
    """
    name = settings.detector.lower()
    if name == "auto":
        yolo = settings.yolo_weights
        name = "yolo" if (yolo.exists() or yolo.with_suffix(".onnx").exists()) else "contour"

    if name == "contour":
        from platrix.detection.contour import ContourDetector

        return ContourDetector(settings)
    if name == "yolo":
        from platrix.detection.yolo import YoloDetector

        return YoloDetector(settings)
    raise ValueError(f"Unknown detector backend: {settings.detector!r}")


__all__ = ["build_detector"]
