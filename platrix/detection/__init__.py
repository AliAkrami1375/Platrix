"""Pluggable plate-detection backends."""

from __future__ import annotations

from platrix.config import Settings
from platrix.detection.base import PlateDetector
from platrix.detection.contour import ContourDetector


def build_detector(settings: Settings) -> PlateDetector:
    """Factory selecting a detector backend from configuration.

    ``auto`` (the default) uses the YOLO detector when a trained model is
    present (``.onnx`` or ``.pt``), otherwise the weights-free contour detector.
    """
    name = settings.detector.lower()
    if name == "auto":
        yolo = settings.yolo_weights
        name = "yolo" if (yolo.exists() or yolo.with_suffix(".onnx").exists()) else "contour"

    if name == "contour":
        return ContourDetector(settings)
    if name == "yolo":
        from platrix.detection.yolo import YoloDetector

        return YoloDetector(settings)
    raise ValueError(f"Unknown detector backend: {settings.detector!r}")


__all__ = ["PlateDetector", "ContourDetector", "build_detector"]
