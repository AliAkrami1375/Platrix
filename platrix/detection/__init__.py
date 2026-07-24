"""Pluggable plate-detection backends."""

from __future__ import annotations

from platrix.config import Settings
from platrix.detection.base import PlateDetector
from platrix.detection.contour import ContourDetector


def build_detector(settings: Settings) -> PlateDetector:
    """Factory selecting a detector backend from configuration."""
    name = settings.detector.lower()
    if name == "contour":
        return ContourDetector(settings)
    if name == "yolo":
        # Imported lazily so the heavy ultralytics dependency is optional.
        from platrix.detection.yolo import YoloDetector

        return YoloDetector(settings)
    raise ValueError(f"Unknown detector backend: {settings.detector!r}")


__all__ = ["PlateDetector", "ContourDetector", "build_detector"]
