"""Pluggable OCR backends for reading plate crops."""

from __future__ import annotations

from platrix.config import Settings
from platrix.ocr.base import PlateOCR, NullOCR


def build_ocr(settings: Settings) -> PlateOCR:
    """Factory selecting an OCR backend from configuration."""
    name = settings.ocr.lower()
    if name in ("none", "off", ""):
        return NullOCR()
    if name == "cnn":
        from platrix.ocr.cnn import CnnOCR

        return CnnOCR(settings)
    raise ValueError(f"Unknown OCR backend: {settings.ocr!r}")


__all__ = ["PlateOCR", "NullOCR", "build_ocr"]
