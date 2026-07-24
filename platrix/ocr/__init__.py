"""Pluggable OCR backends for reading plate crops.

Imports here are intentionally lazy: importing a leaf module such as
``platrix.ocr.segmentation`` must not drag in the detection pipeline (which
would create an import cycle ``ocr -> base -> core -> pipeline -> ocr``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from platrix.config import Settings
    from platrix.ocr.base import PlateOCR


def build_ocr(settings: "Settings") -> "PlateOCR":
    """Factory selecting an OCR backend from configuration."""
    from platrix.ocr.base import NullOCR

    name = settings.ocr.lower()
    if name in ("none", "off", ""):
        return NullOCR()
    if name == "onnx":
        from platrix.ocr.onnx_ocr import OnnxOCR

        return OnnxOCR(settings)
    if name == "cnn":
        from platrix.ocr.cnn import CnnOCR

        return CnnOCR(settings)
    raise ValueError(f"Unknown OCR backend: {settings.ocr!r}")


__all__ = ["build_ocr"]
