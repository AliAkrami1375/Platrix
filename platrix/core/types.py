"""Immutable domain types shared across the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np


@dataclass(slots=True)
class Frame:
    """A single decoded frame from any source."""

    image: np.ndarray  # BGR, HxWx3
    index: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "unknown"

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass(slots=True)
class PlateDetection:
    """A localized plate region within a frame (before OCR)."""

    x: int
    y: int
    w: int
    h: int
    confidence: float
    crop: np.ndarray  # BGR crop of the plate region

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def area(self) -> int:
        return int(self.w * self.h)


@dataclass(slots=True)
class PlateReading:
    """A fully recognized plate: detection + OCR result."""

    detection: PlateDetection
    text: str  # normalized plate string ("" if unreadable)
    text_fa: str  # human-readable Persian representation
    ocr_confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "unknown"
    snapshot_path: Optional[str] = None

    @property
    def score(self) -> float:
        """Combined detection + OCR confidence in ``[0, 1]``."""
        return round(self.detection.confidence * max(self.ocr_confidence, 0.01), 4)

    def to_dict(self) -> dict:
        x, y, w, h = self.detection.bbox
        return {
            "text": self.text,
            "text_fa": self.text_fa,
            "detection_confidence": round(self.detection.confidence, 4),
            "ocr_confidence": round(self.ocr_confidence, 4),
            "score": self.score,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "snapshot_path": self.snapshot_path,
        }
