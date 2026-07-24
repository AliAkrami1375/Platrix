"""The recognition pipeline: detect → read → dedupe → annotate."""

from __future__ import annotations

import time

import cv2
import numpy as np

from platrix.config import Settings
from platrix.core.types import Frame, PlateReading
from platrix.detection import build_detector
from platrix.logging_conf import get_logger
from platrix.ocr import build_ocr
from platrix.ocr.persian import format_iranian_plate

logger = get_logger(__name__)


class RecognitionPipeline:
    """Orchestrates detection and OCR for a single stream of frames.

    The pipeline is stateful only for de-duplication (it remembers recently
    seen plates so the same car isn't logged on every frame). It is *not*
    thread-safe; use one instance per source worker.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.detector = build_detector(settings)
        self.ocr = build_ocr(settings)
        self._recent: dict[str, float] = {}

    def warmup(self) -> None:
        self.detector.warmup()
        self.ocr.warmup()
        logger.info(
            "Pipeline ready — detector=%s ocr=%s",
            self.detector.name,
            self.ocr.name,
        )

    def process(self, frame: Frame) -> list[PlateReading]:
        """Detect and read every plate in *frame*."""
        readings: list[PlateReading] = []
        for detection in self.detector.detect(frame):
            text, ocr_conf = self.ocr.read(detection)
            text_fa = format_iranian_plate(text) if text else ""
            readings.append(
                PlateReading(
                    detection=detection,
                    text=text,
                    text_fa=text_fa,
                    ocr_confidence=ocr_conf,
                    timestamp=frame.timestamp,
                    source=frame.source,
                )
            )
        return readings

    def is_duplicate(self, reading: PlateReading) -> bool:
        """True if this plate text was logged within the dedupe window."""
        if not reading.text:
            return False
        now = time.monotonic()
        window = self.settings.dedupe_seconds
        # Drop expired entries.
        self._recent = {k: t for k, t in self._recent.items() if now - t < window}
        if reading.text in self._recent:
            return True
        self._recent[reading.text] = now
        return False


def annotate(frame_image: np.ndarray, readings: list[PlateReading]) -> np.ndarray:
    """Draw bounding boxes and plate text onto a copy of the frame."""
    canvas = frame_image.copy()
    for reading in readings:
        x, y, w, h = reading.detection.bbox
        color = (0, 200, 0) if reading.text else (0, 165, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        label = reading.text or "PLATE"
        conf = f"{reading.score:.0%}"
        text = f"{label} {conf}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(
            canvas,
            text,
            (x + 3, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    return canvas
