"""Classic OpenCV contour/edge based plate detector.

This backend requires **no trained weights**, which makes Platrix usable out of
the box. It locates rectangular, plate-shaped contours using Canny edges and a
polygon + aspect-ratio filter. It is a hardened, streaming-friendly reworking
of the original proof-of-concept detector.
"""

from __future__ import annotations

import cv2
import numpy as np

from platrix.config import Settings
from platrix.core.types import Frame, PlateDetection
from platrix.detection.base import PlateDetector


class ContourDetector(PlateDetector):
    """Edge + contour based detector for well-framed plates."""

    name = "contour"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def detect(self, frame: Frame) -> list[PlateDetection]:
        image = frame.image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 10, 100)

        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        results: list[PlateDetection] = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) < 4:
                continue

            area = cv2.contourArea(contour)
            if area < self.settings.min_plate_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue
            aspect = w / h
            if not (self.settings.plate_aspect_min < aspect < self.settings.plate_aspect_max):
                continue

            crop = self._safe_crop(image, x, y, w, h)
            if crop.size == 0:
                continue

            # A tighter, more rectangular contour scores higher.
            rectangularity = float(min(area / (w * h), 1.0)) if w * h else 0.0
            confidence = round(0.5 + 0.5 * rectangularity, 4)

            results.append(
                PlateDetection(x=x, y=y, w=w, h=h, confidence=confidence, crop=crop)
            )

        # Keep the strongest, non-overlapping candidates.
        return _non_max_suppression(results, iou_threshold=0.3)

    @staticmethod
    def _safe_crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        pad_x, pad_y = 8, 6
        h_img, w_img = image.shape[:2]
        x0 = max(x - pad_x, 0)
        y0 = max(y - pad_y, 0)
        x1 = min(x + w + pad_x, w_img)
        y1 = min(y + h + pad_y, h_img)
        return image[y0:y1, x0:x1]


def _iou(a: PlateDetection, b: PlateDetection) -> float:
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.w, a.y + a.h
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.w, b.y + b.h
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(ix1 - ix0, 0), max(iy1 - iy0, 0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def _non_max_suppression(
    dets: list[PlateDetection], iou_threshold: float
) -> list[PlateDetection]:
    kept: list[PlateDetection] = []
    for det in sorted(dets, key=lambda d: d.confidence, reverse=True):
        if all(_iou(det, k) < iou_threshold for k in kept):
            kept.append(det)
    return kept
