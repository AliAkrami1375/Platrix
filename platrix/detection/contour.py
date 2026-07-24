"""Classic (weights-free) Iranian plate detector.

This backend keeps the *structure* of the original project — edge/contour based
localization with no trained model — but hardens it for real photos with two
complementary candidate generators and a single, ranked shortlist so it stops
returning stray corners:

1. **Morphological text-region search** — the standard robust ANPR technique:
   black-hat to expose the plate's characters, a horizontal gradient, then a
   wide morphological close that fuses the characters into one plate-shaped
   blob. This is what actually finds plates on full-car photos.
2. **Edge/polygon search** — the original two-pass Canny + ``approxPolyDP``
   rectangle finder, kept for clean, well-cropped plate images.

Candidates from both are scored by how plate-like they are (aspect ratio +
fill), de-duplicated, thresholded, and only the best few are returned.
"""

from __future__ import annotations

import cv2
import numpy as np

from platrix.config import Settings
from platrix.core.types import Frame, PlateDetection
from platrix.detection.base import PlateDetector

# Iranian civilian plates are ~4.5:1. We accept a generous band around that.
_IDEAL_ASPECT = 4.3


class ContourDetector(PlateDetector):
    """Weights-free plate localizer tuned for Iranian plates."""

    name = "contour"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        self._square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # -- public -----------------------------------------------------------
    def detect(self, frame: Frame) -> list[PlateDetection]:
        image = frame.image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Normalize for lighting so thresholds behave across photos.
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        candidates: list[PlateDetection] = []
        candidates += self._morph_candidates(gray, image)
        candidates += self._edge_candidates(gray, image)

        if not candidates:
            return []

        kept = _non_max_suppression(candidates, iou_threshold=0.3)
        # Drop weak guesses so we don't report random rectangles/corners.
        strong = [d for d in kept if d.confidence >= 0.55]
        strong.sort(key=lambda d: d.confidence, reverse=True)
        # A frame almost never has more than 1–2 readable plates.
        return strong[:2]

    # -- candidate generators --------------------------------------------
    def _morph_candidates(self, gray: np.ndarray, image: np.ndarray) -> list[PlateDetection]:
        # Black-hat reveals dark characters on the bright plate background.
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self._rect_kernel)

        grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
        grad_x = np.absolute(grad_x)
        span = grad_x.max() - grad_x.min()
        if span <= 0:
            return []
        grad_x = 255 * (grad_x - grad_x.min()) / span
        grad_x = grad_x.astype("uint8")

        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, self._rect_kernel)
        thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

        results: list[PlateDetection] = []
        for c in contours:
            det = self._make_candidate(c, image, base=0.6)
            if det is not None:
                results.append(det)
        return results

    def _edge_candidates(self, gray: np.ndarray, image: np.ndarray) -> list[PlateDetection]:
        edged = cv2.Canny(gray, 30, 200)
        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        results: list[PlateDetection] = []
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approx) < 4:
                continue
            # A near-rectangular contour is a stronger plate signal.
            base = 0.62 if len(approx) == 4 else 0.5
            det = self._make_candidate(c, image, base=base)
            if det is not None:
                results.append(det)
        return results

    # -- shared candidate builder ----------------------------------------
    def _make_candidate(
        self, contour: np.ndarray, image: np.ndarray, base: float
    ) -> PlateDetection | None:
        area = cv2.contourArea(contour)
        if area < self.settings.min_plate_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0 or w < 60 or h < 15:
            return None

        aspect = w / h
        if not (self.settings.plate_aspect_min < aspect < self.settings.plate_aspect_max):
            return None

        crop = self._safe_crop(image, x, y, w, h)
        if crop.size == 0:
            return None

        # Score: closeness to the ideal aspect ratio + how full the box is.
        aspect_score = max(0.0, 1.0 - abs(aspect - _IDEAL_ASPECT) / _IDEAL_ASPECT)
        fill = float(min(area / (w * h), 1.0)) if w * h else 0.0
        confidence = round(min(base + 0.25 * aspect_score + 0.13 * fill, 0.99), 4)

        return PlateDetection(x=x, y=y, w=w, h=h, confidence=confidence, crop=crop)

    @staticmethod
    def _safe_crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        pad_x, pad_y = 6, 4
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
