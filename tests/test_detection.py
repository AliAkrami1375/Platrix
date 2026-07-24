import numpy as np

from platrix.config import Settings
from platrix.core.types import Frame
from platrix.detection.contour import ContourDetector


def _plate_image() -> np.ndarray:
    """A synthetic scene with one bright plate-shaped rectangle."""
    img = np.full((300, 400, 3), 40, dtype=np.uint8)
    # A plate-like white rectangle, aspect ~3.3 (100x30).
    img[130:160, 150:250] = 235
    return img


def test_contour_detector_finds_plate_shape():
    detector = ContourDetector(Settings())
    frame = Frame(image=_plate_image())
    detections = detector.detect(frame)
    assert len(detections) >= 1
    best = max(detections, key=lambda d: d.confidence)
    aspect = best.w / best.h
    assert 2.2 < aspect < 5.0
    assert best.crop.size > 0


def test_contour_detector_empty_scene():
    detector = ContourDetector(Settings())
    frame = Frame(image=np.full((200, 200, 3), 50, dtype=np.uint8))
    assert detector.detect(frame) == []
