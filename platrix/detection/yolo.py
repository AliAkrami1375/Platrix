"""YOLO plate detector (optional, high-accuracy backend).

Requires the ``ultralytics`` package and a trained plate-detection model at
``settings.yolo_weights``. Enable it with ``PLATRIX_DETECTOR=yolo``.
"""

from __future__ import annotations

from platrix.config import Settings
from platrix.core.types import Frame, PlateDetection
from platrix.detection.base import PlateDetector
from platrix.logging_conf import get_logger

logger = get_logger(__name__)


class YoloDetector(PlateDetector):
    """Ultralytics YOLO wrapper producing plate detections."""

    name = "yolo"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    def warmup(self) -> None:
        self._load()

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'yolo' detector requires the 'ultralytics' package. "
                "Install it with: pip install platrix[yolo]"
            ) from exc

        weights = self.settings.yolo_weights
        if not weights.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {weights}. Set PLATRIX_YOLO_WEIGHTS "
                "or place a trained model there."
            )
        logger.info("Loading YOLO weights from %s", weights)
        self._model = YOLO(str(weights))
        return self._model

    def detect(self, frame: Frame) -> list[PlateDetection]:
        model = self._load()
        preds = model.predict(
            frame.image,
            conf=self.settings.detection_confidence,
            verbose=False,
        )
        detections: list[PlateDetection] = []
        image = frame.image
        h_img, w_img = image.shape[:2]
        for result in preds:
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w_img), min(y2, h_img)
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                crop = image[y1:y2, x1:x2]
                detections.append(
                    PlateDetection(
                        x=x1,
                        y=y1,
                        w=w,
                        h=h,
                        confidence=float(box.conf[0]),
                        crop=crop,
                    )
                )
        return detections
