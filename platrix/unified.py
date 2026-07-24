"""Single-model plate reader.

Runs ONE YOLO model that detects and classifies every plate character directly
in a full frame (via ONNX Runtime), then assembles the characters left→right
into a plate string. This is the "one unified model" path — no separate plate
detector and OCR.
"""

from __future__ import annotations

import json

import numpy as np

from platrix.config import Settings
from platrix.core.types import Frame, PlateDetection, PlateReading
from platrix.detection.yolo import _letterbox, _nms
from platrix.logging_conf import get_logger
from platrix.ocr.persian import format_iranian_plate

logger = get_logger(__name__)


class UnifiedReader:
    """Detect + recognize a plate with a single character-detection model."""

    name = "unified"

    def __init__(self, settings: Settings, infer_size: int = 416) -> None:
        self.settings = settings
        self.infer_size = infer_size
        self._session = None
        self._labels: list[str] = []
        self._input_name = "images"
        self._tried = False

    def available(self) -> bool:
        return self._load() is not None

    def warmup(self) -> None:
        self._load()

    def _load(self):
        if self._session is not None:
            return self._session
        if self._tried:
            return None
        self._tried = True
        weights = self.settings.plate_ocr_weights
        labels_path = weights.with_suffix(".labels.json")
        if not weights.exists() or not labels_path.exists():
            return None
        try:
            import onnxruntime as ort
        except ImportError:  # pragma: no cover
            return None
        logger.info("Loading unified plate model from %s", weights)
        self._session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._labels = json.loads(labels_path.read_text(encoding="utf-8"))
        return self._session

    # -- inference --------------------------------------------------------
    def _detect_chars(self, image: np.ndarray):
        h0, w0 = image.shape[:2]
        blob, ratio, (dw, dh) = _letterbox(image, self.infer_size)
        out = self._session.run(None, {self._input_name: blob[None].astype("float32")})[0]
        preds = np.squeeze(out, 0).T  # (N, 4+nc)
        boxes = preds[:, :4]
        cls_scores = preds[:, 4:]
        cls = cls_scores.argmax(1)
        conf = cls_scores.max(1)

        keep = conf > self.settings.detection_confidence
        boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
        if len(boxes) == 0:
            return []

        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / ratio

        idxs = _nms(xyxy, conf, iou_threshold=0.45)
        return [
            (float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3]),
             int(cls[i]), float(conf[i]))
            for i in idxs
        ]

    def read_frame(self, frame: Frame) -> list[PlateReading]:
        if self._load() is None:
            return []
        chars = self._detect_chars(frame.image)
        if len(chars) < 3:
            return []

        # Cluster characters into a plate by vertical proximity (one plate/frame).
        heights = [c[3] - c[1] for c in chars]
        median_h = float(np.median(heights))
        cys = np.array([(c[1] + c[3]) / 2 for c in chars])
        anchor = np.median(cys)
        row = [c for c, cy in zip(chars, cys) if abs(cy - anchor) < median_h * 0.9]
        if len(row) < 3:
            return []

        row.sort(key=lambda c: c[0])  # left → right
        text = "".join(self._labels[c[4]] for c in row if 0 <= c[4] < len(self._labels))
        confs = [c[5] for c in row]

        x1 = int(min(c[0] for c in row)); y1 = int(min(c[1] for c in row))
        x2 = int(max(c[2] for c in row)); y2 = int(max(c[3] for c in row))
        h_img, w_img = frame.image.shape[:2]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w_img), min(y2, h_img)
        det = PlateDetection(
            x=x1, y=y1, w=max(x2 - x1, 1), h=max(y2 - y1, 1),
            confidence=float(np.mean(confs)), crop=frame.image[y1:y2, x1:x2],
        )
        return [
            PlateReading(
                detection=det,
                text=text,
                text_fa=format_iranian_plate(text),
                ocr_confidence=float(np.mean(confs)),
                timestamp=frame.timestamp,
                source=frame.source,
            )
        ]
