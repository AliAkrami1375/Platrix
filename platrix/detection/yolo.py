"""YOLO plate detector.

Runs a YOLOv8 plate-detection model. Inference goes through **ONNX Runtime**
(portable, no PyTorch/ultralytics needed at serve time) when an ``.onnx`` model
is present; otherwise it falls back to the ultralytics ``.pt`` runtime.

Enable with ``PLATRIX_DETECTOR=yolo``. Train a model with
``python scripts/train_detector.py``.
"""

from __future__ import annotations

import cv2
import numpy as np

from platrix.config import Settings
from platrix.core.types import Frame, PlateDetection
from platrix.detection.base import PlateDetector
from platrix.logging_conf import get_logger

logger = get_logger(__name__)


class YoloDetector(PlateDetector):
    """YOLOv8 plate detector (ONNX Runtime, with an ultralytics fallback)."""

    name = "yolo"

    def __init__(self, settings: Settings, infer_size: int = 416) -> None:
        self.settings = settings
        self.infer_size = infer_size
        self._session = None
        self._input_name = "images"
        self._ultra = None  # ultralytics fallback model
        self._tried = False
        # Secondary ONNX detector for hard frames (loaded lazily if present).
        self._fallback_session = None
        self._fallback_input = "images"

    def warmup(self) -> None:
        self._load()

    # -- model loading ----------------------------------------------------
    def _onnx_path(self):
        onnx = self.settings.yolo_weights.with_suffix(".onnx")
        return onnx if onnx.exists() else None

    def _load(self):
        if self._session is not None or self._ultra is not None:
            return
        if self._tried:
            return
        self._tried = True

        onnx = self._onnx_path()
        if onnx is not None:
            try:
                import onnxruntime as ort

                self._session = ort.InferenceSession(
                    str(onnx), providers=["CPUExecutionProvider"]
                )
                self._input_name = self._session.get_inputs()[0].name
                logger.info("Loaded YOLO ONNX detector from %s", onnx)
                self._load_fallback()
                return
            except ImportError:
                logger.warning("onnxruntime missing; trying ultralytics .pt")

        # Fallback: ultralytics .pt
        weights = self.settings.yolo_weights
        if not weights.exists():
            raise FileNotFoundError(
                f"YOLO weights not found ({weights} / .onnx). "
                "Train one with: python scripts/train_detector.py"
            )
        from ultralytics import YOLO

        self._ultra = YOLO(str(weights))
        logger.info("Loaded YOLO (ultralytics) detector from %s", weights)

    def _load_fallback(self):
        """Load the optional secondary ONNX detector, if the file exists."""
        path = getattr(self.settings, "yolo_fallback_weights", None)
        if not path or not path.exists():
            return
        try:
            import onnxruntime as ort

            self._fallback_session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self._fallback_input = self._fallback_session.get_inputs()[0].name
            logger.info("Loaded YOLO fallback detector from %s", path)
        except Exception as exc:  # pragma: no cover - fallback is best-effort
            logger.warning("Could not load YOLO fallback detector: %s", exc)

    # -- inference --------------------------------------------------------
    def detect(self, frame: Frame) -> list[PlateDetection]:
        self._load()
        if self._session is not None:
            dets = self._detect_onnx(frame, self._session, self._input_name, self.infer_size)
            # Only pay for the second pass when the primary finds nothing.
            if not dets and self._fallback_session is not None:
                dets = self._detect_onnx(
                    frame,
                    self._fallback_session,
                    self._fallback_input,
                    self.settings.yolo_fallback_size,
                )
            return dets
        return self._detect_ultra(frame)

    def _detect_onnx(self, frame: Frame, session, input_name, size) -> list[PlateDetection]:
        image = frame.image
        h0, w0 = image.shape[:2]
        blob, ratio, (dw, dh) = _letterbox(image, size)
        inp = blob[None].astype("float32")
        out = session.run(None, {input_name: inp})[0]

        # YOLOv8 output: (1, 4+nc, N) -> (N, 4+nc). One class here.
        preds = np.squeeze(out, 0).T  # (N, 5)
        boxes_xywh = preds[:, :4]
        scores = preds[:, 4:].max(axis=1)

        keep = scores > self.settings.detection_confidence
        boxes_xywh, scores = boxes_xywh[keep], scores[keep]
        if len(boxes_xywh) == 0:
            return []

        # cxcywh (letterbox coords) -> xyxy in original image
        xyxy = np.empty_like(boxes_xywh)
        xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / ratio

        idxs = _nms(xyxy, scores, iou_threshold=0.45)
        detections: list[PlateDetection] = []
        for i in idxs:
            x1, y1, x2, y2 = xyxy[i]
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2, y2 = min(int(x2), w0), min(int(y2), h0)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            detections.append(
                PlateDetection(
                    x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                    confidence=float(scores[i]), crop=image[y1:y2, x1:x2],
                )
            )
        return detections

    def _detect_ultra(self, frame: Frame) -> list[PlateDetection]:
        preds = self._ultra.predict(
            frame.image, conf=self.settings.detection_confidence, verbose=False
        )
        image = frame.image
        h0, w0 = image.shape[:2]
        detections: list[PlateDetection] = []
        for result in preds:
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w0), min(y2, h0)
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                detections.append(
                    PlateDetection(
                        x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                        confidence=float(box.conf[0]), crop=image[y1:y2, x1:x2],
                    )
                )
        return detections


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize with unchanged aspect ratio using padding; return CHW RGB blob."""
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    nh, nw = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype="uint8")
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    canvas[top : top + nh, left : left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return np.transpose(rgb, (2, 0, 1)), ratio, (float(left), float(top))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep
