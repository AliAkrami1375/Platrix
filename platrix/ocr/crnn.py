"""Segmentation-free CRNN OCR backend (ONNX Runtime).

Reads the **whole plate at once** with a CRNN + CTC model — no character
segmentation, which removes the main source of error on real photos (over/under
splitting). The plate crop is enhanced, resized to a fixed size, run through the
ONNX model, and CTC-greedy-decoded into a plate string.

If the model is missing, the backend degrades gracefully (returns "").
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from platrix.config import Settings
from platrix.core.types import PlateDetection
from platrix.logging_conf import get_logger
from platrix.ocr.base import PlateOCR
from platrix.preprocessing import prep_crnn

logger = get_logger(__name__)

IMG_H, IMG_W = 32, 128  # must match scripts/train_crnn.py


class CrnnOCR(PlateOCR):
    """Whole-plate CRNN reader run through ONNX Runtime."""

    name = "crnn"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = None
        self._labels: list[str] = []
        self._blank = 0
        self._input_name = "input"
        self._tried = False

    def warmup(self) -> None:
        self._load()

    def _load(self):
        if self._session is not None or self._tried:
            return self._session
        self._tried = True
        weights = self.settings.crnn_weights
        labels_path = weights.with_suffix(".labels.json")
        if not weights.exists() or not labels_path.exists():
            logger.warning(
                "CRNN model not found at %s — OCR disabled. Train it with "
                "python scripts/train_crnn.py",
                weights,
            )
            return None
        try:
            import onnxruntime as ort
        except ImportError:  # pragma: no cover
            logger.warning("onnxruntime not installed — OCR disabled.")
            return None

        logger.info("Loading CRNN OCR model from %s", weights)
        self._session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._labels = json.loads(labels_path.read_text(encoding="utf-8"))
        self._blank = len(self._labels)  # CTC blank is the last index
        return self._session

    def read(self, detection: PlateDetection) -> tuple[str, float]:
        session = self._load()
        if session is None:
            return "", 0.0

        img = prep_crnn(detection.crop, (IMG_W, IMG_H))
        inp = (img.astype("float32") / 255.0).reshape(1, 1, IMG_H, IMG_W)
        logits = session.run(None, {self._input_name: inp})[0][0]  # (T, C)
        probs = _softmax(logits)

        text, conf = _ctc_greedy(probs, self._labels, self._blank)
        if not text or conf < self.settings.ocr_min_confidence:
            return text, round(conf, 4)
        return text, round(conf, 4)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _ctc_greedy(probs: np.ndarray, labels: list[str], blank: int) -> tuple[str, float]:
    """Greedy CTC decode: argmax per step, collapse repeats, drop blanks."""
    idx = probs.argmax(axis=1)
    chars: list[str] = []
    confs: list[float] = []
    prev = -1
    for t, i in enumerate(idx):
        i = int(i)
        if i != blank and i != prev and i < len(labels):
            chars.append(labels[i])
            confs.append(float(probs[t, i]))
        prev = i
    text = "".join(chars)
    conf = float(np.mean(confs)) if confs else 0.0
    return text, conf
