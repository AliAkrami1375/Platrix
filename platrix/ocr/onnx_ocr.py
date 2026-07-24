"""ONNX-based Persian OCR backend.

Segments the plate into characters, classifies each with an ONNX CNN via
``onnxruntime`` (portable, no TensorFlow/PyTorch needed at serve time), and maps
predictions to the Iranian plate alphabet. Train the model with
``python scripts/train_ocr.py``.

If the model file is missing the backend degrades gracefully: detection and
snapshot logging keep working while ``read`` returns an empty string.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from platrix.config import Settings
from platrix.core.types import PlateDetection
from platrix.logging_conf import get_logger
from platrix.ocr.base import PlateOCR
from platrix.ocr.plate_grammar import decode_plate
from platrix.ocr.segmentation import segment_characters

logger = get_logger(__name__)

IMG_SIZE = 32  # must match scripts/train_ocr.py


class OnnxOCR(PlateOCR):
    """Segment-then-classify OCR using an ONNX CNN."""

    name = "onnx"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = None
        self._labels: list[str] = []
        self._input_name: str = "input"
        self._tried = False

    def warmup(self) -> None:
        self._load()

    def _load(self):
        if self._session is not None or self._tried:
            return self._session
        self._tried = True

        weights = self.settings.onnx_weights
        if not weights.exists():
            logger.warning(
                "OCR model not found at %s — running detection-only. "
                "Train one with: python scripts/train_ocr.py --data <dataset>",
                weights,
            )
            return None
        try:
            import onnxruntime as ort
        except ImportError:  # pragma: no cover
            logger.warning("onnxruntime not installed — OCR disabled.")
            return None

        labels_path = weights.with_suffix(".labels.json")
        if not labels_path.exists():
            logger.warning("Label map %s missing — OCR disabled.", labels_path)
            return None

        logger.info("Loading ONNX OCR model from %s", weights)
        self._session = ort.InferenceSession(
            str(weights), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._labels = json.loads(labels_path.read_text(encoding="utf-8"))
        return self._session

    def read(self, detection: PlateDetection) -> tuple[str, float]:
        session = self._load()
        if session is None:
            return "", 0.0

        glyphs = segment_characters(detection.crop, out_size=(IMG_SIZE, IMG_SIZE))
        if not glyphs:
            return "", 0.0

        probs = self._classify_tta(session, glyphs)

        # Grammar-constrained decoding (digit/letter by plate position). All
        # glyphs are kept so the plate structure stays intact.
        chars, confidences = decode_plate(probs, self._labels)
        if not chars:
            return "", 0.0

        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        if mean_conf < self.settings.ocr_min_confidence:
            return "", 0.0

        # Labels are already ASCII digits + Persian letters, so join directly:
        # this preserves the Persian letter (e.g. "11و11427") for display and
        # for watchlist matching (which normalizes digits but keeps letters).
        text = "".join(chars)
        return text, round(mean_conf, 4)

    def _classify_tta(self, session, glyphs: list[np.ndarray]) -> np.ndarray:
        """Softmax probabilities per glyph, averaged over small shifts (TTA).

        Averaging a few 1‑pixel shifts smooths out the classifier's sensitivity
        to exactly how a character was framed by the segmenter, reducing
        look‑alike errors (e.g. ۱ vs ۵).
        """
        shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        acc = np.zeros((len(glyphs), len(self._labels)), dtype="float32")
        for dx, dy in shifts:
            variants = np.empty((len(glyphs), 1, IMG_SIZE, IMG_SIZE), dtype="float32")
            for i, g in enumerate(glyphs):
                m = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(g, m, (IMG_SIZE, IMG_SIZE), borderValue=0)
                variants[i, 0] = shifted.astype("float32") / 255.0
            logits = session.run(None, {self._input_name: variants})[0]
            acc += _softmax(logits)
        return acc / len(shifts)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)
