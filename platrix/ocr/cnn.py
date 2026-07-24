"""CNN-based OCR backend.

Segments the plate into characters and classifies each with a Keras CNN. The
class index → character map is loaded from ``<weights>.labels.json`` when present
(one entry per output neuron); otherwise it defaults to the digits 0–9.

If the model file is missing, the backend degrades gracefully: detection and
snapshot logging keep working while ``read`` returns an empty string. Train a
model with ``python -m platrix.ocr.train`` (see ``scripts/train_ocr.py``).
"""

from __future__ import annotations

import json

import numpy as np

from platrix.config import Settings
from platrix.core.types import PlateDetection
from platrix.logging_conf import get_logger
from platrix.ocr.base import PlateOCR
from platrix.ocr.segmentation import segment_characters

logger = get_logger(__name__)

DEFAULT_LABELS = list("0123456789")


class CnnOCR(PlateOCR):
    """Segment-then-classify OCR using a Keras CNN."""

    name = "cnn"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._labels: list[str] = DEFAULT_LABELS
        self._tried = False  # whether we've attempted to load the model
        self.input_size = (60, 120)  # (width, height)

    def warmup(self) -> None:
        self._load()

    def _load(self):
        if self._model is not None:
            return self._model
        if self._tried:  # already tried and failed — don't retry every frame
            return None
        self._tried = True
        weights = self.settings.ocr_weights
        if not weights.exists():
            logger.warning(
                "OCR weights not found at %s — running detection-only. "
                "Train a model or set PLATRIX_OCR=none to silence this.",
                weights,
            )
            return None
        try:
            from tensorflow import keras
        except ImportError:  # pragma: no cover
            logger.warning("TensorFlow not installed — OCR disabled.")
            return None

        logger.info("Loading OCR model from %s", weights)
        self._model = keras.models.load_model(str(weights))

        labels_path = weights.with_suffix(".labels.json")
        if labels_path.exists():
            self._labels = json.loads(labels_path.read_text(encoding="utf-8"))
        return self._model

    def read(self, detection: PlateDetection) -> tuple[str, float]:
        model = self._load()
        if model is None:
            return "", 0.0

        glyphs = segment_characters(detection.crop, out_size=self.input_size)
        if not glyphs:
            return "", 0.0

        width, height = self.input_size
        batch = np.stack(glyphs).astype("float32") / 255.0
        batch = batch.reshape(len(glyphs), height, width, 1)
        probs = model.predict(batch, verbose=0)

        chars: list[str] = []
        confidences: list[float] = []
        for row in probs:
            idx = int(np.argmax(row))
            conf = float(row[idx])
            if conf < self.settings.ocr_min_confidence:
                continue
            if 0 <= idx < len(self._labels):
                chars.append(self._labels[idx])
                confidences.append(conf)

        if not chars:
            return "", 0.0

        text = "".join(chars)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return text, round(mean_conf, 4)
