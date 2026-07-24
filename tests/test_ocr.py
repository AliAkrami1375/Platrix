import numpy as np
import pytest

from platrix.config import Settings
from platrix.core.types import PlateDetection
from platrix.ocr import build_ocr
from platrix.ocr.base import NullOCR
from platrix.ocr.segmentation import segment_characters


def _detection(img: np.ndarray) -> PlateDetection:
    h, w = img.shape[:2]
    return PlateDetection(0, 0, w, h, 0.9, img)


def test_none_backend():
    ocr = build_ocr(Settings(ocr="none"))
    assert isinstance(ocr, NullOCR)
    assert ocr.read(_detection(np.zeros((20, 60, 3), np.uint8))) == ("", 0.0)


def test_onnx_backend_graceful_without_model(tmp_path):
    # Point at a non-existent model: backend must degrade, not crash.
    ocr = build_ocr(Settings(ocr="onnx", onnx_weights=tmp_path / "missing.onnx"))
    assert ocr.name == "onnx"
    assert ocr.read(_detection(np.zeros((30, 90, 3), np.uint8))) == ("", 0.0)


def test_segmentation_splits_characters():
    # White plate with three dark bars → three character bands.
    plate = np.full((60, 200, 3), 240, np.uint8)
    for cx in (40, 100, 160):
        plate[15:45, cx - 8 : cx + 8] = 20
    glyphs = segment_characters(plate, out_size=(32, 32))
    assert len(glyphs) == 3
    assert all(g.shape == (32, 32) for g in glyphs)


@pytest.mark.skipif(
    not (Settings().onnx_weights.exists()),
    reason="OCR model not trained/present",
)
def test_onnx_backend_reads_when_model_present():
    ocr = build_ocr(Settings(ocr="onnx"))
    ocr.warmup()
    plate = np.full((60, 200, 3), 240, np.uint8)
    for cx in (40, 100, 160):
        plate[15:45, cx - 8 : cx + 8] = 20
    text, conf = ocr.read(_detection(plate))
    assert isinstance(text, str)
    assert 0.0 <= conf <= 1.0
