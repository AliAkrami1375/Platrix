"""Image-enhancement layer.

A quality/denoising stage applied to a detected plate crop *before* character
segmentation. Real plate crops are small, noisy and unevenly lit; enhancing them
first makes segmentation far more reliable (fewer spurious/extra characters) and
sharpens the glyphs for the OCR model.

Pipeline: upscale small crops → edge-preserving denoise → local contrast (CLAHE)
→ unsharp mask. Returns an enhanced **grayscale** image.
"""

from __future__ import annotations

import cv2
import numpy as np

_TARGET_HEIGHT = 128  # upscale short crops to at least this many pixels tall


def enhance_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """Return an enhanced grayscale version of a plate crop.

    Steps:
      1. **Upscale** small crops (cubic) so thin strokes survive thresholding.
      2. **Denoise** with a non-local-means / bilateral pass (edge-preserving).
      3. **CLAHE** for local contrast under uneven lighting.
      4. **Unsharp mask** to crisp the character edges.
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return np.zeros((1, 1), dtype="uint8")

    gray = (
        cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        if plate_bgr.ndim == 3
        else plate_bgr
    )

    # 1) Upscale short crops.
    h, w = gray.shape[:2]
    if h < _TARGET_HEIGHT:
        scale = _TARGET_HEIGHT / h
        gray = cv2.resize(
            gray, (max(int(round(w * scale)), 1), _TARGET_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )

    # 2) Edge-preserving denoise.
    try:
        den = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    except cv2.error:  # pragma: no cover - fallback if NlMeans unavailable
        den = cv2.bilateralFilter(gray, 7, 50, 50)

    # 3) Local contrast.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    con = clahe.apply(den)

    # 4) Unsharp mask.
    blur = cv2.GaussianBlur(con, (0, 0), 1.5)
    sharp = cv2.addWeighted(con, 1.6, blur, -0.6, 0)
    return sharp


# --- CRNN whole-plate preprocessing (identical in training and inference) ---
_CRNN_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def prep_crnn(img: np.ndarray, size: tuple[int, int] = (128, 32)) -> np.ndarray:
    """Enhance a plate crop, then size it for the CRNN.

    Quality/denoising layer applied **identically** in training and at serve
    time (so the model never sees a distribution shift):

      1. grayscale
      2. **upscale** small crops (cubic) so thin strokes survive — recovers
         detail on far / low-resolution plates
      3. **edge-preserving denoise** (bilateral — fast enough for real time,
         keeps character edges while removing sensor noise)
      4. **local contrast** (CLAHE) for shadows / glare
      5. **unsharp mask** to crisp the strokes
      6. resize to the model input

    ``size`` is ``(width, height)``. Returns a ``uint8`` grayscale image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    h, w = gray.shape[:2]
    if h < 40:  # small/far plate → upscale before enhancing
        scale = 40.0 / h
        gray = cv2.resize(gray, (max(int(w * scale), 1), 40), interpolation=cv2.INTER_CUBIC)

    gray = cv2.bilateralFilter(gray, 5, 45, 45)          # denoise, keep edges
    gray = _CRNN_CLAHE.apply(gray)                         # local contrast
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray = cv2.addWeighted(gray, 1.4, blurred, -0.4, 0)   # unsharp
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
