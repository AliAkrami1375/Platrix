"""Character segmentation for a cropped plate.

Given a plate crop this isolates the individual characters (ordered left →
right) and returns each as a fixed-size grayscale image ready for the OCR CNN.

The approach is a **vertical projection profile**: normalize height, Otsu-
threshold so characters are white on black (matching the training data), then
scan the column-wise ink profile to find character bands separated by gaps.
Projection segmentation keeps a Persian letter's dots together with its body
(they share the same column band), which contour splitting does not.
"""

from __future__ import annotations

import cv2
import numpy as np

_NORM_HEIGHT = 96  # normalize every plate to this height before segmenting


def _crop_to_plate(gray: np.ndarray) -> np.ndarray:
    """Trim any dark border around the bright plate so Otsu isn't skewed."""
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Only accept it if it keeps most of the crop (a real plate fills the box).
    if w * h >= 0.35 * gray.size and w > 20 and h > 10:
        return gray[y : y + h, x : x + w]
    return gray


def _binarize(plate_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = _crop_to_plate(gray)
    h0, w0 = gray.shape
    scale = _NORM_HEIGHT / h0
    gray = cv2.resize(gray, (max(int(w0 * scale), 1), _NORM_HEIGHT))
    gray = cv2.bilateralFilter(gray, 5, 40, 40)
    # Plates are dark chars on light: INV puts characters in white. Flip if the
    # crop is inverted so the (sparser) foreground stays white.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if binary.mean() > 127:
        binary = cv2.bitwise_not(binary)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    return binary


def _square_pad(glyph: np.ndarray, pad_ratio: float = 0.18) -> np.ndarray:
    h, w = glyph.shape
    side = max(h, w)
    pad = int(side * pad_ratio)
    side += pad * 2
    canvas = np.zeros((side, side), dtype="uint8")
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = glyph
    return canvas


def canonical_glyph(binary_glyph: np.ndarray, out_size: tuple[int, int] = (32, 32)) -> np.ndarray:
    """Canonicalize a white-on-black glyph: tight-crop, square-pad, resize.

    Used by BOTH the segmenter (at inference) and the trainer (on the dataset)
    so the OCR model sees the same glyph framing in training and in production.
    """
    rows = np.where(binary_glyph.sum(axis=1) > 0)[0]
    cols = np.where(binary_glyph.sum(axis=0) > 0)[0]
    if rows.size == 0 or cols.size == 0:
        return cv2.resize(binary_glyph, out_size, interpolation=cv2.INTER_AREA)
    crop = binary_glyph[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    crop = _square_pad(crop)
    return cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)


def _column_bands(binary: np.ndarray) -> list[tuple[int, int]]:
    """Return (x_start, x_end) bands of contiguous inked columns."""
    height, width = binary.shape
    col_ink = (binary > 0).sum(axis=0)
    active = col_ink > max(1, int(0.03 * height))

    bands: list[tuple[int, int]] = []
    start = None
    for x in range(width):
        if active[x] and start is None:
            start = x
        elif not active[x] and start is not None:
            bands.append((start, x))
            start = None
    if start is not None:
        bands.append((start, width))

    # Merge bands separated by a very small gap (dots / broken strokes).
    merged: list[tuple[int, int]] = []
    min_gap = max(2, int(0.012 * width))
    for band in bands:
        if merged and band[0] - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)
    return merged


def segment_characters(
    plate_bgr: np.ndarray,
    out_size: tuple[int, int] = (32, 32),
    max_chars: int = 10,
) -> list[np.ndarray]:
    """Return ordered grayscale character crops (white on black) at ``out_size``.

    ``out_size`` is ``(width, height)``.
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return []

    binary = _binarize(plate_bgr)
    height, width = binary.shape
    min_w = max(2, int(0.008 * width))

    glyphs: list[np.ndarray] = []
    for x0, x1 in _column_bands(binary):
        if x1 - x0 < min_w:
            continue
        band = binary[:, x0:x1]
        rows = np.where(band.sum(axis=1) > 0)[0]
        if rows.size == 0:
            continue
        y0, y1 = int(rows[0]), int(rows[-1])
        bw, bh = x1 - x0, y1 - y0 + 1
        # A character spans a reasonable fraction of the plate height (digits can
        # be short, so keep this permissive) and fills enough of its own box
        # (rejects sparse noise / stray marks).
        if bh < 0.18 * height:
            continue
        density = int((band > 0).sum()) / float(bw * bh)
        if density < 0.06:
            continue
        glyphs.append(canonical_glyph(band[y0 : y1 + 1], out_size))

    return glyphs[:max_chars]
