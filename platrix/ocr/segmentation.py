"""Character segmentation for a cropped plate.

Applies homomorphic filtering to normalize illumination, thresholds the plate,
and extracts individual character boxes ordered left → right. This is a cleaned,
dependency-light reworking of the original segmentation routine.
"""

from __future__ import annotations

import cv2
import numpy as np


def _homomorphic(gray: np.ndarray, sigma: int = 15) -> np.ndarray:
    """Illumination-normalizing homomorphic filter."""
    rows, cols = gray.shape
    img_log = np.log1p(np.asarray(gray, dtype="float") / 255.0)

    m, n = 2 * rows + 1, 2 * cols + 1
    x, y = np.meshgrid(np.linspace(0, n - 1, n), np.linspace(0, m - 1, m))
    center_x, center_y = np.ceil(n / 2), np.ceil(m / 2)
    gaussian = (x - center_x) ** 2 + (y - center_y) ** 2

    h_low = np.exp(-gaussian / (2 * sigma * sigma))
    h_high = 1 - h_low
    h_low = np.fft.ifftshift(h_low)
    h_high = np.fft.ifftshift(h_high)

    freq = np.fft.fft2(img_log, (m, n))
    out_low = np.real(np.fft.ifft2(freq * h_low, (m, n)))
    out_high = np.real(np.fft.ifft2(freq * h_high, (m, n)))

    out = 0.5 * out_low[0:rows, 0:cols] + 1.5 * out_high[0:rows, 0:cols]
    result = np.expm1(out)
    span = np.max(result) - np.min(result)
    if span == 0:
        return np.zeros_like(gray, dtype="uint8")
    result = (result - np.min(result)) / span
    return np.array(255 * result, dtype="uint8")


def segment_characters(
    plate_bgr: np.ndarray,
    out_size: tuple[int, int] = (60, 120),
    max_chars: int = 10,
) -> list[np.ndarray]:
    """Return ordered grayscale character crops resized to ``out_size``.

    ``out_size`` is ``(width, height)`` matching the CNN input.
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return []

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    # Trim likely borders/bolts on the left/right edges.
    h, w = gray.shape
    if w > 60:
        gray = gray[:, 20 : w - 15]

    filtered = _homomorphic(gray)
    binary = (filtered < 65).astype("uint8") * 255
    binary = cv2.resize(binary, (150, 180), interpolation=cv2.INTER_AREA)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = [cv2.boundingRect(c) for c in contours]

    # Character-height heuristic: keep boxes near the tallest glyph.
    if not boxes:
        return []
    max_h = max(bh for (_, _, _, bh) in boxes)
    height_floor = max_h - max_h / 5.0

    chars: list[tuple[int, np.ndarray]] = []
    for (x, y, bw, bh) in boxes:
        if bw <= 2 or bh < height_floor:
            continue
        y0, y1 = max(y - 5, 0), min(y + bh + 5, binary.shape[0])
        x0, x1 = max(x - 5, 0), min(x + bw + 5, binary.shape[1])
        glyph = binary[y0:y1, x0:x1]
        if glyph.size == 0:
            continue
        glyph = cv2.resize(glyph, out_size, interpolation=cv2.INTER_AREA)
        chars.append((x, glyph))

    chars.sort(key=lambda item: item[0])  # left → right
    return [g for _, g in chars[:max_chars]]
