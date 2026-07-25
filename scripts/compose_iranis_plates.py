"""Compose realistic Iranian plates from REAL character crops (Iranis dataset).

The synthetic plate generator uses one stylized font, which confuses some
characters (e.g. 4 vs 6). This composer instead pastes *real* plate-character
crops (the Iranis per-class folders) into the standard layout, so the reader
learns genuine character shapes. Images are written as ``{i}__{plate}.png`` for
the CRNN trainer's direct-label format.

Usage:
    python scripts/compose_iranis_plates.py --iranis Iranis-dataset \
        --out /root/crnn_mixed --count 8000
"""

from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

import cv2
import numpy as np

# Iranis folder -> Persian plate character.
FOLDER_TO_FA = {
    "A": "ا", "B": "ب", "D": "د", "Gh": "ق", "H": "ه", "J": "ج", "L": "ل",
    "M": "م", "N": "ن", "P": "پ", "Sad": "ص", "Sin": "س", "T": "ط", "V": "و",
    "Y": "ی", "Taxi": "ت", "PuV": "ع", "PwD": "ژ",
}
DIGITS = [str(i) for i in range(10)]


def _index(iranis: Path) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for folder in DIGITS + list(FOLDER_TO_FA):
        files = glob.glob(str(iranis / folder / "*"))
        if files:
            idx[folder] = files
    return idx


def _crop(idx, folder, rng) -> np.ndarray:
    img = cv2.imread(rng.choice(idx[folder]), cv2.IMREAD_GRAYSCALE)
    return img if img is not None else np.full((30, 20), 255, np.uint8)


def _augment(bgr: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    h, w = bgr.shape[:2]
    # perspective
    if rng.random() < 0.8:
        m = 0.06
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        offset = (rng.uniform(-m, m, src.shape) * np.array([w, h])).astype(np.float32)
        dst = (src + offset).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        bgr = cv2.warpPerspective(bgr, M, (w, h), borderValue=(120, 120, 120))
    # rotation
    ang = rng.uniform(-6, 6)
    R = cv2.getRotationMatrix2D((w / 2, h / 2), ang, rng.uniform(0.9, 1.05))
    bgr = cv2.warpAffine(bgr, R, (w, h), borderValue=(120, 120, 120))
    # brightness/contrast
    bgr = cv2.convertScaleAbs(bgr, alpha=rng.uniform(0.6, 1.25), beta=rng.randint(-30, 30))
    # blur + noise
    if rng.random() < 0.5:
        k = int(rng.choice([3, 3, 5]))
        bgr = cv2.GaussianBlur(bgr, (k, k), 0)
    if rng.random() < 0.6:
        bgr = np.clip(bgr.astype(np.int16) + rng.normal(0, rng.uniform(3, 14), bgr.shape).astype(np.int16), 0, 255).astype(np.uint8)
    return bgr


def compose(seq_folders, idx, rng) -> np.ndarray:
    H = 70
    glyphs = []
    for fol in seq_folders:
        g = _crop(idx, fol, rng)
        gh, gw = g.shape
        nh = H - 12
        nw = max(6, int(gw * nh / gh))
        glyphs.append(cv2.resize(g, (nw, nh)))

    gap = rng.randint(4, 10)
    margin = rng.randint(8, 16)
    strip = 34  # blue IR strip on the left
    # extra gap before the 2-digit region (last two glyphs)
    total_w = sum(g.shape[1] for g in glyphs)
    W = strip + margin * 2 + total_w + gap * (len(glyphs) - 1) + rng.randint(6, 16)
    plate = np.full((H, W, 3), (245, 245, 245), np.uint8)
    # IR strip
    plate[:, :strip] = (150, 70, 30)
    cv2.putText(plate, "IR", (4, H // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    x = strip + margin
    for i, g in enumerate(glyphs):
        gh, gw = g.shape
        y = (H - gh) // 2 + rng.randint(-2, 2)
        y = max(0, min(y, H - gh))
        g3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        plate[y:y + gh, x:x + gw] = np.minimum(plate[y:y + gh, x:x + gw], g3)
        x += gw + gap + (rng.randint(6, 12) if i == 5 else 0)

    # place plate onto a mid-grey scene with margin so crops vary
    ph, pw = plate.shape[:2]
    scene = np.full((ph + 24, pw + 24, 3), rng.randint(60, 150), np.uint8)
    scene[12:12 + ph, 12:12 + pw] = plate
    return _augment(scene, rng)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iranis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nprng = np.random.RandomState(args.seed)
    idx = _index(Path(args.iranis))
    letters = [f for f in FOLDER_TO_FA if f in idx]
    out = Path(args.out) / "images"
    out.mkdir(parents=True, exist_ok=True)
    print(f"composing {args.count} plates from {len(idx)} Iranis classes", flush=True)

    for i in range(args.count):
        seq = [rng.choice(DIGITS), rng.choice(DIGITS), rng.choice(letters),
               rng.choice(DIGITS), rng.choice(DIGITS), rng.choice(DIGITS),
               rng.choice(DIGITS), rng.choice(DIGITS)]
        plate_str = "".join(FOLDER_TO_FA.get(f, f) for f in seq)
        img = compose(seq, idx, nprng)
        cv2.imwrite(str(out / f"ir{i}__{plate_str}.png"), img)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{args.count}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
