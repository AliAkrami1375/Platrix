"""Train a single end-to-end plate reader: one YOLO that detects & classifies
every plate character directly in a full frame (no separate detector + OCR).

Generated plate images already ship per-character YOLO labels. To make the model
work on full photos (not just tight plate crops), each plate is composited onto
a random background scene and its character boxes are transformed accordingly.
At inference the model detects all characters, which are ordered left→right into
the plate string.

Outputs:
    models/plate_ocr_yolo.onnx        # single unified model (32 char classes)
    models/plate_ocr_yolo.labels.json # class index -> Persian character

Usage:
    python scripts/train_unified.py --plates /root/crnn_ds15k --epochs 40
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Generator class index (0..31) -> Persian plate character.
CODE_TO_PERSIAN = {
    "EIN": "ع", "B": "ب", "N": "ن", "T": "ت", "H": "ح", "D": "د", "Q": "ق",
    "J": "ج", "HE": "ه", "SIN": "س", "SAD": "ص", "TA": "ط", "V": "و", "M": "م",
    "Y": "ی", "L": "ل", "Z": "ز", "ZH": "ژ", "TH": "ث", "P": "پ", "SH": "ش",
    "A": "ا",
}


def build_labels(data_yaml: Path) -> list[str]:
    """Read the generator's class names and map them to Persian characters."""
    names: dict[int, str] = {}
    for line in data_yaml.read_text().splitlines():
        line = line.strip()
        if line and line[0].isdigit() and ":" in line:
            idx, name = line.split(":", 1)
            names[int(idx)] = name.strip().strip("'\"")
    out = []
    for i in range(len(names)):
        code = names[i]
        out.append(code if code.isdigit() else CODE_TO_PERSIAN.get(code, code))
    return out


def _backgrounds(bg_dir: Path | None) -> list[np.ndarray]:
    bgs: list[np.ndarray] = []
    if bg_dir and bg_dir.is_dir():
        for f in sorted(glob.glob(str(bg_dir / "*")))[:400]:
            im = cv2.imread(f)
            if im is not None:
                bgs.append(im)
    return bgs


def composite(plate_bgr, boxes, bgs, rng):
    """Paste a plate onto a random scene; return (scene, transformed_boxes)."""
    W = rng.randint(480, 960)
    H = rng.randint(360, 720)
    if bgs and rng.random() < 0.7:
        bg = bgs[rng.randint(0, len(bgs) - 1)]
        scene = cv2.resize(bg, (W, H))
        scene = cv2.convertScaleAbs(scene, alpha=rng.uniform(0.7, 1.1), beta=rng.randint(-20, 20))
    else:
        base = rng.randint(40, 180)
        scene = np.full((H, W, 3), base, np.uint8)
        scene = cv2.add(scene, rng.randint(0, 30, (H, W, 3), dtype=np.uint8))

    ph, pw = plate_bgr.shape[:2]
    target_w = int(rng.uniform(0.30, 0.72) * W)
    scale = target_w / pw
    nw, nh = target_w, max(1, int(ph * scale))
    if nh >= H:
        nh = int(H * 0.6); nw = int(pw * nh / ph)
    plate = cv2.resize(plate_bgr, (nw, nh))
    ox = rng.randint(0, max(1, W - nw))
    oy = rng.randint(int(0.25 * H), max(int(0.25 * H) + 1, H - nh))
    scene[oy : oy + nh, ox : ox + nw] = plate

    new_boxes = []
    for cls, cx, cy, bw, bh in boxes:
        nx = (ox + cx * nw) / W
        ny = (oy + cy * nh) / H
        nbw = bw * nw / W
        nbh = bh * nh / H
        new_boxes.append((cls, nx, ny, nbw, nbh))
    return scene, new_boxes


def make_dataset(plates_root: Path, out: Path, bg_dir: Path | None, seed: int):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    bgs = _backgrounds(bg_dir)
    imgs = sorted(glob.glob(str(plates_root / "images" / "*.png")))
    print(f"compositing {len(imgs)} plates onto scenes ({len(bgs)} backgrounds)")

    for subset, sl in (("train", slice(0, int(len(imgs) * 0.92))),
                       ("val", slice(int(len(imgs) * 0.92), None))):
        (out / "images" / subset).mkdir(parents=True, exist_ok=True)
        (out / "labels" / subset).mkdir(parents=True, exist_ok=True)
        for img_path in imgs[sl]:
            stem = Path(img_path).stem
            lbl = plates_root / "labels" / f"{stem}.txt"
            if not lbl.exists():
                continue
            boxes = []
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) == 5:
                    boxes.append((int(p[0]), *map(float, p[1:])))
            plate = cv2.imread(img_path)
            if plate is None:
                continue
            scene, nb = composite(plate, boxes, bgs, np_rng)
            cv2.imwrite(str(out / "images" / subset / f"{stem}.jpg"), scene)
            (out / "labels" / subset / f"{stem}.txt").write_text(
                "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in nb)
            )

    names = build_labels(plates_root / "data.yaml")
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {json.dumps(list(range(len(names))))}\n".replace(
            json.dumps(list(range(len(names)))),
            "\n" + "\n".join(f"  {i}: '{i}'" for i in range(len(names))),
        )
    )
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the single unified plate model")
    ap.add_argument("--plates", required=True, help="Generated plates root (images/ + labels/)")
    ap.add_argument("--backgrounds", default="/tmp/ircp/IRCP_dataset_640X480")
    ap.add_argument("--work", default="/root/unified_ds")
    ap.add_argument("--out", default="models/plate_ocr_yolo.onnx")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import shutil

    from ultralytics import YOLO

    work = Path(args.work)
    names = make_dataset(Path(args.plates), work, Path(args.backgrounds), args.seed)

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(work / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device="cpu", project=str(work / "runs"), name="unified",
        exist_ok=True, verbose=True, plots=False,
    )
    best = work / "runs" / "unified" / "weights" / "best.pt"
    exported = YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, dynamic=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, out)
    shutil.copy(best, out.with_suffix(".pt"))
    out.with_suffix(".labels.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    print(f"Saved unified model -> {out}  ({len(names)} classes)")


if __name__ == "__main__":
    main()
