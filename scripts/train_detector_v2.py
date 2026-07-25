"""Train an improved plate detector on a combined dataset.

Merges two sources into one single-class ("plate") YOLO dataset:
  1. Real Pascal-VOC plate photos (images + XML boxes).
  2. Composited Iranian plate scenes whose per-character YOLO boxes are collapsed
     into one tight plate box (the union of the character boxes).

Then fine-tunes yolov8n at a larger image size / more epochs for tighter boxes,
and exports ONNX.

Usage:
    python scripts/train_detector_v2.py --voc /tmp/pd --scenes /root/unified_ds \
        --scene-limit 2500 --epochs 24 --imgsz 448
"""

from __future__ import annotations

import argparse
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def _voc_to_yolo(voc_root: Path, out: Path, rng: random.Random, val_frac: float):
    images = sorted(glob.glob(str(voc_root / "images" / "*")))
    rng.shuffle(images)
    split = int(len(images) * (1 - val_frac))
    n = 0
    for i, img in enumerate(images):
        subset = "train" if i < split else "val"
        name = Path(img).stem
        xml = voc_root / "annotations" / f"{name}.xml"
        if not xml.exists():
            continue
        root = ET.parse(xml).getroot()
        w = int(root.find("size/width").text)
        h = int(root.find("size/height").text)
        lines = []
        for obj in root.findall("object"):
            b = obj.find("bndbox")
            x1, y1, x2, y2 = (int(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
            lines.append(f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        shutil.copy(img, out / "images" / subset / f"real_{name}{Path(img).suffix}")
        (out / "labels" / subset / f"real_{name}.txt").write_text("\n".join(lines))
        n += 1
    print(f"real VOC plates: {n}")


def _scene_char_to_plate(scenes_root: Path, out: Path, limit: int):
    """Collapse per-character boxes into one plate box (their union)."""
    total = 0
    for subset in ("train", "val"):
        imgs = sorted(glob.glob(str(scenes_root / "images" / subset / "*")))
        if limit and subset == "train":
            imgs = imgs[:limit]
        elif limit and subset == "val":
            imgs = imgs[: max(limit // 10, 50)]
        for img in imgs:
            name = Path(img).stem
            lbl = scenes_root / "labels" / subset / f"{name}.txt"
            if not lbl.exists():
                continue
            xs1, ys1, xs2, ys2 = [], [], [], []
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                cx, cy, bw, bh = map(float, p[1:])
                xs1.append(cx - bw / 2); ys1.append(cy - bh / 2)
                xs2.append(cx + bw / 2); ys2.append(cy + bh / 2)
            if not xs1:
                continue
            x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
            # small margin so the box isn't razor-tight
            mx, my = (x2 - x1) * 0.04, (y2 - y1) * 0.12
            x1, y1 = max(x1 - mx, 0), max(y1 - my, 0)
            x2, y2 = min(x2 + mx, 1), min(y2 + my, 1)
            line = f"0 {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}"
            shutil.copy(img, out / "images" / subset / f"scene_{name}{Path(img).suffix}")
            (out / "labels" / subset / f"scene_{name}.txt").write_text(line)
            total += 1
    print(f"composited Iranian scenes: {total}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--work", default="/root/plate_det_ds")
    ap.add_argument("--out", default="models/plate_yolo.onnx")
    ap.add_argument("--scene-limit", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from ultralytics import YOLO

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    for subset in ("train", "val"):
        (work / "images" / subset).mkdir(parents=True, exist_ok=True)
        (work / "labels" / subset).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    _voc_to_yolo(Path(args.voc), work, rng, val_frac=0.15)
    _scene_char_to_plate(Path(args.scenes), work, args.scene_limit)

    (work / "data.yaml").write_text(
        f"path: {work}\ntrain: images/train\nval: images/val\nnames:\n  0: plate\n"
    )

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(work / "data.yaml"), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device="cpu", project=str(work / "runs"), name="det",
        exist_ok=True, verbose=True, plots=False,
    )
    best = work / "runs" / "det" / "weights" / "best.pt"
    exported = YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, dynamic=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, out)
    shutil.copy(best, out.with_suffix(".pt"))
    print(f"Saved detector -> {out}")


if __name__ == "__main__":
    main()
