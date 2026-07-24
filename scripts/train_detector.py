"""Fine-tune a YOLO plate detector and export it to ONNX.

Converts a Pascal-VOC plate-detection dataset (images + XML boxes) to YOLO
format, trains an ``yolov8n`` model for the single "plate" class, and exports
the best weights so the Platrix ``yolo`` detector backend can use them.

Usage:
    python scripts/train_detector.py --voc /path/to/voc --epochs 30
"""

from __future__ import annotations

import argparse
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def voc_to_yolo(voc_root: Path, out_root: Path, val_frac: float = 0.15, seed: int = 1) -> Path:
    images = sorted(glob.glob(str(voc_root / "images" / "*")))
    random.seed(seed)
    random.shuffle(images)
    split = int(len(images) * (1 - val_frac))
    subsets = {"train": images[:split], "val": images[split:]}

    for subset, files in subsets.items():
        (out_root / "images" / subset).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / subset).mkdir(parents=True, exist_ok=True)
        for img in files:
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
                xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            shutil.copy(img, out_root / "images" / subset / Path(img).name)
            (out_root / "labels" / subset / f"{name}.txt").write_text("\n".join(lines))
        print(f"{subset}: {len(files)} images")

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        f"path: {out_root}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: plate\n"
    )
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Platrix YOLO plate detector")
    parser.add_argument("--voc", required=True, help="VOC dataset root (images/ + annotations/)")
    parser.add_argument("--work", default="/tmp/plate_yolo", help="Working dir for YOLO dataset")
    parser.add_argument("--out", default="models/plate_yolo.onnx")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    from ultralytics import YOLO

    work = Path(args.work)
    data_yaml = voc_to_yolo(Path(args.voc), work)

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        project=str(work / "runs"),
        name="plate",
        exist_ok=True,
        verbose=True,
        plots=False,
    )

    best = work / "runs" / "plate" / "weights" / "best.pt"
    exported = YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, dynamic=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, out)
    shutil.copy(best, out.with_suffix(".pt"))
    print(f"Saved detector -> {out}")


if __name__ == "__main__":
    main()
