"""Train the Persian character-recognition CNN and export it to ONNX.

Consumes a labelled Persian-glyph dataset with this layout::

    <root>/labels.csv          # columns: File_Name, Label
    <root>/data/data/<id>.png  # RGB glyph images (any size; resized internally)

Labels are normalized to their base character (NFKC folds the Arabic
presentation forms back to a single letter), Arabic kaf/yeh are mapped to their
Persian forms, and punctuation is dropped — leaving digits 0-9 plus the Persian
letters used on plates.

The result is written as::

    models/ocr_cnn.onnx          # inference graph (used by the 'onnx' backend)
    models/ocr_cnn.labels.json   # output-neuron -> character map

Usage:
    python scripts/train_ocr.py --data /path/to/dataset --epochs 6
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from pathlib import Path

import sys

import cv2
import numpy as np
from PIL import Image

# Allow running as a plain script (`python scripts/train_ocr.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from platrix.ocr.segmentation import canonical_glyph  # noqa: E402

IMG_SIZE = 32  # model input is 1 x IMG_SIZE x IMG_SIZE (grayscale)

# Arabic -> Persian letter unification so plate text is consistent.
_ARABIC_TO_PERSIAN = {"ك": "ک", "ي": "ی", "ة": "ه"}
# Characters that never appear on a plate.
_DROP = {"،", "؛"}


def normalize_label(raw: str) -> str | None:
    base = unicodedata.normalize("NFKC", raw).strip()
    base = _ARABIC_TO_PERSIAN.get(base, base)
    if not base or base in _DROP:
        return None
    return base


def _image_dir(root: Path) -> Path:
    for cand in (root / "data" / "data", root / "data", root):
        if cand.is_dir() and any(cand.glob("*.png")):
            return cand
    raise FileNotFoundError(f"Could not find image files under {root}")


def load_dataset(root: Path):
    labels_csv = root / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found in {root}")
    img_dir = _image_dir(root)

    rows = list(csv.reader(labels_csv.open(encoding="utf-8-sig")))[1:]
    samples: list[tuple[Path, str]] = []
    for file_name, label, *_ in rows:
        base = normalize_label(label)
        if base is None:
            continue
        path = img_dir / f"{file_name}.png"
        if path.exists():
            samples.append((path, base))

    classes = sorted({c for _, c in samples})
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Loaded {len(samples)} samples across {len(classes)} classes")

    x = np.empty((len(samples), 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    y = np.empty(len(samples), dtype=np.int64)
    for i, (path, cls) in enumerate(samples):
        gray = np.asarray(Image.open(path).convert("L"))
        # Binarize to white-on-black, then canonicalize EXACTLY as the segmenter
        # does at inference (tight-crop, square-pad, resize) so train == serve.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        glyph = canonical_glyph(binary, (IMG_SIZE, IMG_SIZE))
        x[i, 0] = glyph.astype(np.float32) / 255.0
        y[i] = cls_to_idx[cls]
    return x, y, classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Platrix Persian OCR (ONNX)")
    parser.add_argument("--data", required=True, help="Dataset root (contains labels.csv)")
    parser.add_argument("--out", default="models/ocr_cnn.onnx")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    import cv2
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y, classes = load_dataset(Path(args.data))

    def augment(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Random affine so the model tolerates how the segmenter frames glyphs."""
        h, w = img.shape
        angle = rng.uniform(-12, 12)
        scale = rng.uniform(0.65, 1.15)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        m[0, 2] += rng.uniform(-0.18, 0.18) * w
        m[1, 2] += rng.uniform(-0.18, 0.18) * h
        return cv2.warpAffine(img, m, (w, h), borderValue=0.0)

    class AugDataset(Dataset):
        def __init__(self, xs, ys):
            self.xs, self.ys = xs, ys
            self.rng = np.random.RandomState(args.seed)

        def __len__(self):
            return len(self.xs)

        def __getitem__(self, i):
            img = augment(self.xs[i, 0], self.rng)
            return torch.from_numpy(img[None]), int(self.ys[i])

    # Deterministic train/val split.
    idx = np.random.permutation(len(x))
    split = int(len(x) * 0.9)
    tr, va = idx[:split], idx[split:]
    train_ds = AugDataset(x[tr], y[tr])  # augmented
    val_ds = TensorDataset(torch.from_numpy(x[va]), torch.from_numpy(y[va]))  # clean
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    class OcrCNN(nn.Module):
        def __init__(self, n: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Dropout(0.3), nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(128, n),
            )

        def forward(self, t):
            return self.net(t)

    model = OcrCNN(len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        print(f"epoch {epoch}/{args.epochs}  val_acc={correct / total:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    # Save a checkpoint first so a failed export never loses training.
    torch.save(model.state_dict(), out.with_suffix(".pt"))
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
    # dynamo=False uses the classic TorchScript exporter (no onnxscript needed).
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )
    out.with_suffix(".labels.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved model  -> {out}")
    print(f"Saved labels -> {out.with_suffix('.labels.json')}  ({len(classes)} classes)")


if __name__ == "__main__":
    main()
