"""Train a segmentation-free plate reader (CRNN + CTC) and export to ONNX.

Instead of splitting a plate into characters (fragile on real photos), a single
CRNN reads the **whole plate** at once. It is trained on realistic full-plate
images (official template + font, with perspective/blur/illumination
augmentation) whose ground-truth text is encoded in the filename
``{i}_{DD}-{CHAR}-{DDD}-{DD}.png``.

Outputs:
    models/ocr_crnn.onnx          # 1 x 1 x 32 x 128 -> (T, num_classes+1) logits
    models/ocr_crnn.labels.json   # class list (CTC blank is the last index)

Usage:
    python scripts/train_crnn.py --data /path/to/generated_plates --epochs 12
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMG_H, IMG_W = 32, 128

# Generator letter codes -> Persian plate characters.
_CODE_TO_PERSIAN = {
    "EIN": "ع", "B": "ب", "N": "ن", "T": "ت", "H": "ح", "D": "د", "Q": "ق",
    "J": "ج", "HE": "ه", "SIN": "س", "SAD": "ص", "TA": "ط", "V": "و", "M": "م",
    "Y": "ی", "L": "ل", "Z": "ز", "ZH": "ژ", "TH": "ث", "P": "پ", "SH": "ش",
    "A": "ا",
}


def parse_label(stem: str) -> list[str] | None:
    """`'123_32-Y-528-86'` -> ['3','2','ی','5','2','8','8','6']."""
    if "_" not in stem:
        return None
    parts = stem.split("_", 1)[1].split("-")
    if len(parts) != 4:
        return None
    d1, code, d2, region = parts
    letter = _CODE_TO_PERSIAN.get(code)
    if letter is None:
        return None
    seq = list(d1) + [letter] + list(d2) + list(region)
    return seq if all(c.isdigit() or c in _CODE_TO_PERSIAN.values() for c in seq) else None


def load_dataset(root: Path):
    files = sorted(glob.glob(str(root / "images" / "*.png")))
    samples = []
    for f in files:
        seq = parse_label(Path(f).stem)
        if seq is not None:
            samples.append((f, seq))
    classes = sorted({c for _, s in samples for c in s})
    print(f"{len(samples)} plates, {len(classes)} classes")
    return samples, classes


def load_gray(path: str) -> np.ndarray:
    g = np.asarray(Image.open(path).convert("L"))
    return cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Platrix CRNN plate reader")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/ocr_crnn.onnx")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    samples, classes = load_dataset(Path(args.data))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    blank = len(classes)

    # Preload images (7k * 32*128 = ~28 MB) for speed.
    X = np.empty((len(samples), IMG_H, IMG_W), np.float32)
    Y = []
    for i, (f, seq) in enumerate(samples):
        X[i] = load_gray(f).astype(np.float32) / 255.0
        Y.append([cls_to_idx[c] for c in seq])

    perm = rng.permutation(len(samples))
    split = int(len(samples) * 0.92)
    tr, va = perm[:split], perm[split:]

    def _aug(img):
        # light runtime augmentation on top of the generator's baked-in variety
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        if rng.random() < 0.5:
            img = np.clip(img + rng.normal(0, rng.uniform(0.01, 0.05), img.shape), 0, 1).astype(np.float32)
        return img

    class DS(Dataset):
        def __init__(self, idxs, train):
            self.idxs, self.train = idxs, train

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, k):
            i = self.idxs[k]
            img = _aug(X[i]) if self.train else X[i]
            return torch.from_numpy(img[None].copy()), torch.tensor(Y[i], dtype=torch.long)

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        targets = torch.cat([b[1] for b in batch])
        lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
        return imgs, targets, lengths

    train_dl = DataLoader(DS(tr, True), batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(DS(va, False), batch_size=args.batch, collate_fn=collate)

    class CRNN(nn.Module):
        def __init__(self, n_cls):
            super().__init__()

            def blk(i, o, k=3, s=1, p=1):
                return nn.Sequential(nn.Conv2d(i, o, k, s, p), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

            self.cnn = nn.Sequential(
                blk(1, 64), nn.MaxPool2d(2, 2),        # 16x64
                blk(64, 128), nn.MaxPool2d(2, 2),      # 8x32
                blk(128, 256), blk(256, 256), nn.MaxPool2d((2, 1), (2, 1)),  # 4x32
                blk(256, 256), nn.MaxPool2d((2, 1), (2, 1)),                  # 2x32
                blk(256, 256, k=2, s=1, p=0),          # 1x31
            )
            self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(256, n_cls)

        def forward(self, x):
            f = self.cnn(x).squeeze(2).permute(0, 2, 1)
            r, _ = self.rnn(f)
            return self.fc(r)

    model = CRNN(len(classes) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ctc = nn.CTCLoss(blank=blank, zero_infinity=True)

    def greedy(logits_row):
        idx = logits_row.argmax(1)
        out, prev = [], -1
        for i in idx.tolist():
            if i != blank and i != prev:
                out.append(classes[i])
            prev = i
        return "".join(out)

    for ep in range(1, args.epochs + 1):
        model.train()
        for imgs, targets, lengths in train_dl:
            logits = model(imgs)
            logp = logits.log_softmax(2).permute(1, 0, 2)
            in_len = torch.full((imgs.size(0),), logp.size(0), dtype=torch.long)
            loss = ctc(logp, targets, in_len, lengths)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, targets, lengths in val_dl:
                logits = model(imgs)
                pos = 0
                for b in range(imgs.size(0)):
                    n = int(lengths[b])
                    gt = "".join(classes[int(t)] for t in targets[pos : pos + n]); pos += n
                    correct += greedy(logits[b]) == gt
                    total += 1
        print(f"epoch {ep}/{args.epochs}  loss={loss.item():.3f}  plate_acc={correct / total:.3f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model, torch.zeros(1, 1, IMG_H, IMG_W), str(out),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13, dynamo=False,
    )
    out.with_suffix(".labels.json").write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}  ({len(classes)} classes + blank)")


if __name__ == "__main__":
    main()
