"""Background 'Learn' training job.

Trains the CRNN plate reader on user-labelled samples (a plate crop + its text),
optionally mixed with synthetic plates for volume. Runs as a detached process so
it survives browser refreshes; it streams progress to a JSON file the dashboard
polls. Can install a CUDA build of PyTorch on request and train on the GPU.

Usage:
    python scripts/learn_train.py --job data/learn/job.json \
        --samples data/learn/samples.json --out models/ocr_crnn.onnx \
        --synthetic /root/crnn_ds15k --device gpu --install-cuda --epochs 15
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from platrix.preprocessing import prep_crnn  # noqa: E402

IMG_H, IMG_W = 32, 128
CODE_TO_PERSIAN = {
    "EIN": "ع", "B": "ب", "N": "ن", "T": "ت", "H": "ح", "D": "د", "Q": "ق",
    "J": "ج", "HE": "ه", "SIN": "س", "SAD": "ص", "TA": "ط", "V": "و", "M": "م",
    "Y": "ی", "L": "ل", "Z": "ز", "ZH": "ژ", "TH": "ث", "P": "پ", "SH": "ش", "A": "ا",
}


class Job:
    """Writes structured progress to a JSON file (the UI polls it)."""

    def __init__(self, path: Path):
        self.path = path
        self.state = {
            "status": "running", "step": "starting", "progress": 0,
            "epoch": 0, "epochs": 0, "accuracy": None, "device": "cpu",
            "log": [], "message": "", "started": None, "finished": None,
        }

    def update(self, **kw):
        self.state.update(kw)
        self._flush()

    def log(self, line: str):
        self.state["log"] = (self.state["log"] + [line])[-200:]
        self._flush()

    def _flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False))
        tmp.replace(self.path)


def _normalize(text: str) -> list[str]:
    return [c for c in text if c.strip()]


def _load_user_samples(samples_json: Path):
    import cv2

    if not samples_json.exists():
        return []
    items = json.loads(samples_json.read_text(encoding="utf-8"))
    out = []
    for it in items:
        img = cv2.imread(it["image_path"])
        if img is None or not it.get("plate_text"):
            continue
        h, w = img.shape[:2]
        b = it["bbox"]
        x1, y1 = int(b["x"] * w), int(b["y"] * h)
        x2, y2 = int((b["x"] + b["w"]) * w), int((b["y"] + b["h"]) * h)
        crop = img[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size == 0:
            continue
        seq = _normalize(it["plate_text"])
        if seq:
            out.append((crop, seq))
    return out


def _load_synthetic(scenes: Path, limit: int):
    import cv2

    files = sorted(glob.glob(str(scenes / "images" / "*.png")))[:limit]
    out = []
    for f in files:
        stem = Path(f).stem
        parts = stem.split("_", 1)[1].split("-") if "_" in stem else []
        if len(parts) != 4:
            continue
        letter = CODE_TO_PERSIAN.get(parts[1])
        if letter is None:
            continue
        seq = list(parts[0]) + [letter] + list(parts[2]) + list(parts[3])
        img = cv2.imread(f)
        if img is not None:
            out.append((img, seq))
    return out


def _maybe_install_cuda(job: Job) -> str:
    """Return the torch device string, installing a CUDA build if requested."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    job.update(step="installing CUDA PyTorch (this can take a few minutes)")
    job.log("Installing torch CUDA build…")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             "torch", "--index-url", "https://download.pytorch.org/whl/cu121"],
            check=True, capture_output=True, text=True, timeout=1800,
        )
        import importlib

        import torch  # noqa: F811

        importlib.reload(torch)
        if torch.cuda.is_available():
            job.log("CUDA is now available.")
            return "cuda"
        job.log("CUDA build installed but no usable GPU found — using CPU.")
    except Exception as exc:  # noqa: BLE001
        job.log(f"CUDA install failed ({exc}); falling back to CPU.")
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", default="models/ocr_crnn.onnx")
    ap.add_argument("--synthetic", default="")
    ap.add_argument("--synthetic-limit", type=int, default=4000)
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--install-cuda", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    job = Job(Path(args.job))
    job.update(status="running", started=time.strftime("%Y-%m-%dT%H:%M:%S"),
               step="selecting device")
    try:
        import cv2  # noqa: F401

        # 1) Device
        device = "cpu"
        if args.device == "gpu":
            device = _maybe_install_cuda(job) if args.install_cuda else "cpu"
            if device == "cpu":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:  # noqa: BLE001
                    device = "cpu"
        elif args.device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # noqa: BLE001
                device = "cpu"
        job.update(device=device, step="preparing dataset", progress=5)

        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset

        # 2) Dataset
        samples = _load_user_samples(Path(args.samples))
        job.log(f"{len(samples)} user samples")
        if args.synthetic and Path(args.synthetic).exists():
            syn = _load_synthetic(Path(args.synthetic), args.synthetic_limit)
            job.log(f"{len(syn)} synthetic plates")
            # oversample user data so it isn't drowned out
            factor = max(1, len(syn) // max(len(samples), 1) // 8) if samples else 1
            samples = samples * factor + syn
        if not samples:
            job.update(status="error", message="No usable samples. Add labelled images first.")
            return

        classes = sorted({c for _, seq in samples for c in seq})
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        blank = len(classes)
        job.log(f"{len(samples)} training crops · {len(classes)} classes")

        X = np.empty((len(samples), IMG_H, IMG_W), np.float32)
        Y = []
        for i, (crop, seq) in enumerate(samples):
            X[i] = prep_crnn(crop, (IMG_W, IMG_H)).astype(np.float32) / 255.0
            Y.append([cls_to_idx[c] for c in seq])

        rng = np.random.RandomState(1)
        perm = rng.permutation(len(samples))
        split = max(int(len(samples) * 0.92), len(samples) - 200)
        tr, va = perm[:split], perm[split:]

        class DS(Dataset):
            def __init__(self, idxs): self.idxs = idxs
            def __len__(self): return len(self.idxs)
            def __getitem__(self, k):
                i = self.idxs[k]
                return torch.from_numpy(X[i][None].copy()), torch.tensor(Y[i], dtype=torch.long)

        def collate(b):
            imgs = torch.stack([x[0] for x in b])
            tgts = torch.cat([x[1] for x in b])
            lens = torch.tensor([len(x[1]) for x in b], dtype=torch.long)
            return imgs, tgts, lens

        train_dl = DataLoader(DS(tr), batch_size=args.batch, shuffle=True, collate_fn=collate)
        val_dl = DataLoader(DS(va if len(va) else tr[:64]), batch_size=args.batch, collate_fn=collate)

        # 3) Model (CRNN + CTC)
        def blk(i, o, k=3, s=1, p=1):
            return nn.Sequential(nn.Conv2d(i, o, k, s, p), nn.BatchNorm2d(o), nn.ReLU(True))

        class CRNN(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.cnn = nn.Sequential(
                    blk(1, 64), nn.MaxPool2d(2, 2), blk(64, 128), nn.MaxPool2d(2, 2),
                    blk(128, 256), blk(256, 256), nn.MaxPool2d((2, 1), (2, 1)),
                    blk(256, 256), nn.MaxPool2d((2, 1), (2, 1)), blk(256, 256, 2, 1, 0))
                self.rnn = nn.LSTM(256, 128, 2, bidirectional=True, batch_first=True)
                self.fc = nn.Linear(256, n)
            def forward(self, x):
                f = self.cnn(x).squeeze(2).permute(0, 2, 1)
                r, _ = self.rnn(f)
                return self.fc(r)

        dev = torch.device(device)
        model = CRNN(len(classes) + 1).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ctc = nn.CTCLoss(blank=blank, zero_infinity=True)

        def greedy(row):
            idx = row.argmax(1); out, prev = [], -1
            for i in idx.tolist():
                if i != blank and i != prev and i < len(classes):
                    out.append(classes[i])
                prev = i
            return "".join(out)

        out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

        def export():
            model.eval()
            torch.onnx.export(
                model.cpu(), torch.zeros(1, 1, IMG_H, IMG_W), str(out_path),
                input_names=["input"], output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=13, dynamo=False)
            out_path.with_suffix(".labels.json").write_text(
                json.dumps(classes, ensure_ascii=False), encoding="utf-8")
            model.to(dev)

        job.update(step="training", epochs=args.epochs, progress=10)
        best = -1.0
        for ep in range(1, args.epochs + 1):
            model.train()
            for imgs, tgts, lens in train_dl:
                imgs = imgs.to(dev)
                logp = model(imgs).log_softmax(2).permute(1, 0, 2)
                in_len = torch.full((imgs.size(0),), logp.size(0), dtype=torch.long)
                loss = ctc(logp, tgts, in_len, lens)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, tgts, lens in val_dl:
                    logits = model(imgs.to(dev)).cpu()
                    pos = 0
                    for b in range(imgs.size(0)):
                        n = int(lens[b]); gt = "".join(classes[int(t)] for t in tgts[pos:pos+n]); pos += n
                        correct += greedy(logits[b]) == gt; total += 1
            acc = correct / max(total, 1)
            job.update(epoch=ep, accuracy=round(acc, 4),
                       progress=10 + int(85 * ep / args.epochs))
            job.log(f"epoch {ep}/{args.epochs}  loss={loss.item():.3f}  acc={acc:.3f}")
            if acc >= best:
                best = acc
                job.update(step="checkpointing"); export(); job.update(step="training")

        export()
        job.update(status="done", step="complete", progress=100,
                   finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   message=f"Training complete — best accuracy {best:.1%}. Model updated.")
    except Exception as exc:  # noqa: BLE001
        job.update(status="error", message=str(exc))
        job.log(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
