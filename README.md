<div align="center">

<img src="assets/banner.png" alt="Platrix — Real-Time License Plate Recognition, Fully Under Your Control" width="100%" />

# ▣ Platrix

### Real-time, self-hosted Automatic License Plate Recognition (ALPR) for Iranian plates

[![CI](https://github.com/AliAkrami1375/Platrix/actions/workflows/ci.yml/badge.svg)](https://github.com/AliAkrami1375/Platrix/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED)](docker-compose.yml)

**English** · [فارسی](README.fa.md)

Detect and read Iranian vehicle license plates from **images**, **video files**,
**webcams** and **online (RTSP/HTTP) cameras** — in real time — with a built-in
web dashboard, a REST + WebSocket API, and a searchable, timestamped log of every
plate it sees.

</div>

---

## ✨ Highlights

| | |
|---|---|
| 🎥 **Any source** | Webcam, RTSP/HTTP IP cameras, video files, or single images — auto-detected from one string |
| ⚡ **Real-time pipeline** | Threaded capture → detect → OCR → persist, with frame-striding, FPS throttling and duplicate suppression |
| 🧠 **Pluggable engines** | Detection: classic **contour** (zero weights, works instantly) or **YOLO**. OCR: **CNN** or none |
| 🗄️ **Full audit trail** | Every read stored in SQLite with UTC timestamp, confidence, source and a cropped **snapshot** image |
| 🌐 **Web dashboard** | Live annotated MJPEG feed, drag-in image recognition, and a live-updating detection table |
| 🔌 **Clean API** | REST endpoints + a WebSocket event stream for integration with your own systems |
| 📦 **Self-hosted** | One `docker compose up`, or `pip install` and run. No cloud, no external calls — your data stays yours |

---

## 🏗️ Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │                  Platrix                      │
                       │                                               │
  Webcam / RTSP  ─────▶│  Source  ─▶  Pipeline  ─▶  Storage  ─▶  SQLite │
  Video / Image        │  (cv2)      detect+OCR     +snapshots          │
                       │                │                               │
                       │                ▼                               │
                       │   FastAPI  ──  MJPEG stream · REST · WebSocket │
                       └───────────────────────┬──────────────────────┘
                                                │
                                        Web dashboard / clients
```

The recognition flow has three swappable stages:

1. **Detection** — locate the plate region.
   - `contour` *(default)* — OpenCV Canny edges + polygon/aspect-ratio filtering + non-max suppression. **Requires no trained model**, so Platrix runs the moment it's installed.
   - `yolo` — Ultralytics YOLO for robust detection under angle, motion and clutter (bring your own weights).
2. **Segmentation + OCR** — split the plate into characters and classify them.
   - `cnn` — homomorphic-filter character segmentation feeding a Keras CNN, mapped to the Iranian plate alphabet (digits `0–9` + Persian letters).
   - `none` — detection-only mode (still logs snapshots and timestamps).
3. **Persistence** — de-duplicated readings are written to SQLite with a cropped JPEG snapshot per event.

---

## 🚀 Quick start

### Option A — Docker (recommended)

```bash
git clone https://github.com/AliAkrami1375/Platrix.git
cd Platrix
docker compose up --build
```

Open **http://localhost:8080**. Data and snapshots persist in `./data`.

### Option B — Local Python

```bash
git clone https://github.com/AliAkrami1375/Platrix.git
cd Platrix
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

platrix serve                 # dashboard on http://localhost:8080
```

> The default `contour` detector needs no weights. For CNN OCR or YOLO, also
> install the ML extras: `pip install -r requirements-ml.txt`.

---

## 💻 Usage

### Web dashboard

`platrix serve` launches the dashboard where you can:

- paste a **source** (`0`, `rtsp://user:pass@host/stream`, `/path/video.mp4`) and hit **Start** to watch the live annotated feed;
- **drag in an image** to recognize a plate on the spot;
- watch the **detection table** populate live over WebSocket, with snapshots.

### Command line

```bash
platrix run 0                       # default webcam
platrix run rtsp://cam.local/stream # network camera
platrix run drive.mp4 --show        # video file, with a preview window
platrix run car.jpg                 # a single image
platrix events --limit 20           # print the latest detections
platrix events --plate 12B          # search the log
```

### Auto-start a camera on boot

```bash
PLATRIX_DEFAULT_SOURCE="rtsp://user:pass@192.168.1.50:554/stream" platrix serve
```

---

## 🔌 API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health` | Liveness + version |
| `GET`  | `/api/status` | Stream state, FPS, engines, stats |
| `POST` | `/api/stream/start` | `{ "source": "0", "loop": false }` — start a live source |
| `POST` | `/api/stream/stop` | Stop the active source |
| `GET`  | `/api/stream/mjpeg` | Annotated MJPEG video stream |
| `POST` | `/api/recognize` | Multipart image upload → detected plates + annotated preview |
| `GET`  | `/api/events?limit=&plate=` | Query the detection log |
| `GET`  | `/api/stats` | Aggregate statistics |
| `WS`   | `/ws/events` | Live detection events as JSON |

```bash
# Recognize an uploaded image
curl -F "file=@car.jpg" http://localhost:8080/api/recognize

# Start an RTSP camera
curl -X POST http://localhost:8080/api/stream/start \
     -H "Content-Type: application/json" \
     -d '{"source":"rtsp://user:pass@host/stream"}'
```

---

## ⚙️ Configuration

Every setting is an environment variable with the `PLATRIX_` prefix (or a `.env`
file — see [`.env.example`](.env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `PLATRIX_DETECTOR` | `contour` | `contour` or `yolo` |
| `PLATRIX_OCR` | `cnn` | `cnn` or `none` |
| `PLATRIX_YOLO_WEIGHTS` | `models/plate_yolo.pt` | YOLO detector weights |
| `PLATRIX_OCR_WEIGHTS` | `models/ocr_cnn.h5` | CNN OCR weights |
| `PLATRIX_FRAME_STRIDE` | `1` | Process every Nth frame |
| `PLATRIX_MAX_FPS` | `30` | Throttle source read rate |
| `PLATRIX_DEDUPE_SECONDS` | `4` | Suppress repeated logs of the same plate |
| `PLATRIX_DEFAULT_SOURCE` | *(empty)* | Auto-start this source on boot |
| `PLATRIX_HOST` / `PLATRIX_PORT` | `0.0.0.0` / `8080` | Server bind address |

---

## 🧠 Training the OCR model

The CNN OCR backend needs a model. Train one from a labelled character dataset
(one folder per class):

```
dataset/
  0/  1/  2/ … 9/       # digit classes
  alef/ be/ jim/ …      # Persian letter classes
```

```bash
python scripts/train_ocr.py --data dataset --epochs 60
# → models/ocr_cnn.h5  +  models/ocr_cnn.labels.json
```

The label file records the output-neuron order so Platrix maps predictions back
to the correct characters automatically. Until a model exists, Platrix runs in
detection-only mode and still logs plates + snapshots.

---

## 🗂️ Project layout

```
platrix/
├── config.py          # environment-driven settings
├── core/              # domain types + recognition pipeline
├── detection/         # contour + YOLO detectors (pluggable)
├── ocr/               # segmentation, CNN OCR, Persian plate formatting
├── sources/           # image / video / webcam / RTSP frame sources
├── storage/           # SQLite event store + snapshot writer
├── server/            # FastAPI app, MJPEG streaming, WebSocket
├── web/               # dashboard (HTML/CSS/JS)
└── cli.py             # `platrix` command-line entrypoint
scripts/train_ocr.py   # OCR training utility
tests/                 # pytest suite
```

---

## 🧪 Development

```bash
pip install -e .[dev]
ruff check platrix     # lint
pytest -q              # tests
```

---

## 🗺️ Roadmap

- [ ] Pre-trained plate + OCR weights as downloadable releases
- [ ] Multi-camera fan-out with per-source event streams
- [ ] Optional plate anonymization / retention policies
- [ ] Prometheus metrics endpoint

---

## 📄 License

Released under the [MIT License](LICENSE).

> **Responsible use.** Platrix is intended for lawful applications such as
> parking management, access control and traffic analytics. You are responsible
> for complying with the privacy and surveillance laws that apply to you.
