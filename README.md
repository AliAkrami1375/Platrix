<div align="center">

<img src="assets/banner.png" alt="Platrix — Real-Time License Plate Recognition, Fully Under Your Control" width="100%" />

# ▣ Platrix

### Real-time, self-hosted license-plate **surveillance** for Iranian plates

[![CI](https://github.com/AliAkrami1375/Platrix/actions/workflows/ci.yml/badge.svg)](https://github.com/AliAkrami1375/Platrix/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED)](docker-compose.yml)
[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Model-Dibachain%2FPlatrix-yellow)](https://huggingface.co/Dibachain/Platrix)

**English** · [فارسی](README.fa.md)

Platrix turns any set of cameras into an **always-on license-plate monitoring
system**. It detects and reads Iranian plates in real time from **images**,
**video files**, **webcams** and **online RTSP/HTTP cameras**, logs every read
with a timestamp and a cropped snapshot, matches plates against **white / black
lists**, and lets you **teach it new plates from your own photos** — all from a
self-hosted web dashboard with a token-secured REST + WebSocket API. No cloud,
no third-party calls: your footage and data never leave your server.

</div>

---

## ✨ What it does

| | |
|---|---|
| 🛰️ **Always-on surveillance** | Add many cameras, flip each to **always-on**; they auto-start on boot, auto-reconnect, and run concurrently with a live per-camera connection status |
| 🌐 **Any source, any network** | Webcam, RTSP/HTTP IP cameras, video files or single images — drop Platrix onto any network and point it at your cameras |
| ⚡ **Real-time deep pipeline** | YOLO plate **detector** → image-quality **enhancement** → segmentation-free **CRNN reader**, all via ONNX Runtime; frame-striding, FPS throttling and duplicate suppression |
| 🎯 **Accurate on real photos** | The reader is trained on **real Iranian plate characters** and reads the whole plate at once — no fragile character splitting |
| 🛡️ **Watchlists & alerts** | Register named plates on a **whitelist** or **blacklist**; matches are flagged and alerted live |
| 🚦 **Entry / exit lanes** | Tag a camera as an **entry** or **exit**; every read is logged with its direction |
| 🔎 **Searchable log + CSV export** | Filter detections by plate, direction, list or **date range**, and export the results to CSV |
| 🧠 **Learn / train in the browser** | Upload photos, draw a box around the plate, type the plate, and **train the model in the background** — with live progress that survives a page refresh, and optional GPU |
| 🔐 **Secure by default** | Login with a username/password you can change from the UI, plus **API access tokens** for programmatic use; interactive **API docs** at `/docs` |
| 📱 **Responsive dashboard** | Desktop sidebar app on the big screen, mobile app with bottom navigation on phones |
| 📦 **Truly self-hosted** | One `docker compose up`, or `pip install` and run |

---

## 🏗️ How it works

```
                         ┌──────────────────────────────────────────────┐
  Cameras (RTSP/HTTP) ──▶│  Multi-camera manager                        │
  Webcams / files        │   └─ per camera: read ─▶ detect ─▶ enhance ─▶ │
  Uploaded images        │                          read ─▶ persist      │
                         │                                               │
                         │  FastAPI · token auth · REST · WebSocket ·    │
                         │  MJPEG streams · SQLite + snapshots           │
                         └───────────────────────┬──────────────────────┘
                                                 │
                              Web dashboard · API clients · CSV export
```

The recognition pipeline has three stages, each a portable ONNX model run with
**ONNX Runtime** (no PyTorch/TensorFlow needed to *run* Platrix):

1. **Detect** — a YOLOv8 model locates the plate in the frame. Weak / non-plate
   detections are ignored, so clutter doesn't produce false reads.
2. **Enhance** — the plate crop is upscaled, denoised, contrast-corrected and
   sharpened. The *same* enhancement is applied during training, so there is no
   train/serve mismatch (this is what makes the enhancement actually help).
3. **Read** — a segmentation-free **CRNN + CTC** model reads the entire plate at
   once and outputs the standard layout `DD L DDD DD`
   (two digits · letter · three digits · two-digit region), e.g. `۸۱ و ۶۳۸ ۱۳`.

A weights-free classic detector and a per-character OCR are included as
fallbacks; `auto` selects the best available.

---

## 🚀 Quick start

### Option A — Docker (recommended)

```bash
git clone https://github.com/AliAkrami1375/Platrix.git
cd Platrix
# fetch the trained models (see "Models" below)
docker compose up --build
```

Open **http://localhost:8080** and sign in (default `admin` / `admin` — change it
in **Settings**). Data, snapshots and the database persist in `./data`.

### Option B — Local Python

```bash
git clone https://github.com/AliAkrami1375/Platrix.git
cd Platrix
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

huggingface-cli download Dibachain/Platrix \
    ocr_crnn.onnx ocr_crnn.labels.json plate_yolo.onnx --local-dir models/

platrix serve                 # dashboard on http://localhost:8080
```

> Running Platrix needs only the lightweight `requirements.txt` (ONNX Runtime).
> **Training** (the Learn tab / scripts) additionally needs PyTorch — see below.

---

## 🎥 Using Platrix as a surveillance system

1. **Video Stream** tab → **Add a camera**: give it a name, paste the stream URL
   (`rtsp://user:pass@host:554/stream`, an HTTP MJPEG URL, or `0` for a webcam),
   and pick a lane direction. **Test** grabs a preview frame to confirm the link.
2. Flip the camera's **always-on** switch. It now runs continuously, reconnects
   itself if the network drops, and **auto-starts whenever the server boots**.
   The status dot shows `online / reconnecting / error` and its live FPS.
3. Add as many cameras as you need — they run **concurrently**. Every plate is
   logged with the camera's name, direction, confidence, timestamp and a
   cropped snapshot.
4. Watch the live annotated feed, search the **Detections** history (by plate,
   direction, list or date range), and **export to CSV** for reporting.

---

## 🧠 Teach it new plates (Learn / Train)

The **Learn** tab lets anyone improve the model without touching the code:

1. **Annotate** — upload a photo, drag a box around the plate on the canvas, and
   type the plate exactly.
2. **Build a dataset** — your labelled samples are stored server-side.
3. **Train** — choose the compute device (**Auto / CPU / GPU**; GPU is opt-in and
   can install a CUDA build of PyTorch if you allow it), then **Start training**.
   The job runs **in the background as a detached process**, so it keeps going
   even if you refresh or close the tab. Live progress (step, %, epoch, accuracy
   and a log) is shown and resumes on reload.
4. **Apply** — one click hot-swaps the freshly trained model into recognition.

> GPU drivers are **detected and used** when present; Platrix never force-installs
> drivers on your host. `GET /api/system/gpu` reports what was found.

---

## 🔐 Security & API

- **Login** with a username/password, changeable from **Settings**
  (PBKDF2-hashed, stored in the DB). Set `PLATRIX_AUTH_PASSWORD` /
  `PLATRIX_SECRET_KEY` for production.
- **The API is token-based.** Create named **API tokens** in Settings and call
  the API with `Authorization: Bearer <token>`. The login cookie only authorizes
  the browser's own MJPEG/snapshot `<img>` requests.
- **Interactive API docs** (Swagger UI) at **`/docs`** — click *Authorize*, paste
  a token, and try every endpoint. Great for integrating or presenting the API.

```bash
# Recognize an uploaded image with an API token
curl -H "Authorization: Bearer pltx_xxx" \
     -F "file=@car.jpg" http://localhost:8080/api/recognize
```

### REST / WebSocket reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/login` · `/api/logout` · `GET /api/me` | Session auth → returns a token |
| `POST` | `/api/account/password` | Change username / password |
| `GET/POST/DELETE` | `/api/tokens` | Manage API access tokens |
| `GET`  | `/api/status` · `/api/health` | Engine, cameras, stats, liveness |
| `POST` | `/api/recognize` | Image upload → detected plates + annotated preview |
| `GET/POST/PATCH/DELETE` | `/api/cameras` | Manage cameras; `PATCH {enabled}` arms always-on |
| `POST` | `/api/cameras/test` | Test a stream URL → `{ ok, message, preview }` |
| `POST` | `/api/stream/start` · `/api/stream/stop` · `GET /api/stream/mjpeg?camera=` | Live view |
| `GET`  | `/api/events?plate=&direction=&list_type=&date_from=&date_to=` | Search the log |
| `GET`  | `/api/events/export` | Download the filtered log as CSV |
| `GET/POST/DELETE` | `/api/watchlist` | White / black list |
| `GET/POST/DELETE` | `/api/learn/samples` · `POST /api/learn/train` · `GET /api/learn/status` · `POST /api/learn/apply` | Learn / train |
| `GET`  | `/api/system/gpu` | GPU detection |
| `WS`   | `/ws/events` | Live detection events as JSON |

---

## ⚙️ Configuration

Every setting is an environment variable with the `PLATRIX_` prefix (or a `.env`
file — see [`.env.example`](.env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `PLATRIX_MODE` | `two-stage` | `two-stage` (detector + reader) or `unified` (experimental single model) |
| `PLATRIX_DETECTOR` | `auto` | `auto` · `yolo` · `contour` |
| `PLATRIX_OCR` | `auto` | `auto` · `crnn` · `onnx` · `cnn` · `none` |
| `PLATRIX_DETECTION_CONFIDENCE` | `0.5` | Ignore weaker (non-plate) detections |
| `PLATRIX_DEDUPE_SECONDS` | `4` | Suppress repeated logs of the same plate |
| `PLATRIX_AUTH_ENABLED` | `true` | Require login for the dashboard & API |
| `PLATRIX_AUTH_USER` / `PLATRIX_AUTH_PASSWORD` | `admin` / `admin` | **Change these** |
| `PLATRIX_SECRET_KEY` | *(dev default)* | Cookie/token signing key — set a long random value |
| `PLATRIX_DEFAULT_SOURCE` | *(empty)* | Auto-view a source on boot |
| `PLATRIX_HOST` / `PLATRIX_PORT` | `0.0.0.0` / `8080` | Bind address |

---

## 📦 Models

The trained models are hosted on Hugging Face (not committed to this repo):

**➜ https://huggingface.co/Dibachain/Platrix**

### Download them

```bash
# Option 1 — Hugging Face CLI (recommended)
pip install -U "huggingface_hub[cli]"
huggingface-cli download Dibachain/Platrix \
    plate_yolo.onnx ocr_crnn.onnx ocr_crnn.labels.json ocr_cnn.onnx ocr_cnn.labels.json \
    --local-dir models/

# Option 2 — plain download, no extra tools
base=https://huggingface.co/Dibachain/Platrix/resolve/main
for f in plate_yolo.onnx ocr_crnn.onnx ocr_crnn.labels.json ocr_cnn.onnx ocr_cnn.labels.json; do
    curl -L "$base/$f" -o "models/$f"
done
```

| File | Role |
|------|------|
| `plate_yolo.onnx` | YOLO plate **detector** |
| `ocr_crnn.onnx` (+ `.labels.json`) | Whole-plate **CRNN reader** (recommended) |
| `ocr_cnn.onnx` (+ `.labels.json`) | Per-character classifier (lightweight fallback) |

Until models are in place, Platrix runs in detection-only mode (still logs
snapshots + timestamps) and you can label plates by hand in the dashboard.

### Try it on the sample images

The [`img-test/`](img-test/) folder ships a few real plate photos (also mirrored
on Hugging Face under `img-test/`) so you can verify recognition immediately:

```bash
curl -H "Authorization: Bearer pltx_xxx" \
     -F "file=@img-test/sample-01.jpg" http://localhost:8080/api/recognize
```

Or just drop one into the **Image Detection** tab.

### Training

Install the training extras first (CPU build shown):

```bash
pip install torch torchvision ultralytics --index-url https://download.pytorch.org/whl/cpu
```

```bash
# Whole-plate CRNN reader — from realistic full-plate images
python scripts/train_crnn.py --data /path/to/plates --epochs 16
# → models/ocr_crnn.onnx

# Compose plates from REAL character crops (fixes look-alikes like 4 vs 6)
python scripts/compose_iranis_plates.py --iranis /path/to/char-dataset --out ds --count 8000

# Plate detector (real photos + Iranian scenes, tighter boxes)
python scripts/train_detector.py --voc /path/to/voc --epochs 30
# → models/plate_yolo.onnx
```

The Learn tab wraps the same training in a friendly, progress-tracked UI, or
email **[dibachain@gmail.com](mailto:dibachain@gmail.com)** to request the models.

---

## 🗂️ Project layout

```
platrix/
├── config.py              # environment-driven settings
├── core/                  # domain types + recognition pipeline
├── detection/             # YOLO + contour detectors
├── ocr/                   # CRNN reader, segmentation, Persian plate formatting
├── preprocessing.py       # image-quality / denoising enhancement layer
├── sources/               # image / video / webcam / RTSP frame sources
├── storage/               # SQLite: events, cameras, watchlist, tokens, samples
├── server/                # FastAPI app, auth, multi-camera manager, GPU detect
├── unified.py             # experimental single-model reader
├── web/                   # dashboard (HTML/CSS/JS)
└── cli.py                 # `platrix` command-line entrypoint
scripts/                   # training utilities (CRNN, detector, composer, learn job)
tests/                     # pytest suite
```

---

## 🧪 Development

```bash
pip install -e .[dev]
ruff check platrix
pytest -q
```

---

## 📄 License

Released under the [MIT License](LICENSE).

> **Responsible use.** Platrix is intended for lawful applications such as
> parking management, access control and traffic analytics. You are responsible
> for complying with the privacy and surveillance laws that apply to you.
