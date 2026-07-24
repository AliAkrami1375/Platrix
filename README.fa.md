<div align="center" dir="rtl">

<img src="assets/banner.png" alt="Platrix — تشخیص بلادرنگ پلاک، کاملاً تحت کنترل شما" width="100%" />

# ▣ پلاتریکس (Platrix)

### سامانهٔ بلادرنگ و سِلف‌هاست تشخیص و خواندن پلاک خودروهای ایرانی

[![CI](https://github.com/AliAkrami1375/Platrix/actions/workflows/ci.yml/badge.svg)](https://github.com/AliAkrami1375/Platrix/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED)](docker-compose.yml)

[English](README.md) · **فارسی**

</div>

<div dir="rtl">

پلاتریکس یک موتور **بلادرنگ (Real-time)** برای تشخیص و خواندن پلاک خودروهای ایرانی است
که به‌صورت کاملاً **سِلف‌هاست** روی سرور خودتان اجرا می‌شود. ورودی آن می‌تواند
**عکس**، **فایل ویدیو**، **وب‌کم** یا **دوربین آنلاین (RTSP/HTTP)** باشد و خروجی آن یک
**داشبورد تحت وب**، یک **API از نوع REST و WebSocket** و یک **لاگ زمان‌دار و قابل‌جستجو**
از تمام پلاک‌های شناسایی‌شده است.

---

## ✨ ویژگی‌ها

| | |
|---|---|
| 🎥 **هر منبعی** | وب‌کم، دوربین شبکه (RTSP/HTTP)، فایل ویدیو یا عکس — همه با یک رشتهٔ ساده |
| ⚡ **پردازش بلادرنگ** | خط پردازش چندنخی: دریافت فریم ← تشخیص ← OCR ← ذخیره، همراه با کنترل نرخ فریم و حذف تکراری‌ها |
| 🧠 **موتورهای قابل‌تعویض** | تشخیص: **کانتور** (بدون نیاز به مدل) یا **YOLO** — خواندن: **CNN** یا خاموش |
| 🛡️ **لیست سفید و سیاه** | ثبت پلاک با نام دلخواه در **لیست سفید (مجاز)** یا **لیست سیاه (ممنوع)**؛ موارد منطبق نشانه‌گذاری و هشدار زنده داده می‌شوند |
| 🚦 **ورود / خروج** | هر منبع را می‌توانید به‌عنوان لِین **ورود** یا **خروج** علامت بزنید؛ جهت هر تردد ثبت می‌شود |
| 🔎 **جستجوی رخدادها** | فیلتر تشخیص‌ها بر اساس پلاک، جهت (ورود/خروج) و لیست سفید/سیاه، مستقیم از داخل اپ |
| 🗄️ **ثبت کامل رخدادها** | هر پلاک با زمان دقیق، میزان اطمینان، منبع و یک **تصویر بُرش‌خورده (Snapshot)** در پایگاه‌داده ذخیره می‌شود |
| 📱 **داشبورد موبایل‌محور** | رابط ریسپانسیو شبیه اپلیکیشن موبایل با **ناوبری پایین (Bottom Nav)** — زنده، رخدادها، لیست‌ها و آمار |
| 🔌 **API تمیز** | مجموعه endpointهای REST به‌همراه یک جریان رویداد WebSocket برای اتصال به سامانه‌های شما |
| 📦 **سِلف‌هاست** | تنها با یک `docker compose up` یا نصب با `pip` — بدون ابر، بدون ارسال داده به بیرون |

---

## 🏗️ معماری

مسیر پردازش از سه مرحلهٔ **قابل‌تعویض** تشکیل شده است:

۱. **تشخیص پلاک (Detection)** — پیدا کردن ناحیهٔ پلاک.
   - `contour` *(پیش‌فرض)* — لبه‌یابی Canny به‌همراه فیلتر چندضلعی و نسبت ابعاد و حذف هم‌پوشانی‌ها. **بدون نیاز به هیچ مدل آموزش‌دیده‌ای** کار می‌کند، پس بلافاصله پس از نصب قابل استفاده است.
   - `yolo` — استفاده از YOLO برای تشخیص مقاوم در زاویه، حرکت و شلوغی تصویر (نیازمند وزن‌های آموزش‌دیده).

۲. **جداسازی کاراکتر و خواندن (OCR)** — تقسیم پلاک به کاراکترها و دسته‌بندی آن‌ها.
   - `cnn` — جداسازی کاراکتر با فیلتر هومومورفیک و دسته‌بندی با شبکهٔ CNN، نگاشت‌شده به الفبای پلاک ایران (ارقام `۰–۹` و حروف فارسی).
   - `none` — حالت فقط‌تشخیص (بازهم Snapshot و زمان ثبت می‌شود).

۳. **ذخیره‌سازی (Persistence)** — رخدادهای بدون‌تکرار به‌همراه یک تصویر JPEG بُرش‌خورده در SQLite ذخیره می‌شوند.

</div>

```
   دوربین / RTSP ─────▶  Source  ─▶  Pipeline  ─▶  Storage  ─▶  SQLite
   ویدیو / عکس           (cv2)      تشخیص+OCR     +Snapshots
                                        │
                                        ▼
                          FastAPI ── MJPEG · REST · WebSocket
                                        │
                                 داشبورد و کلاینت‌ها
```

<div dir="rtl">

---

## 🚀 شروع سریع

### روش اول — داکر (پیشنهادی)

</div>

```bash
git clone https://github.com/AliAkrami1375/Platrix.git
cd Platrix
docker compose up --build
```

<div dir="rtl">

سپس آدرس **http://localhost:8080** را باز کنید. داده‌ها و Snapshotها در پوشهٔ `./data` باقی می‌مانند.

### روش دوم — پایتون محلی

</div>

```bash
git clone https://github.com/AliAkrami1375/Platrix.git
cd Platrix
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

platrix serve                 # داشبورد روی http://localhost:8080
```

<div dir="rtl">

> موتور پیش‌فرض `contour` به هیچ وزنی نیاز ندارد. برای OCR با CNN یا تشخیص با YOLO،
> افزونه‌های یادگیری ماشین را هم نصب کنید: `pip install -r requirements-ml.txt`

---

## 💻 نحوهٔ استفاده

### داشبورد وب

با اجرای `platrix serve` داشبوردی باز می‌شود که در آن می‌توانید:

- یک **منبع** وارد کنید (`0`، `rtsp://user:pass@host/stream`، `/path/video.mp4`) و با زدن **Start** جریان زنده را ببینید؛
- یک **عکس را بکشید و رها کنید** تا پلاکش همان‌جا خوانده شود؛
- **جدول رخدادها** را ببینید که به‌صورت زنده و از طریق WebSocket به‌همراه Snapshot به‌روز می‌شود.

### خط فرمان (CLI)

</div>

```bash
platrix run 0                       # وب‌کم پیش‌فرض
platrix run rtsp://cam.local/stream # دوربین شبکه
platrix run drive.mp4 --show        # فایل ویدیو، همراه با پنجرهٔ پیش‌نمایش
platrix run car.jpg                 # یک عکس
platrix events --limit 20           # نمایش آخرین رخدادها
platrix events --plate 12B          # جستجو در لاگ
```

<div dir="rtl">

### اجرای خودکار یک دوربین هنگام بالا آمدن سرویس

</div>

```bash
PLATRIX_DEFAULT_SOURCE="rtsp://user:pass@192.168.1.50:554/stream" platrix serve
```

<div dir="rtl">

---

## 🔌 رابط برنامه‌نویسی (API)

| متد | مسیر | توضیح |
|-----|------|-------|
| `GET`  | `/api/health` | بررسی سلامت و نسخه |
| `GET`  | `/api/status` | وضعیت جریان، FPS، موتورها و آمار |
| `POST` | `/api/stream/start` | شروع یک منبع زنده — `{ "source": "0" }` |
| `POST` | `/api/stream/stop` | توقف منبع فعال |
| `GET`  | `/api/stream/mjpeg` | جریان ویدیوی حاشیه‌نویسی‌شده (MJPEG) |
| `POST` | `/api/recognize` | آپلود عکس ← پلاک‌های شناسایی‌شده + پیش‌نمایش |
| `GET`  | `/api/events?limit=&plate=&direction=&list_type=` | جستجو در لاگ رخدادها |
| `GET`  | `/api/stats` | آمار تجمیعی |
| `GET`  | `/api/watchlist?list_type=` | فهرست پلاک‌های لیست سفید/سیاه |
| `POST` | `/api/watchlist` | افزودن پلاک با نام — `{ "plate", "name", "list_type": "white"\|"black" }` |
| `DELETE` | `/api/watchlist/{id}` | حذف یک پلاک از لیست |
| `WS`   | `/ws/events` | رویدادهای زندهٔ تشخیص به‌صورت JSON |

</div>

```bash
# خواندن پلاک از یک عکس آپلودی
curl -F "file=@car.jpg" http://localhost:8080/api/recognize

# شروع یک دوربین RTSP
curl -X POST http://localhost:8080/api/stream/start \
     -H "Content-Type: application/json" \
     -d '{"source":"rtsp://user:pass@host/stream"}'
```

<div dir="rtl">

---

## ⚙️ پیکربندی

هر تنظیم یک متغیر محیطی با پیشوند `PLATRIX_` است (یا فایل `.env` — به [`.env.example`](.env.example) نگاه کنید):

| متغیر | پیش‌فرض | توضیح |
|-------|---------|-------|
| `PLATRIX_DETECTOR` | `contour` | `contour` یا `yolo` |
| `PLATRIX_OCR` | `cnn` | `cnn` یا `none` |
| `PLATRIX_YOLO_WEIGHTS` | `models/plate_yolo.pt` | وزن‌های تشخیص YOLO |
| `PLATRIX_OCR_WEIGHTS` | `models/ocr_cnn.h5` | وزن‌های OCR |
| `PLATRIX_FRAME_STRIDE` | `1` | پردازش هر N اُمین فریم |
| `PLATRIX_MAX_FPS` | `30` | محدودسازی نرخ خواندن فریم |
| `PLATRIX_DEDUPE_SECONDS` | `4` | جلوگیری از ثبت تکراری یک پلاک |
| `PLATRIX_DEFAULT_SOURCE` | *(خالی)* | اجرای خودکار این منبع هنگام بالا آمدن |
| `PLATRIX_HOST` / `PLATRIX_PORT` | `0.0.0.0` / `8080` | آدرس و پورت سرور |

---

## 🧠 آموزش مدل OCR

موتور OCR مبتنی بر CNN به یک مدل نیاز دارد. آن را از یک دیتاست کاراکتری برچسب‌خورده
(هر کلاس در یک پوشه) آموزش دهید:

</div>

```
dataset/
  0/  1/  2/ … 9/       # کلاس ارقام
  alef/ be/ jim/ …      # کلاس حروف فارسی
```

```bash
python scripts/train_ocr.py --data dataset --epochs 60
# ← models/ocr_cnn.h5  +  models/ocr_cnn.labels.json
```

<div dir="rtl">

فایل برچسب‌ها ترتیب نورون‌های خروجی را نگه می‌دارد تا پلاتریکس به‌صورت خودکار
پیش‌بینی‌ها را به کاراکتر درست نگاشت کند. تا زمانی که مدلی وجود نداشته باشد،
پلاتریکس در حالت «فقط‌تشخیص» کار می‌کند و همچنان پلاک‌ها و Snapshotها را ثبت می‌کند.

---

## 🗂️ ساختار پروژه

</div>

```
platrix/
├── config.py          # تنظیمات مبتنی بر متغیر محیطی
├── core/              # انواع دامنه + خط پردازش
├── detection/         # تشخیص‌گرهای contour و YOLO
├── ocr/               # جداسازی، CNN، و قالب‌بندی پلاک فارسی
├── sources/           # منابع فریم: عکس/ویدیو/وب‌کم/RTSP
├── storage/           # ذخیره‌سازی رخداد و Snapshot در SQLite
├── server/            # اپلیکیشن FastAPI، MJPEG و WebSocket
├── web/               # داشبورد (HTML/CSS/JS)
└── cli.py             # نقطهٔ ورود خط فرمان
scripts/train_ocr.py   # ابزار آموزش OCR
tests/                 # مجموعه تست‌ها
```

<div dir="rtl">

---

## 🧪 توسعه

</div>

```bash
pip install -e .[dev]
ruff check platrix     # بررسی کد
pytest -q              # تست‌ها
```

<div dir="rtl">

---

## 📄 مجوز

این پروژه تحت [مجوز MIT](LICENSE) منتشر شده است.

> **استفادهٔ مسئولانه.** پلاتریکس برای کاربردهای قانونی مانند مدیریت پارکینگ،
> کنترل تردد و تحلیل ترافیک طراحی شده است. رعایت قوانین حریم خصوصی و نظارت تصویری
> که بر شما اعمال می‌شود، بر عهدهٔ خودتان است.

</div>
