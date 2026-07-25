<div align="center" dir="rtl">

<img src="assets/banner.png" alt="Platrix — تشخیص بلادرنگ پلاک، کاملاً تحت کنترل شما" width="100%" />

# ▣ پلاتریکس (Platrix)

### سامانهٔ بلادرنگ و سِلف‌هاستِ **نظارت تصویری** پلاک خودروهای ایرانی

[![CI](https://github.com/AliAkrami1375/Platrix/actions/workflows/ci.yml/badge.svg)](https://github.com/AliAkrami1375/Platrix/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED)](docker-compose.yml)
[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Model-Dibachain%2FPlatrix-yellow)](https://huggingface.co/Dibachain/Platrix)

[English](README.md) · **فارسی**

</div>

<div dir="rtl">

پلاتریکس هر مجموعه دوربینی را به یک **سامانهٔ نظارتی همیشه‌روشنِ پلاک‌خوان**
تبدیل می‌کند. پلاک‌های ایرانی را به‌صورت بلادرنگ از **عکس**، **فایل ویدیو**،
**وب‌کم** و **دوربین‌های آنلاین RTSP/HTTP** تشخیص و می‌خواند، هر تردد را با زمان و
تصویر برش‌خورده ثبت می‌کند، پلاک‌ها را با **لیست سفید/سیاه** تطبیق می‌دهد، و
می‌گذارد **از روی عکس‌های خودت پلاک‌های جدید را به آن یاد بدهی** — همه از یک
داشبورد سِلف‌هاست با API امن مبتنی بر توکن. بدون ابر، بدون ارسال داده به بیرون؛
تصاویر و داده‌هایت هرگز از سرور خارج نمی‌شوند.

---

## ✨ چه می‌کند

| | |
|---|---|
| 🛰️ **نظارت همیشه‌روشن** | چند دوربین اضافه کن، هرکدام را **همیشه‌روشن** کن؛ موقع بوت خودکار روشن می‌شوند، خودکار وصل می‌مانند و هم‌زمان اجرا می‌شوند، با **وضعیت اتصال زندهٔ** هر دوربین |
| 🌐 **هر منبع، هر شبکه** | وب‌کم، دوربین شبکه RTSP/HTTP، فایل ویدیو یا عکس — پلاتریکس را روی هر شبکه‌ای بگذار و به دوربین‌هایت وصل کن |
| ⚡ **خط پردازش عمیقِ بلادرنگ** | **تشخیص** پلاک با YOLO ← **بهبود کیفیت** ← **خوانش** با CRNN، همه با ONNX Runtime؛ به‌همراه کنترل نرخ فریم و حذف تکراری‌ها |
| 🎯 **دقیق روی عکس واقعی** | مدل روی **کاراکترهای واقعی پلاک ایران** آموزش دیده و کل پلاک را یکجا می‌خواند — بدون تقسیم شکنندهٔ کاراکتر |
| 🛡️ **لیست‌ها و هشدار** | ثبت پلاک با نام در **لیست سفید (مجاز)** یا **سیاه (ممنوع)**؛ موارد منطبق نشانه‌گذاری و هشدار زنده داده می‌شوند |
| 🚦 **ورود / خروج** | هر دوربین را **ورود** یا **خروج** علامت بزن؛ جهت هر تردد ثبت می‌شود |
| 🔎 **جستجو + خروجی CSV** | فیلتر تشخیص‌ها بر اساس پلاک، جهت، لیست و **بازهٔ تاریخ**، و خروجی CSV برای گزارش |
| 🧠 **آموزش از داخل مرورگر** | عکس بارگذاری کن، دور پلاک کادر بکش، پلاک را بنویس و **مدل را در پس‌زمینه آموزش بده** — با progress زنده که با رفرش صفحه از بین نمی‌رود و GPU اختیاری |
| 🔐 **امن به‌صورت پیش‌فرض** | ورود با یوزر/پسورد قابل‌تغییر از UI، به‌همراه **توکن‌های دسترسی API** و **مستندات API** در `/docs` |
| 📱 **داشبورد ریسپانسیو** | روی دسکتاپ ساید‌بار حرفه‌ای، روی موبایل اپ با ناوبری پایین |
| 📦 **کاملاً سِلف‌هاست** | فقط یک `docker compose up`، یا `pip install` و اجرا |

---

## 🏗️ چطور کار می‌کند

خط پردازش سه مرحله دارد، هر مرحله یک مدل ONNX که با **ONNX Runtime** اجرا می‌شود
(برای *اجرا* نیازی به PyTorch/TensorFlow نیست):

۱. **تشخیص** — یک مدل YOLOv8 محل پلاک را در فریم پیدا می‌کند. تشخیص‌های ضعیف/غیرپلاک
   نادیده گرفته می‌شوند تا شلوغی تصویر خوانش اشتباه ندهد.
۲. **بهبود کیفیت** — کراپ پلاک بزرگ‌نمایی، نویزگیری، اصلاح کنتراست و شارپ می‌شود.
   **همین** بهبود هنگام آموزش هم اعمال می‌شود (train==serve)، برای همین واقعاً کمک می‌کند.
۳. **خوانش** — مدل **CRNN + CTC** بدون segmentation کل پلاک را یکجا می‌خواند و در قالب
   استاندارد `دو رقم · حرف · سه رقم · دو رقم` خروجی می‌دهد، مثل `۸۱ و ۶۳۸ ۱۳`.

</div>

```
  دوربین (RTSP/HTTP) ─▶ مدیر چند‌دوربینه ─▶ [تشخیص ─▶ بهبود ─▶ خوانش ─▶ ثبت]
  وب‌کم / فایل / عکس                            │
                                    FastAPI · توکن · REST · WebSocket · MJPEG · SQLite
                                                 │
                                     داشبورد · کلاینت‌های API · خروجی CSV
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

آدرس **http://localhost:8080** را باز کن و وارد شو (پیش‌فرض `admin` / `admin` — از
بخش تنظیمات عوضش کن). داده‌ها، Snapshotها و دیتابیس در `./data` باقی می‌مانند.

### روش دوم — پایتون محلی

</div>

```bash
git clone https://github.com/AliAkrami1375/Platrix.git
cd Platrix
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

huggingface-cli download Dibachain/Platrix \
    ocr_crnn.onnx ocr_crnn.labels.json plate_yolo.onnx --local-dir models/

platrix serve                 # داشبورد روی http://localhost:8080
```

<div dir="rtl">

---

## 🎥 استفاده به‌عنوان سیستم نظارتی

۱. تب **Video Stream** ← **افزودن دوربین**: نام، آدرس استریم (`rtsp://user:pass@host:554/stream`
   یا `0` برای وب‌کم) و جهت لِین را بده. دکمهٔ **Test** یک فریم می‌گیرد تا اتصال را تأیید کند.
۲. کلید **همیشه‌روشن** دوربین را بزن. حالا دائم اجرا می‌شود، اگر شبکه قطع شود خودش
   دوباره وصل می‌شود، و **موقع بوت سرور خودکار روشن می‌شود**. نقطهٔ وضعیت
   `online / reconnecting / error` و FPS زنده را نشان می‌دهد.
۳. هرچند دوربین که خواستی اضافه کن — **هم‌زمان** اجرا می‌شوند. هر پلاک با نام دوربین،
   جهت، اطمینان، زمان و Snapshot ثبت می‌شود.
۴. فید زندهٔ حاشیه‌نویسی‌شده را ببین، تاریخچهٔ **Detections** را جستجو کن (پلاک، جهت،
   لیست یا بازهٔ تاریخ) و برای گزارش **خروجی CSV** بگیر.

---

## 🧠 یاد دادن پلاک جدید (Learn / Train)

تب **Learn** می‌گذارد هر کسی بدون دست‌زدن به کد، مدل را بهتر کند:

۱. **آنوتیشن** — عکس بارگذاری کن، روی canvas دور پلاک کادر بکش و پلاک را دقیق بنویس.
۲. **دیتاست** — نمونه‌های برچسب‌خورده سمت سرور ذخیره می‌شوند.
۳. **آموزش** — دستگاه محاسبه را انتخاب کن (**Auto / CPU / GPU**؛ GPU اختیاری است و در
   صورت اجازهٔ تو می‌تواند نسخهٔ CUDA پای‌تورچ را نصب کند)، بعد **Start training**.
   آموزش **در پس‌زمینه به‌صورت فرآیند مستقل** اجرا می‌شود، پس اگر صفحه را رفرش یا حتی
   ببندی، ادامه پیدا می‌کند. progress زنده (مرحله، درصد، epoch، دقت و لاگ) نمایش داده
   می‌شود و با بازکردن دوباره ادامه می‌یابد.
۴. **Apply** — با یک کلیک مدل تازه‌آموزش‌دیده جایگزین می‌شود.

> درایور GPU در صورت وجود **تشخیص و استفاده** می‌شود؛ پلاتریکس هرگز درایور را به‌زور روی
> سیستم نصب نمی‌کند. `GET /api/system/gpu` وضعیت سخت‌افزار را برمی‌گرداند.

---

## 🔐 امنیت و API

- **ورود** با یوزر/پسورد، قابل‌تغییر از **تنظیمات** (هش‌شده با PBKDF2). برای محیط
  عملیاتی `PLATRIX_AUTH_PASSWORD` و `PLATRIX_SECRET_KEY` را تنظیم کن.
- **API توکن‌محور است.** در تنظیمات **توکن API** بساز و با
  `Authorization: Bearer <token>` صدا بزن.
- **مستندات تعاملی API** (Swagger UI) در **`/docs`** — روی *Authorize* بزن، توکن را
  paste کن و همهٔ endpointها را تست کن.

</div>

```bash
curl -H "Authorization: Bearer pltx_xxx" \
     -F "file=@car.jpg" http://localhost:8080/api/recognize
```

<div dir="rtl">

---

## ⚙️ پیکربندی

هر تنظیم یک متغیر محیطی با پیشوند `PLATRIX_` است (یا فایل `.env`):

| متغیر | پیش‌فرض | توضیح |
|-------|---------|-------|
| `PLATRIX_DETECTOR` | `auto` | `auto` · `yolo` · `contour` |
| `PLATRIX_OCR` | `auto` | `auto` · `crnn` · `onnx` · `none` |
| `PLATRIX_DETECTION_CONFIDENCE` | `0.5` | نادیده‌گرفتن تشخیص‌های ضعیف/غیرپلاک |
| `PLATRIX_AUTH_ENABLED` | `true` | نیاز به ورود برای داشبورد و API |
| `PLATRIX_AUTH_USER` / `PLATRIX_AUTH_PASSWORD` | `admin` / `admin` | **حتماً عوض کن** |
| `PLATRIX_SECRET_KEY` | *(پیش‌فرض)* | کلید امضای کوکی/توکن — یک رشتهٔ تصادفی بلند بگذار |
| `PLATRIX_HOST` / `PLATRIX_PORT` | `0.0.0.0` / `8080` | آدرس و پورت |

---

## 📦 مدل‌ها

مدل‌های آموزش‌دیده روی Hugging Face هستند (در این ریپو قرار ندارند):

**➜ https://huggingface.co/Dibachain/Platrix**

### دانلود مدل‌ها

</div>

```bash
# روش ۱ — با ابزار Hugging Face (پیشنهادی)
pip install -U "huggingface_hub[cli]"
huggingface-cli download Dibachain/Platrix \
    plate_yolo.onnx ocr_crnn.onnx ocr_crnn.labels.json ocr_cnn.onnx ocr_cnn.labels.json \
    --local-dir models/

# روش ۲ — دانلود ساده، بدون ابزار اضافه
base=https://huggingface.co/Dibachain/Platrix/resolve/main
for f in plate_yolo.onnx ocr_crnn.onnx ocr_crnn.labels.json ocr_cnn.onnx ocr_cnn.labels.json; do
    curl -L "$base/$f" -o "models/$f"
done
```

<div dir="rtl">

می‌توانی خودت هم آموزش بدهی (`scripts/train_crnn.py`، `scripts/train_detector.py`،
`scripts/compose_iranis_plates.py`) یا از داخل تب **Learn** — یا مدل را با ایمیل
**[dibachain@gmail.com](mailto:dibachain@gmail.com)** درخواست کن.

تا زمانی که مدلی نباشد، پلاتریکس در حالت «فقط‌تشخیص» کار می‌کند و همچنان پلاک‌ها و
Snapshotها را ثبت می‌کند؛ می‌توانی در داشبورد پلاک را دستی لیبل بزنی.

### تصاویر تست

پوشهٔ [`img-test/`](img-test/) چند عکس واقعی پلاک دارد (روی Hugging Face هم زیر
`img-test/` هست) تا فوری تشخیص را امتحان کنی — در تب **Image Detection** یا با API:

```bash
curl -H "Authorization: Bearer pltx_xxx" \
     -F "file=@img-test/sample-01.jpg" http://localhost:8080/api/recognize
```

<p align="center">
  <img src="img-test/sample-01.jpg" width="32%" />
  <img src="img-test/sample-02.jpg" width="32%" />
  <img src="img-test/sample-03.jpg" width="32%" />
  <img src="img-test/sample-05.jpg" width="32%" />
  <img src="img-test/sample-06.jpg" width="32%" />
  <img src="img-test/sample-08.jpg" width="32%" />
</p>

---

## 📄 مجوز

تحت [مجوز MIT](LICENSE) منتشر شده است.

> **استفادهٔ مسئولانه.** پلاتریکس برای کاربردهای قانونی مانند مدیریت پارکینگ، کنترل
> تردد و تحلیل ترافیک است. رعایت قوانین حریم خصوصی و نظارت تصویری بر عهدهٔ خودت است.

</div>
