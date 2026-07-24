"""Frame sources: images, video files, webcams and network cameras."""

from __future__ import annotations

from pathlib import Path

from platrix.sources.base import FrameSource
from platrix.sources.image import ImageSource
from platrix.sources.video import VideoSource


def open_source(spec: str, *, loop: bool = False) -> FrameSource:
    """Open a frame source from a string specification.

    Accepts:

    * ``"0"`` / ``"1"`` …           → local webcam index
    * ``"rtsp://…"`` / ``"http://…"`` → network camera stream
    * a path to an image             → single-frame image source
    * a path to a video file         → video source
    """
    spec = spec.strip()
    if spec.isdigit():
        return VideoSource(int(spec), loop=False)
    if spec.startswith(("rtsp://", "http://", "https://", "udp://", "tcp://")):
        return VideoSource(spec, loop=loop)

    path = Path(spec)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if path.suffix.lower() in image_exts:
        return ImageSource(path)
    return VideoSource(str(path), loop=loop)


def test_source(spec: str, timeout: float = 8.0):
    """Try to open *spec* and grab a single frame.

    Returns ``(ok, message, frame_bgr_or_None)``. Never raises — a failure is
    reported through the return value so the API can surface a friendly error.
    """
    import time

    import cv2

    spec = (spec or "").strip()
    if not spec:
        return False, "Empty source", None

    target: "int | str" = int(spec) if spec.isdigit() else spec
    cap = cv2.VideoCapture(target)
    try:
        deadline = time.monotonic() + timeout
        if not cap.isOpened():
            return False, "Could not open the source (bad URL or unreachable)", None
        frame = None
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                return True, f"Connected · {w}x{h}", frame
        return False, "Opened but no frame received (timeout)", None
    finally:
        cap.release()


__all__ = ["FrameSource", "ImageSource", "VideoSource", "open_source", "test_source"]
