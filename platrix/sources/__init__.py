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


__all__ = ["FrameSource", "ImageSource", "VideoSource", "open_source"]
