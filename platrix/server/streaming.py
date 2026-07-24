"""Live stream manager.

Runs a background worker thread that pulls frames from a source, feeds them
through the recognition pipeline, persists de-duplicated readings, keeps the
latest annotated JPEG for MJPEG streaming, and pushes events to subscribers.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from typing import Callable, Optional

import cv2
import numpy as np

from platrix.config import Settings
from platrix.core.pipeline import RecognitionPipeline, annotate
from platrix.logging_conf import get_logger
from platrix.sources import open_source
from platrix.storage import EventStore

logger = get_logger(__name__)

EventCallback = Callable[[dict], None]


class StreamManager:
    """Owns the currently active source and its worker thread."""

    def __init__(self, settings: Settings, store: EventStore) -> None:
        self.settings = settings
        self.store = store
        self.pipeline = RecognitionPipeline(settings)
        self.pipeline.warmup()

        self._thread: Optional[threading.Thread] = None
        self._source = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._placeholder: Optional[bytes] = None
        self._current_spec: Optional[str] = None
        self._direction: str = "unknown"
        self._fps: float = 0.0
        self._subscribers: list[EventCallback] = []
        self._recent_events: deque[dict] = deque(maxlen=50)

    # -- subscriptions ----------------------------------------------------
    def subscribe(self, callback: EventCallback) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _broadcast(self, event: dict) -> None:
        self._recent_events.append(event)
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception:  # noqa: BLE001 - a bad subscriber must not kill the loop
                logger.exception("Event subscriber raised")

    # -- lifecycle --------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> dict:
        return {
            "running": self.is_running,
            "source": self._current_spec,
            "direction": self._direction,
            "fps": round(self._fps, 1),
            "mode": "unified" if self.pipeline.unified is not None else "two-stage",
            "detector": self.pipeline.detector.name,
            "ocr": self.pipeline.ocr.name,
        }

    def start(self, spec: str, loop: bool = False, direction: str = "unknown") -> None:
        """Start (or restart) the worker on a new source specification."""
        self.stop()
        self._stop.clear()
        self._direction = direction if direction in ("entry", "exit") else "unknown"
        self._source = open_source(spec, loop=loop)
        self._current_spec = spec
        self._thread = threading.Thread(
            target=self._run, name="platrix-stream", daemon=True
        )
        self._thread.start()
        logger.info("Stream started on %s", spec)

    def stop(self) -> None:
        self._stop.set()
        if self._source is not None and hasattr(self._source, "stop"):
            self._source.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        self._source = None
        self._current_spec = None

    # -- worker -----------------------------------------------------------
    def _run(self) -> None:
        stride = max(self.settings.frame_stride, 1)
        min_interval = 1.0 / self.settings.max_fps if self.settings.max_fps > 0 else 0.0
        last_ts = time.monotonic()
        try:
            for frame in self._source.frames():
                if self._stop.is_set():
                    break

                now = time.monotonic()
                if min_interval and (now - last_ts) < min_interval:
                    continue
                dt = now - last_ts
                last_ts = now
                self._fps = 1.0 / dt if dt > 0 else self._fps

                if frame.index % stride != 0:
                    self._encode(frame.image)
                    continue

                readings = self.pipeline.process(frame)
                for reading in readings:
                    if self.pipeline.is_duplicate(reading):
                        continue
                    event = self.store.record(reading, direction=self._direction)
                    self._broadcast(event.to_dict())

                self._encode(annotate(frame.image, readings))
        except Exception:  # noqa: BLE001
            logger.exception("Stream worker crashed")
        finally:
            logger.info("Stream worker stopped")

    def _encode(self, image) -> None:
        ok, buf = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality]
        )
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

    # -- MJPEG stream -----------------------------------------------------
    def _placeholder_jpeg(self) -> bytes:
        """A 'no signal' frame so the stream never hangs with an empty body."""
        if self._placeholder is None:
            img = np.zeros((360, 640, 3), dtype="uint8")
            img[:] = (18, 22, 30)
            text = "NO ACTIVE SOURCE"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(
                img, text, ((640 - tw) // 2, (360 + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 130, 150), 2, cv2.LINE_AA,
            )
            ok, buf = cv2.imencode(".jpg", img)
            self._placeholder = buf.tobytes() if ok else b""
        return self._placeholder

    def mjpeg(self) -> Iterator[bytes]:
        """Yield a multipart MJPEG byte stream of the latest annotated frames.

        Always emits a frame (a placeholder when idle) so the client never sits
        on an empty, never-closing response.
        """
        boundary = b"--frame\r\n"
        while True:
            with self._lock:
                jpeg = self._latest_jpeg
            if jpeg is None:
                jpeg = self._placeholder_jpeg()
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.05)

    @property
    def recent_events(self) -> list[dict]:
        return list(self._recent_events)
