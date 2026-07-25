"""Multi-camera monitoring.

Runs several cameras at once, each in its own worker thread with its own
recognition pipeline. Cameras marked *enabled* start automatically when the
server boots and keep themselves connected (auto-reconnect), so Platrix acts as
an always-on surveillance system. Each worker exposes a live connection status
and an annotated MJPEG feed.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from datetime import datetime, timezone
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
ADHOC_ID = 0  # reserved id for the ad-hoc "view a URL" worker


class CameraWorker:
    """One camera: reads frames, recognizes plates, tracks connection status."""

    def __init__(
        self,
        settings: Settings,
        store: EventStore,
        cam_id: int,
        name: str,
        url: str,
        direction: str,
        broadcast: EventCallback,
        loop: bool = False,
    ) -> None:
        self.settings = settings
        self.store = store
        self.cam_id = cam_id
        self.name = name
        self.url = url
        self.direction = direction
        self._broadcast = broadcast
        self._loop = loop

        self.pipeline = RecognitionPipeline(settings)
        self.pipeline.warmup()

        self._thread: Optional[threading.Thread] = None
        self._source = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._placeholder: Optional[bytes] = None
        self._fps = 0.0
        self._last_frame: Optional[float] = None
        self._error: Optional[str] = None

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.cam_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._source is not None and hasattr(self._source, "stop"):
            self._source.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def state(self) -> str:
        if not self.alive:
            return "error" if self._error else "stopped"
        if self._last_frame is None:
            return "connecting"
        if time.monotonic() - self._last_frame > 4.0:
            return "reconnecting"
        return "online"

    def status(self) -> dict:
        return {
            "id": self.cam_id,
            "name": self.name,
            "state": self.state,
            "fps": round(self._fps, 1),
            "error": self._error,
            "last_seen": (
                datetime.fromtimestamp(self._wall_last, timezone.utc).isoformat()
                if getattr(self, "_wall_last", None) else None
            ),
        }

    # -- worker loop ------------------------------------------------------
    def _run(self) -> None:
        min_interval = 1.0 / self.settings.max_fps if self.settings.max_fps > 0 else 0.0
        stride = max(self.settings.frame_stride, 1)
        last_ts = time.monotonic()
        try:
            self._source = open_source(self.url, loop=self._loop)
            for frame in self._source.frames():
                if self._stop.is_set():
                    break
                now = time.monotonic()
                if min_interval and (now - last_ts) < min_interval:
                    continue
                dt = now - last_ts
                last_ts = now
                self._fps = 1.0 / dt if dt > 0 else self._fps
                self._last_frame = now
                self._wall_last = time.time()

                if frame.index % stride != 0:
                    self._encode(frame.image)
                    continue

                readings = self.pipeline.process(frame)
                for reading in readings:
                    if self.pipeline.is_duplicate(reading):
                        continue
                    reading.source = f"{self.name}"
                    event = self.store.record(reading, direction=self.direction)
                    self._broadcast(event.to_dict())
                self._encode(annotate(frame.image, readings))
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            logger.warning("Camera %s (%s) stopped: %s", self.cam_id, self.name, exc)
        finally:
            if self._source is not None:
                self._source.close()

    def _encode(self, image) -> None:
        ok, buf = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality]
        )
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

    def _placeholder_jpeg(self) -> bytes:
        if self._placeholder is None:
            img = np.full((360, 640, 3), (18, 22, 30), dtype="uint8")
            msg = f"{self.name}: {self.state}"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(img, msg, ((640 - tw) // 2, (360 + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 130, 150), 2, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", img)
            self._placeholder = buf.tobytes() if ok else b""
        # placeholder text is dynamic; regenerate lightly when disconnected
        if self.state not in ("online", "connecting"):
            img = np.full((360, 640, 3), (18, 22, 30), dtype="uint8")
            msg = f"{self.name}: {self.state}"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(img, msg, ((640 - tw) // 2, (360 + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 130, 150), 2, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", img)
            if ok:
                return buf.tobytes()
        return self._placeholder

    def mjpeg(self) -> Iterator[bytes]:
        boundary = b"--frame\r\n"
        while True:
            with self._lock:
                jpeg = self._latest_jpeg
            if jpeg is None:
                jpeg = self._placeholder_jpeg()
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.05)


class MultiStreamManager:
    """Owns all running camera workers (persistent cameras + the ad-hoc view)."""

    def __init__(self, settings: Settings, store: EventStore) -> None:
        self.settings = settings
        self.store = store
        self._workers: dict[int, CameraWorker] = {}
        self._lock = threading.Lock()
        self._subscribers: list[EventCallback] = []
        self._recent_events: deque[dict] = deque(maxlen=50)
        # A throwaway pipeline just to report engine names in /api/status.
        self._probe = RecognitionPipeline(settings)

    # -- subscriptions ----------------------------------------------------
    def subscribe(self, cb: EventCallback) -> None:
        with self._lock:
            self._subscribers.append(cb)

    def unsubscribe(self, cb: EventCallback) -> None:
        with self._lock:
            if cb in self._subscribers:
                self._subscribers.remove(cb)

    def _broadcast(self, event: dict) -> None:
        self._recent_events.append(event)
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception:  # noqa: BLE001
                logger.exception("event subscriber failed")

    @property
    def recent_events(self) -> list[dict]:
        return list(self._recent_events)

    # -- camera control ---------------------------------------------------
    def start_camera(self, cam: dict, loop: bool = True) -> None:
        # loop=True keeps file-based cameras alive (RTSP/live streams ignore it
        # and auto-reconnect instead) — i.e. always-on monitoring.
        cid = cam["id"]
        self.stop_camera(cid)
        worker = CameraWorker(
            self.settings, self.store, cid, cam.get("name") or cam["url"],
            cam["url"], cam.get("direction", "unknown"), self._broadcast, loop=loop,
        )
        worker.start()
        with self._lock:
            self._workers[cid] = worker
        logger.info("Camera started: %s (%s)", cam.get("name"), cam["url"])

    def stop_camera(self, cam_id: int) -> None:
        with self._lock:
            worker = self._workers.pop(cam_id, None)
        if worker:
            worker.stop()

    def start_adhoc(self, url: str, direction: str = "unknown", loop: bool = False) -> None:
        self.start_camera(
            {"id": ADHOC_ID, "name": f"view:{url}", "url": url, "direction": direction},
            loop=loop,
        )

    def start_enabled(self, cameras: list[dict]) -> None:
        for cam in cameras:
            if cam.get("enabled"):
                try:
                    self.start_camera(cam)
                except Exception:  # noqa: BLE001
                    logger.exception("failed to auto-start camera %s", cam.get("name"))

    def stop_all(self) -> None:
        for cid in list(self._workers):
            self.stop_camera(cid)

    # -- status / streams -------------------------------------------------
    def statuses(self) -> dict[int, dict]:
        with self._lock:
            return {cid: w.status() for cid, w in self._workers.items()}

    def worker(self, cam_id: int) -> Optional[CameraWorker]:
        return self._workers.get(cam_id)

    def any_running(self) -> bool:
        return any(w.alive for w in self._workers.values())

    def overall_status(self) -> dict:
        running = [w for w in self._workers.values() if w.alive]
        return {
            "running": bool(running),
            "active_cameras": len(running),
            "fps": round(max((w._fps for w in running), default=0.0), 1),
            "detector": self._probe.detector.name if self._probe.unified is None else "unified",
            "ocr": self._probe.ocr.name,
        }

    def mjpeg(self, cam_id: int = ADHOC_ID) -> Iterator[bytes]:
        worker = self._workers.get(cam_id)
        if worker is None:
            # nothing running for this id → a small idle stream
            idle = _idle_jpeg()
            while True:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + idle + b"\r\n"
                time.sleep(0.2)
        yield from worker.mjpeg()

    # convenience for one-shot recognition (image upload) reuses a pipeline
    @property
    def pipeline(self) -> RecognitionPipeline:
        return self._probe


def _idle_jpeg() -> bytes:
    img = np.full((360, 640, 3), (18, 22, 30), dtype="uint8")
    cv2.putText(img, "NO ACTIVE CAMERA", (150, 190), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (120, 130, 150), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""
