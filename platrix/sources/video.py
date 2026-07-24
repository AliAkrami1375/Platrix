"""Video / webcam / network-camera frame source (OpenCV VideoCapture)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime, timezone

import cv2

from platrix.core.types import Frame
from platrix.logging_conf import get_logger

logger = get_logger(__name__)


class VideoSource:
    """Reads frames from anything OpenCV's :class:`cv2.VideoCapture` accepts.

    That includes local video files, webcam indices (``0``, ``1`` …) and network
    streams (``rtsp://``, ``http://`` MJPEG, …). For network streams the source
    transparently reconnects on transient read failures.
    """

    name = "video"

    def __init__(
        self,
        target: "int | str",
        *,
        loop: bool = False,
        reconnect_delay: float = 2.0,
        max_reconnects: int = 30,
    ) -> None:
        self.target = target
        self.loop = loop
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self._cap: cv2.VideoCapture | None = None
        self._stopped = False

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.target)
        if not cap.isOpened():
            raise ConnectionError(f"Unable to open video source: {self.target!r}")
        # Keep only the freshest frame for live streams (reduce latency).
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:  # pragma: no cover - not all backends support it
            pass
        return cap

    @property
    def is_live(self) -> bool:
        return isinstance(self.target, int) or (
            isinstance(self.target, str)
            and self.target.startswith(("rtsp://", "http://", "https://", "udp://", "tcp://"))
        )

    def frames(self) -> Iterator[Frame]:
        self._cap = self._open()
        index = 0
        reconnects = 0
        source_label = f"video:{self.target}"

        while not self._stopped:
            ok, image = self._cap.read()
            if not ok:
                if self.loop and not self.is_live:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self.is_live and reconnects < self.max_reconnects:
                    reconnects += 1
                    logger.warning(
                        "Stream read failed — reconnecting (%d/%d) in %.1fs",
                        reconnects,
                        self.max_reconnects,
                        self.reconnect_delay,
                    )
                    self._cap.release()
                    time.sleep(self.reconnect_delay)
                    try:
                        self._cap = self._open()
                    except ConnectionError:
                        continue
                    continue
                break

            reconnects = 0
            yield Frame(
                image=image,
                index=index,
                timestamp=datetime.now(timezone.utc),
                source=source_label,
            )
            index += 1

        self.close()

    def stop(self) -> None:
        self._stopped = True

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
